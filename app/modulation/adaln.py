"""
AdaLN - 自适应层归一化

核心机制：情感/场景信号通过调制 LayerNorm 的 scale 和 shift 来深度干预推理

公式: AdaLN(x, e) = γ(e) * RMSNorm(x) + β(e)

与简单向量加法的区别：
- 向量加法: output = x + e  (只影响值)
- AdaLN: output = γ(e) * norm(x) + β(e)  (同时影响分布和值)

位置: 应用在每一层的每一步，实现深度调制
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces.registry import Registry


class RMSNorm(nn.Module):
    """
    RMS Layer Normalization
    
    比 LayerNorm 更高效，不减去均值。
    被 LLaMA 等现代 LLM 广泛采用。
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


@Registry.register("modulation", "adaln")
class AdaptiveLayerNorm(nn.Module):
    """
    自适应层归一化 (AdaLN)
    
    情感/场景信号不是简单加法，而是调制归一化的 scale 和 shift。
    
    公式:
        γ = proj_gamma(condition) + 1  # 初始化为恒等
        β = proj_beta(condition)        # 初始化为 0
        output = γ * RMSNorm(x) + β
    
    使用示例:
        adaln = AdaptiveLayerNorm(d_model=768, d_condition=128)
        
        # 在每层应用
        x_norm = adaln(x, emotion_vector)
    """
    
    def __init__(
        self,
        d_model: int,
        d_condition: int,
        eps: float = 1e-6,
        use_rms: bool = True,
    ):
        """
        Args:
            d_model: 输入特征维度
            d_condition: 条件向量维度
            eps: 数值稳定性
            use_rms: 使用 RMSNorm (True) 或 LayerNorm (False)
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_condition = d_condition
        self.eps = eps
        self.use_rms = use_rms
        
        # 条件投影：从控制向量投影出 scale 和 shift
        self.gamma_proj = nn.Linear(d_condition, d_model)
        self.beta_proj = nn.Linear(d_condition, d_model)
        
        # Zero-Initialization：确保初始时为恒等变换
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)   # γ = 1
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)   # β = 0
    
    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        自适应归一化
        
        Args:
            x: 输入特征 [batch, seq_len, d_model] 或 [batch, d_model]
            condition: 条件向量 [batch, d_condition]
            
        Returns:
            调制后的特征，形状与输入相同
        """
        # 归一化
        if self.use_rms:
            # RMSNorm
            rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            x_norm = x * rms
        else:
            # LayerNorm（手动实现以支持任意形状）
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 从条件向量计算 scale 和 shift
        gamma = self.gamma_proj(condition)  # [batch, d_model]
        beta = self.beta_proj(condition)    # [batch, d_model]
        
        # 处理维度
        if x.dim() == 3:
            # [batch, seq_len, d_model]
            gamma = gamma.unsqueeze(1)  # [batch, 1, d_model]
            beta = beta.unsqueeze(1)    # [batch, 1, d_model]
        
        # 应用 scale 和 shift
        return gamma * x_norm + beta
    
    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_condition={self.d_condition}, use_rms={self.use_rms}"


class AdaLNBlock(nn.Module):
    """
    AdaLN Block - 完整的自适应归一化块
    
    包含：
    - AdaLN
    - 残差连接
    - 可选的 FFN
    
    用于在 Mamba/Transformer 层中嵌入情感调制。
    """
    
    def __init__(
        self,
        d_model: int,
        d_condition: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: 特征维度
            d_condition: 条件维度
            d_ff: FFN 隐藏维度 (None 表示不使用 FFN)
            dropout: Dropout 率
        """
        super().__init__()
        
        self.adaln = AdaptiveLayerNorm(d_model, d_condition)
        
        if d_ff is not None:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
        else:
            self.ffn = None
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: 输入 [batch, seq_len, d_model]
            condition: 条件 [batch, d_condition]
        """
        # AdaLN
        h = self.adaln(x, condition)
        
        # FFN (如果有)
        if self.ffn is not None:
            h = x + self.ffn(h)
        else:
            h = x + self.dropout(h)
        
        return h


class MultiConditionAdaLN(nn.Module):
    """
    多条件 AdaLN
    
    同时接收情感和场景条件，融合后进行调制。
    """
    
    def __init__(
        self,
        d_model: int,
        d_emotion: int,
        d_scene: int,
        fusion_type: str = "add",
    ):
        """
        Args:
            d_model: 特征维度
            d_emotion: 情感向量维度
            d_scene: 场景向量维度
            fusion_type: 融合方式 ('add', 'concat', 'gate')
        """
        super().__init__()
        
        self.fusion_type = fusion_type
        
        if fusion_type == "concat":
            d_condition = d_emotion + d_scene
        elif fusion_type == "gate":
            d_condition = d_emotion
            self.gate = nn.Sequential(
                nn.Linear(d_scene, d_emotion),
                nn.Sigmoid(),
            )
        else:  # add
            d_condition = d_emotion
            self.scene_proj = nn.Linear(d_scene, d_emotion)
        
        self.adaln = AdaptiveLayerNorm(d_model, d_condition)
    
    def forward(
        self,
        x: torch.Tensor,
        emotion: torch.Tensor,
        scene: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: 输入 [batch, seq_len, d_model]
            emotion: 情感条件 [batch, d_emotion]
            scene: 场景条件 [batch, d_scene]
        """
        # 融合条件
        if self.fusion_type == "concat":
            condition = torch.cat([emotion, scene], dim=-1)
        elif self.fusion_type == "gate":
            gate = self.gate(scene)
            condition = gate * emotion
        else:  # add
            condition = emotion + self.scene_proj(scene)
        
        return self.adaln(x, condition)
