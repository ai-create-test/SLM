"""
Hierarchical Dynamics - 层次化动力学模型 (AMHVQ+ 适配)

支持 HierarchicalLatent 和 UnifiedLatent 的动力学预测。
"""

from typing import List, Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces.base_module import BaseModule, ModuleOutput, LatentVector, HierarchicalLatent
from ..interfaces.unified_latent import UnifiedLatent
from ..interfaces.config import ModelConfig
from ..interfaces.registry import Registry


@dataclass
class HierarchicalDynamicsOutput(ModuleOutput):
    """层次化动力学输出"""
    predicted_global: torch.Tensor = None       # 预测的 global [batch, d_latent]
    predicted_chunks: torch.Tensor = None       # 预测的 chunks [batch, num_chunks, d_latent]
    predicted_hierarchical: HierarchicalLatent = None  # 完整的预测结果
    hidden_states: torch.Tensor = None          # 中间隐藏状态
    final_state: Optional[torch.Tensor] = None  # 最终状态


class SetEncoder(nn.Module):
    """
    集合编码器
    
    将 HierarchicalLatent 的多个 chunks 编码为固定长度的表示。
    使用置换不变的方法处理 chunk 集合。
    """
    
    def __init__(
        self,
        d_latent: int = 512,
        d_output: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_latent: 潜空间维度
            d_output: 输出维度
            num_heads: 注意力头数
            dropout: Dropout 率
        """
        super().__init__()
        
        self.d_latent = d_latent
        self.d_output = d_output
        
        # 可学习的聚合 query (用于 attention pooling)
        self.aggregate_query = nn.Parameter(torch.randn(1, 1, d_latent))
        
        # Cross-attention for aggregation
        self.cross_attn = nn.MultiheadAttention(
            d_latent,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # 投影到输出维度
        self.proj = nn.Sequential(
            nn.Linear(d_latent * 2, d_output),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_output, d_output),
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(d_output)
    
    def forward(
        self,
        hierarchical: HierarchicalLatent,
    ) -> torch.Tensor:
        """
        编码 HierarchicalLatent 为固定长度向量
        
        Args:
            hierarchical: HierarchicalLatent
            
        Returns:
            set_encoding: [batch, d_output]
        """
        batch_size = hierarchical.batch_size
        device = hierarchical.device
        
        # 获取 global 和 chunks
        global_vec = hierarchical.global_.squeeze(1)  # [batch, d_latent]
        chunks = hierarchical.chunks  # [batch, num_chunks, d_latent]
        
        # 使用 cross-attention 聚合 chunks
        query = self.aggregate_query.expand(batch_size, -1, -1)
        
        chunk_mask = hierarchical.chunk_mask
        if chunk_mask is not None:
            key_padding_mask = (chunk_mask == 0)
        else:
            key_padding_mask = None
        
        aggregated, _ = self.cross_attn(
            query, chunks, chunks,
            key_padding_mask=key_padding_mask,
        )
        aggregated = aggregated.squeeze(1)  # [batch, d_latent]
        
        # 组合 global 和 aggregated
        combined = torch.cat([global_vec, aggregated], dim=-1)  # [batch, d_latent * 2]
        
        # 投影
        output = self.proj(combined)
        output = self.norm(output)
        
        return output
    
    def encode_sequence(
        self,
        hierarchical_sequence: List[HierarchicalLatent],
    ) -> torch.Tensor:
        """
        编码 HierarchicalLatent 序列
        
        Args:
            hierarchical_sequence: HierarchicalLatent 列表
            
        Returns:
            sequence_encoding: [batch, seq_len, d_output]
        """
        encodings = [self(h) for h in hierarchical_sequence]
        return torch.stack(encodings, dim=1)


@Registry.register("dynamics", "hierarchical")
class HierarchicalDynamics(BaseModule):
    """
    层次化动力学模型 (AMHVQ+)
    
    预测下一个 HierarchicalLatent 或 UnifiedLatent。
    
    流程:
        1. SetEncoder 编码历史 HierarchicalLatent 序列
        2. Dynamics 核心 (Mamba/GRU) 预测中间状态
        3. 解码为 global 和 chunks
    """
    
    MODULE_TYPE = "dynamics"
    
    def __init__(
        self,
        d_latent: int = 512,
        d_model: int = 768,
        num_chunks: int = 8,
        d_state: int = 64,
        num_layers: int = 4,
        brain_type: str = "gru",
        dropout: float = 0.1,
    ):
        """
        Args:
            d_latent: 潜空间维度
            d_model: 内部隐藏维度
            num_chunks: 每个 HierarchicalLatent 的 chunk 数
            d_state: SSM 状态维度
            num_layers: 层数
            brain_type: "mamba" 或 "gru"
            dropout: Dropout 率
        """
        super().__init__()
        
        self.d_latent = d_latent
        self.d_model = d_model
        self.num_chunks = num_chunks
        
        # 集合编码器
        self.set_encoder = SetEncoder(
            d_latent=d_latent,
            d_output=d_model,
            dropout=dropout,
        )
        
        # 动力学核心
        self.brain_type = brain_type
        if brain_type == "gru":
            self.dynamics_core = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
        else:
            # Mamba 需要额外依赖，这里用 GRU 作为替代
            self.dynamics_core = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
        
        # 解码 global
        self.global_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_latent),
        )
        
        # 解码 chunks
        self.chunks_decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, num_chunks * d_latent),
        )
        
        # Layer norms
        self.global_norm = nn.LayerNorm(d_latent)
        self.chunks_norm = nn.LayerNorm(d_latent)
    
    def forward(
        self,
        z_sequence: Union[torch.Tensor, List[HierarchicalLatent], List[UnifiedLatent]],
        hidden: Optional[torch.Tensor] = None,
    ) -> HierarchicalDynamicsOutput:
        """
        前向传播
        
        Args:
            z_sequence: 历史潜向量序列
                - torch.Tensor: [batch, seq_len, d_model] (已编码)
                - List[HierarchicalLatent]: HierarchicalLatent 序列
                - List[UnifiedLatent]: UnifiedLatent 序列
            hidden: 隐藏状态
            
        Returns:
            HierarchicalDynamicsOutput
        """
        # 处理不同输入类型
        if isinstance(z_sequence, torch.Tensor):
            encoded_sequence = z_sequence
        elif isinstance(z_sequence, list):
            if len(z_sequence) == 0:
                raise ValueError("Empty sequence")
            
            first = z_sequence[0]
            if isinstance(first, UnifiedLatent):
                hierarchicals = [u.semantic for u in z_sequence]
            elif isinstance(first, HierarchicalLatent):
                hierarchicals = z_sequence
            else:
                raise ValueError(f"Unsupported type: {type(first)}")
            
            encoded_sequence = self.set_encoder.encode_sequence(hierarchicals)
        else:
            raise ValueError(f"Unsupported input type: {type(z_sequence)}")
        
        batch_size = encoded_sequence.shape[0]
        
        # 动力学核心
        output, new_hidden = self.dynamics_core(encoded_sequence, hidden)
        
        # 取最后一个时间步
        last_hidden = output[:, -1, :]  # [batch, d_model]
        
        # 解码 global
        predicted_global = self.global_decoder(last_hidden)
        predicted_global = self.global_norm(predicted_global)  # [batch, d_latent]
        
        # 解码 chunks
        chunks_flat = self.chunks_decoder(last_hidden)  # [batch, num_chunks * d_latent]
        predicted_chunks = chunks_flat.view(batch_size, self.num_chunks, self.d_latent)
        predicted_chunks = self.chunks_norm(predicted_chunks)
        
        # 构建 HierarchicalLatent
        predicted_hierarchical = HierarchicalLatent(
            global_=predicted_global.unsqueeze(1),
            chunks=predicted_chunks,
        )
        
        return HierarchicalDynamicsOutput(
            data=predicted_global,
            predicted_global=predicted_global,
            predicted_chunks=predicted_chunks,
            predicted_hierarchical=predicted_hierarchical,
            hidden_states=output,
            final_state=new_hidden,
        )
    
    def predict_next(
        self,
        history: Union[List[HierarchicalLatent], List[UnifiedLatent]],
    ) -> HierarchicalLatent:
        """便捷方法：预测下一个 HierarchicalLatent"""
        output = self.forward(history)
        return output.predicted_hierarchical
    
    @classmethod
    def from_config(cls, config: ModelConfig, **kwargs) -> "HierarchicalDynamics":
        """从配置创建实例"""
        return cls(
            d_latent=getattr(config, 'd_latent', 512),
            d_model=getattr(config, 'd_model', 768),
            num_chunks=getattr(config, 'max_chunks', 8),
            d_state=getattr(config, 'd_state', 64),
            num_layers=getattr(config, 'dynamics_layers', 4),
            brain_type=getattr(config, 'brain_type', 'gru'),
            dropout=getattr(config, 'dropout', 0.1),
            **kwargs,
        )


class UnifiedDynamics(BaseModule):
    """
    统一动力学模型
    
    支持三通道的动力学预测。
    """
    
    MODULE_TYPE = "dynamics"
    
    def __init__(
        self,
        d_latent: int = 512,
        d_model: int = 768,
        num_chunks: int = 8,
        use_three_channel: bool = True,
        **kwargs,
    ):
        super().__init__()
        
        self.use_three_channel = use_three_channel
        
        # 语义通道动力学
        self.hierarchical_dynamics = HierarchicalDynamics(
            d_latent=d_latent,
            d_model=d_model,
            num_chunks=num_chunks,
            **kwargs,
        )
        
        # 结构通道暂时不需要动力学 (结构通常不变)
        # 符号通道暂时不需要动力学 (符号是精确锚定的)
    
    def forward(
        self,
        sequence: List[UnifiedLatent],
        hidden: Optional[torch.Tensor] = None,
    ) -> UnifiedLatent:
        """
        预测下一个 UnifiedLatent
        
        Args:
            sequence: UnifiedLatent 序列
            hidden: 隐藏状态
            
        Returns:
            predicted: 预测的 UnifiedLatent
        """
        # 预测语义通道
        dynamics_output = self.hierarchical_dynamics(sequence, hidden)
        
        # 构建 UnifiedLatent (不含结构和符号)
        predicted = UnifiedLatent(
            semantic=dynamics_output.predicted_hierarchical,
            structure=None,
            symbols=None,
            scene=sequence[-1].scene if sequence else "chat",
            metadata={"predicted": True},
        )
        
        return predicted
    
    @classmethod
    def from_config(cls, config: ModelConfig, **kwargs) -> "UnifiedDynamics":
        """从配置创建实例"""
        return cls(
            d_latent=getattr(config, 'd_latent', 512),
            d_model=getattr(config, 'd_model', 768),
            num_chunks=getattr(config, 'max_chunks', 8),
            use_three_channel=getattr(config, 'use_three_channel', True),
            **kwargs,
        )
