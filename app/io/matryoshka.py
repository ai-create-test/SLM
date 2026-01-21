"""
Matryoshka Projection - 俄罗斯套娃嵌套投影

AMHVQ+ 核心组件：实现单一向量的多精度嵌套表示。

原理:
    z = [z_0, z_1, z_2, ..., z_511]
         ↑         ↑              ↑
        z[:64]   z[:128]       z[:256]       z[:512]
        极简     基础          完整          高保真
        
    训练时对不同前缀长度施加多级损失，
    推理时按需截断使用。
    
参考: Matryoshka Representation Learning (2024)
"""

from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MatryoshkaOutput:
    """
    Matryoshka 投影输出
    """
    full: torch.Tensor                # 完整向量 [batch, d_output]
    nested: dict                      # 嵌套向量 {dim: [batch, dim]}
    
    def get_level(self, level: int) -> torch.Tensor:
        """获取指定级别的嵌套向量"""
        dims = sorted(self.nested.keys())
        if level < 0 or level >= len(dims):
            raise ValueError(f"Level {level} out of range [0, {len(dims)-1}]")
        dim = dims[level]
        return self.nested[dim]


class MatryoshkaProjection(nn.Module):
    """
    Matryoshka 嵌套投影层
    
    使用示例:
        proj = MatryoshkaProjection(
            d_input=768, 
            d_output=512, 
            nesting_dims=[64, 128, 256, 512]
        )
        
        # 投影
        z = proj(hidden)  # [batch, 512]
        
        # 获取不同精度
        z_fast = proj.get_nested(z, level=0)    # [batch, 64]
        z_normal = proj.get_nested(z, level=2)  # [batch, 256]
        
        # 训练时计算多级损失
        loss = proj.multi_level_loss(z, target)
    """
    
    def __init__(
        self,
        d_input: int,
        d_output: int,
        nesting_dims: Optional[List[int]] = None,
        use_layer_norm: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_input: 输入维度
            d_output: 输出维度 (最大嵌套维度)
            nesting_dims: 嵌套维度列表，如 [64, 128, 256, 512]
            use_layer_norm: 是否使用 LayerNorm
            dropout: Dropout 率
        """
        super().__init__()
        
        self.d_input = d_input
        self.d_output = d_output
        
        # 默认嵌套维度
        if nesting_dims is None:
            nesting_dims = [64, 128, 256, d_output]
        
        # 确保最大维度匹配
        if max(nesting_dims) != d_output:
            nesting_dims = [d for d in nesting_dims if d <= d_output]
            if d_output not in nesting_dims:
                nesting_dims.append(d_output)
            nesting_dims = sorted(nesting_dims)
        
        self.nesting_dims = sorted(nesting_dims)
        self.num_levels = len(self.nesting_dims)
        
        # 投影层
        self.proj = nn.Linear(d_input, d_output)
        
        if use_layer_norm:
            self.norm = nn.LayerNorm(d_output)
        else:
            self.norm = None
        
        self.dropout = nn.Dropout(dropout)
        
        # 每个嵌套级别的可学习缩放因子
        self.level_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1))
            for _ in self.nesting_dims
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch, d_input] 或 [batch, seq_len, d_input]
            
        Returns:
            z: [batch, d_output] 或 [batch, seq_len, d_output]
        """
        z = self.proj(x)
        
        if self.norm is not None:
            z = self.norm(z)
        
        z = self.dropout(z)
        
        return z
    
    def forward_with_nested(self, x: torch.Tensor) -> MatryoshkaOutput:
        """
        前向传播，同时返回所有嵌套级别
        
        Returns:
            MatryoshkaOutput
        """
        z = self.forward(x)
        
        nested = {}
        for i, dim in enumerate(self.nesting_dims):
            nested_z = z[..., :dim] * self.level_scales[i]
            nested[dim] = nested_z
        
        return MatryoshkaOutput(full=z, nested=nested)
    
    def get_nested(self, z: torch.Tensor, level: int) -> torch.Tensor:
        """
        获取指定级别的嵌套向量
        
        Args:
            z: 完整向量 [batch, d_output]
            level: 嵌套级别 (0 = 最小, num_levels-1 = 最大)
            
        Returns:
            嵌套向量 [batch, nesting_dims[level]]
        """
        if level < 0 or level >= self.num_levels:
            raise ValueError(f"Level {level} out of range [0, {self.num_levels - 1}]")
        
        dim = self.nesting_dims[level]
        return z[..., :dim] * self.level_scales[level]
    
    def get_nested_by_dim(self, z: torch.Tensor, dim: int) -> torch.Tensor:
        """
        根据维度获取嵌套向量
        
        Args:
            z: 完整向量
            dim: 目标维度
            
        Returns:
            截断后的向量
        """
        if dim not in self.nesting_dims:
            # 找到最接近的维度
            dim = min(self.nesting_dims, key=lambda d: abs(d - dim) if d >= dim else float('inf'))
        
        level = self.nesting_dims.index(dim)
        return self.get_nested(z, level)
    
    def multi_level_loss(
        self,
        z: torch.Tensor,
        target: torch.Tensor,
        loss_type: str = "mse",
        level_weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        计算多级别损失
        
        对每个嵌套级别分别计算损失，鼓励前缀也有意义。
        
        Args:
            z: 预测向量 [batch, d_output]
            target: 目标向量 [batch, d_output]
            loss_type: 损失类型 ("mse" 或 "cosine")
            level_weights: 每个级别的权重
            
        Returns:
            total_loss: 加权总损失
        """
        if level_weights is None:
            # 默认权重：越高级别权重越大
            level_weights = [1.0] * self.num_levels
        
        assert len(level_weights) == self.num_levels
        
        losses = []
        for i, dim in enumerate(self.nesting_dims):
            z_level = z[..., :dim]
            target_level = target[..., :dim]
            
            if loss_type == "mse":
                loss = F.mse_loss(z_level, target_level)
            elif loss_type == "cosine":
                # 1 - cosine_similarity
                cos_sim = F.cosine_similarity(z_level, target_level, dim=-1)
                loss = (1 - cos_sim).mean()
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            
            losses.append(level_weights[i] * loss)
        
        return sum(losses) / len(losses)
    
    def multi_level_contrastive_loss(
        self,
        z: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        多级别对比损失
        
        Args:
            z: anchor 向量 [batch, d_output]
            positive: 正样本 [batch, d_output]
            negatives: 负样本 [batch, num_neg, d_output]
            temperature: 温度参数
            
        Returns:
            对比损失
        """
        losses = []
        
        for dim in self.nesting_dims:
            z_level = F.normalize(z[..., :dim], dim=-1)
            pos_level = F.normalize(positive[..., :dim], dim=-1)
            neg_level = F.normalize(negatives[..., :, :dim], dim=-1)
            
            # 正样本相似度
            pos_sim = (z_level * pos_level).sum(dim=-1) / temperature
            
            # 负样本相似度
            neg_sim = torch.bmm(
                neg_level, z_level.unsqueeze(-1)
            ).squeeze(-1) / temperature  # [batch, num_neg]
            
            # InfoNCE loss
            logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
            labels = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
            loss = F.cross_entropy(logits, labels)
            
            losses.append(loss)
        
        return sum(losses) / len(losses)
    
    @property
    def levels_info(self) -> str:
        """返回嵌套级别信息"""
        return f"Levels: {self.nesting_dims}"


# ============================================================
# 工具函数
# ============================================================

def adaptive_precision(
    z: torch.Tensor,
    complexity_score: torch.Tensor,
    nesting_dims: List[int],
) -> torch.Tensor:
    """
    根据复杂度自适应选择精度
    
    Args:
        z: 完整向量 [batch, d_output]
        complexity_score: 复杂度分数 [batch] (0-1)
        nesting_dims: 嵌套维度列表
        
    Returns:
        自适应精度的向量 (混合不同级别)
    """
    batch_size = z.shape[0]
    sorted_dims = sorted(nesting_dims)
    num_levels = len(sorted_dims)
    
    # 根据复杂度选择级别
    levels = (complexity_score * num_levels).long().clamp(0, num_levels - 1)
    
    # 创建输出 (以最大维度为准)
    max_dim = max(sorted_dims)
    output = torch.zeros(batch_size, max_dim, device=z.device, dtype=z.dtype)
    
    for b in range(batch_size):
        dim = sorted_dims[levels[b]]
        output[b, :dim] = z[b, :dim]
    
    return output
