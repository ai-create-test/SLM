"""
FiLM - Feature-wise Linear Modulation

轻量级的条件调制层。

公式: FiLM(x, c) = γ(c) * x + β(c)

与 AdaLN 的区别：
- AdaLN: 在归一化后应用 scale/shift
- FiLM: 直接在特征上应用 scale/shift

适用场景：
- 计算资源受限
- 需要更快的推理速度
- 条件影响不需要太深
"""

from typing import Optional
import torch
import torch.nn as nn


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation
    
    公式: output = γ(condition) * x + β(condition)
    
    使用示例:
        film = FiLM(d_feature=768, d_condition=128)
        output = film(features, condition_vector)
    """
    
    def __init__(
        self,
        d_feature: int,
        d_condition: int,
    ):
        """
        Args:
            d_feature: 输入特征维度
            d_condition: 条件向量维度
        """
        super().__init__()
        
        self.d_feature = d_feature
        self.d_condition = d_condition
        
        # 条件投影
        self.scale_net = nn.Linear(d_condition, d_feature)
        self.shift_net = nn.Linear(d_condition, d_feature)
        
        # Zero-Initialization
        nn.init.zeros_(self.scale_net.weight)
        nn.init.ones_(self.scale_net.bias)   # γ = 1
        nn.init.zeros_(self.shift_net.weight)
        nn.init.zeros_(self.shift_net.bias)  # β = 0
    
    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        特征调制
        
        Args:
            x: 输入特征 [batch, ..., d_feature]
            condition: 条件向量 [batch, d_condition]
            
        Returns:
            调制后的特征
        """
        scale = self.scale_net(condition)  # [batch, d_feature]
        shift = self.shift_net(condition)  # [batch, d_feature]
        
        # 扩展维度以匹配输入
        while scale.dim() < x.dim():
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        
        return scale * x + shift


class FiLMBlock(nn.Module):
    """
    FiLM Block - 带残差连接的 FiLM
    
    适合嵌入到现有网络架构中。
    """
    
    def __init__(
        self,
        d_feature: int,
        d_condition: int,
        use_residual: bool = True,
        use_norm: bool = True,
    ):
        """
        Args:
            d_feature: 特征维度
            d_condition: 条件维度
            use_residual: 是否使用残差连接
            use_norm: 是否在输出后归一化
        """
        super().__init__()
        
        self.film = FiLM(d_feature, d_condition)
        self.use_residual = use_residual
        
        if use_norm:
            self.norm = nn.LayerNorm(d_feature)
        else:
            self.norm = None
    
    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: 输入 [batch, seq_len, d_feature]
            condition: 条件 [batch, d_condition]
        """
        h = self.film(x, condition)
        
        if self.use_residual:
            h = x + h
        
        if self.norm is not None:
            h = self.norm(h)
        
        return h


class HierarchicalFiLM(nn.Module):
    """
    层次化 FiLM
    
    在多个层级应用 FiLM 调制，实现由粗到细的条件控制。
    """
    
    def __init__(
        self,
        d_features: list,
        d_condition: int,
    ):
        """
        Args:
            d_features: 各层级的特征维度列表
            d_condition: 条件向量维度
        """
        super().__init__()
        
        self.films = nn.ModuleList([
            FiLM(d_feat, d_condition)
            for d_feat in d_features
        ])
    
    def forward(
        self,
        features: list,
        condition: torch.Tensor,
    ) -> list:
        """
        Args:
            features: 各层级特征的列表
            condition: 条件向量
            
        Returns:
            调制后的特征列表
        """
        return [
            film(feat, condition)
            for film, feat in zip(self.films, features)
        ]
