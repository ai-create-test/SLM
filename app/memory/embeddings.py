"""
Embeddings Module - Token嵌入与位置编码

核心功能：
1. Token ID → 稠密向量 (TokenEmbedding)
2. 位置信息编码 (RoPE - 旋转位置编码)
3. 组合输出模型可计算的张量

设计原则：
- 基于PyTorch实现
- 兼容Transformer架构
- 支持多种位置编码方式
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EmbeddingConfig:
    """嵌入层配置"""
    vocab_size: int = 100000       # 词表大小
    d_model: int = 768             # 嵌入维度
    max_seq_len: int = 2048        # 最大序列长度
    dropout: float = 0.1          # Dropout率
    padding_idx: int = 0           # 填充token ID
    
    # RoPE参数
    rope_theta: float = 10000.0    # RoPE基础频率
    rope_scaling: Optional[float] = None  # 长度外推缩放


class TokenEmbedding(nn.Module):
    """
    Token嵌入层
    
    将离散的Token ID转换为连续的稠密向量。
    
    使用示例：
        embedding = TokenEmbedding(vocab_size=50000, d_model=768)
        token_ids = torch.tensor([[1, 2, 3, 4]])
        embeddings = embedding(token_ids)  # [1, 4, 768]
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int = 0,
        dropout: float = 0.1,
        init_std: float = 0.02,
    ):
        """
        Args:
            vocab_size: 词表大小
            d_model: 嵌入维度
            padding_idx: 填充token的ID
            dropout: Dropout率
            init_std: 权重初始化标准差
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx
        
        # 嵌入层
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
        # 缩放因子（参考Transformer原论文）
        self.scale = math.sqrt(d_model)
        
        # 初始化权重
        self._init_weights(init_std)
    
    def _init_weights(self, std: float) -> None:
        """初始化嵌入权重（正态分布）"""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=std)
        if self.padding_idx is not None:
            nn.init.zeros_(self.embedding.weight[self.padding_idx])
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            token_ids: Token ID张量 [batch, seq_len]
            
        Returns:
            嵌入向量 [batch, seq_len, d_model]
        """
        # 嵌入并缩放
        embeddings = self.embedding(token_ids) * self.scale
        
        # 应用dropout
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def get_embedding_weight(self) -> torch.Tensor:
        """获取嵌入权重矩阵"""
        return self.embedding.weight


class PositionalEncoding(nn.Module):
    """
    经典正弦位置编码
    
    来自"Attention Is All You Need"论文的原始位置编码。
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # 预计算位置编码
        pe = self._create_pe(max_seq_len, d_model)
        self.register_buffer('pe', pe)
    
    def _create_pe(self, max_len: int, d_model: int) -> torch.Tensor:
        """创建位置编码矩阵"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, max_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        
        Args:
            x: 输入张量 [batch, seq_len, d_model]
            
        Returns:
            添加位置编码后的张量
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """
    旋转位置编码 (RoPE)
    
    来自"RoFormer: Enhanced Transformer with Rotary Position Embedding"论文。
    被LLaMA、GPT-NeoX等现代LLM广泛采用。
    
    优势：
    - 相对位置信息隐式编码
    - 更好的长度外推能力
    - 计算效率高
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        scaling_factor: Optional[float] = None,
    ):
        """
        Args:
            d_model: 嵌入维度（必须是偶数）
            max_seq_len: 最大序列长度
            theta: 基础频率
            scaling_factor: 长度外推缩放因子
        """
        super().__init__()
        
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scaling_factor = scaling_factor
        
        # 预计算频率
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int) -> None:
        """构建RoPE缓存"""
        # 计算频率
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.d_model, 2).float() / self.d_model)
        )
        self.register_buffer('inv_freq', inv_freq)
        
        # 位置索引
        t = torch.arange(seq_len, dtype=torch.float)
        
        # 应用缩放
        if self.scaling_factor is not None:
            t = t / self.scaling_factor
        
        # 计算外积得到相位角
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        
        # 构建cos和sin缓存
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos()
        sin_cached = emb.sin()
        
        self.register_buffer('cos_cached', cos_cached)
        self.register_buffer('sin_cached', sin_cached)
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """将张量的后半部分旋转到前半部分"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        应用旋转位置编码
        
        Args:
            x: 输入张量 [batch, seq_len, d_model]
            position_ids: 可选的位置ID [batch, seq_len]
            
        Returns:
            应用RoPE后的张量
        """
        seq_len = x.size(1)
        
        # 获取cos和sin
        if position_ids is not None:
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
        else:
            cos = self.cos_cached[:seq_len].unsqueeze(0)
            sin = self.sin_cached[:seq_len].unsqueeze(0)
        
        # 应用旋转
        x_rotated = (x * cos) + (self._rotate_half(x) * sin)
        
        return x_rotated
    
    def extend_cache(self, new_seq_len: int) -> None:
        """扩展缓存以支持更长序列"""
        if new_seq_len <= self.max_seq_len:
            return
        
        self.max_seq_len = new_seq_len
        self._build_cache(new_seq_len)


class CombinedEmbedding(nn.Module):
    """
    组合嵌入层
    
    整合Token嵌入和位置编码，提供统一接口。
    
    使用示例：
        embedding = CombinedEmbedding(
            vocab_size=50000,
            d_model=768,
            position_encoding='rope'
        )
        token_ids = torch.tensor([[1, 2, 3, 4]])
        output = embedding(token_ids)  # [1, 4, 768]
    """
    
    POSITION_ENCODINGS = {
        'sinusoidal': PositionalEncoding,
        'rope': RotaryPositionalEmbedding,
    }
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 2048,
        position_encoding: str = 'rope',
        dropout: float = 0.1,
        padding_idx: int = 0,
        **kwargs
    ):
        """
        Args:
            vocab_size: 词表大小
            d_model: 嵌入维度
            max_seq_len: 最大序列长度
            position_encoding: 位置编码类型 ('sinusoidal' 或 'rope')
            dropout: Dropout率
            padding_idx: 填充token ID
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.position_encoding_type = position_encoding
        
        # Token嵌入
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            padding_idx=padding_idx,
            dropout=dropout,
        )
        
        # 位置编码
        if position_encoding not in self.POSITION_ENCODINGS:
            raise ValueError(
                f"Unknown position encoding: {position_encoding}. "
                f"Available: {list(self.POSITION_ENCODINGS.keys())}"
            )
        
        if position_encoding == 'rope':
            self.position_encoding = RotaryPositionalEmbedding(
                d_model=d_model,
                max_seq_len=max_seq_len,
                **{k: v for k, v in kwargs.items() if k in ['theta', 'scaling_factor']}
            )
        else:
            self.position_encoding = PositionalEncoding(
                d_model=d_model,
                max_seq_len=max_seq_len,
                dropout=dropout,
            )
    
    def forward(
        self,
        token_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            token_ids: Token ID张量 [batch, seq_len]
            position_ids: 可选的位置ID
            
        Returns:
            嵌入向量 [batch, seq_len, d_model]
        """
        # Token嵌入
        embeddings = self.token_embedding(token_ids)
        
        # 位置编码
        if self.position_encoding_type == 'rope':
            embeddings = self.position_encoding(embeddings, position_ids)
        else:
            embeddings = self.position_encoding(embeddings)
        
        return embeddings
    
    @property
    def embedding_weight(self) -> torch.Tensor:
        """获取token嵌入权重"""
        return self.token_embedding.get_embedding_weight()
