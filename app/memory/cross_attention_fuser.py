"""
Cross-Attention Fuser - 记忆融合层

核心功能：使用 Cross-Attention 融合当前状态与检索到的记忆

公式: Output = CrossAttention(Q=current, KV=memories)
"""

from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces.base_module import MemoryItem


class CrossAttentionFuser(nn.Module):
    """
    Cross-Attention 记忆融合层
    
    将检索到的历史记忆与当前状态融合。
    
    使用示例:
        fuser = CrossAttentionFuser(d_latent=512)
        
        current = encoder(current_paragraph)  # [batch, d_latent]
        memories = retriever.retrieve(...)    # List[MemoryItem]
        
        fused = fuser(current, memories)      # [batch, d_latent]
    """
    
    def __init__(
        self,
        d_latent: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_memory_gate: bool = True,
    ):
        """
        Args:
            d_latent: 潜向量维度
            num_heads: 注意力头数
            dropout: Dropout 率
            use_memory_gate: 是否使用记忆门控
        """
        super().__init__()
        
        self.d_latent = d_latent
        self.num_heads = num_heads
        self.use_memory_gate = use_memory_gate
        
        # Cross-Attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_latent,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # 层归一化
        self.norm = nn.LayerNorm(d_latent)
        
        # 记忆门控 (控制记忆融合的强度)
        if use_memory_gate:
            self.memory_gate = nn.Sequential(
                nn.Linear(d_latent * 2, d_latent),
                nn.Sigmoid(),
            )
        else:
            self.memory_gate = None
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(d_latent, d_latent * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_latent * 2, d_latent),
        )
    
    def forward(
        self,
        current: torch.Tensor,
        memory_vectors: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        融合当前状态与记忆
        
        Args:
            current: 当前状态 [batch, d_latent]
            memory_vectors: 记忆向量 [batch, num_memories, d_latent]
            memory_mask: 记忆掩码 [batch, num_memories] (True=padding)
            
        Returns:
            融合后的状态 [batch, d_latent]
        """
        batch_size = current.shape[0]
        
        # 扩展 current 为序列格式
        query = current.unsqueeze(1)  # [batch, 1, d_latent]
        
        # Cross-Attention: current attends to memories
        attn_output, attn_weights = self.cross_attention(
            query=query,
            key=memory_vectors,
            value=memory_vectors,
            key_padding_mask=memory_mask,
        )
        
        attn_output = attn_output.squeeze(1)  # [batch, d_latent]
        
        # 记忆门控
        if self.use_memory_gate and self.memory_gate is not None:
            gate = self.memory_gate(torch.cat([current, attn_output], dim=-1))
            fused = current + gate * attn_output
        else:
            fused = current + attn_output
        
        # 归一化和投影
        fused = self.norm(fused)
        output = fused + self.output_proj(fused)
        
        return output
    
    def forward_from_items(
        self,
        current: torch.Tensor,
        memory_items: List[MemoryItem],
    ) -> torch.Tensor:
        """
        从 MemoryItem 列表融合
        
        Args:
            current: 当前状态 [batch, d_latent]
            memory_items: MemoryItem 列表
        """
        if not memory_items:
            return current
        
        # 堆叠记忆向量
        memory_vectors = torch.stack([
            item.vector for item in memory_items
        ]).unsqueeze(0)  # [1, num_memories, d_latent]
        
        # 如果 batch_size > 1，复制记忆
        if current.shape[0] > 1:
            memory_vectors = memory_vectors.expand(current.shape[0], -1, -1)
        
        return self.forward(current, memory_vectors)


class MultiHeadMemoryFusion(nn.Module):
    """
    多头记忆融合
    
    使用多个注意力头分别关注不同类型的记忆。
    """
    
    def __init__(
        self,
        d_latent: int = 512,
        num_fusion_heads: int = 4,
        num_attn_heads: int = 8,
    ):
        super().__init__()
        
        self.heads = nn.ModuleList([
            CrossAttentionFuser(d_latent, num_attn_heads)
            for _ in range(num_fusion_heads)
        ])
        
        self.combine = nn.Linear(d_latent * num_fusion_heads, d_latent)
        self.norm = nn.LayerNorm(d_latent)
    
    def forward(
        self,
        current: torch.Tensor,
        memory_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """
        多头融合
        """
        outputs = [head(current, memory_vectors) for head in self.heads]
        combined = torch.cat(outputs, dim=-1)
        output = self.combine(combined)
        return self.norm(current + output)
