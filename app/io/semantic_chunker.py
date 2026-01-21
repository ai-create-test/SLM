"""
Semantic Chunker - 语义分块器

AMHVQ+ 核心组件：将变长文本切分为语义连贯的块。

原理:
    1. 学习预测每个 token 是否为语义边界
    2. 根据边界将文本切分为 N 个语义块 (N 自适应)
    3. 每个块独立编码，保留结构信息
    
优势:
    - 短句只产生 1-2 个块
    - 复杂段落产生 4-8 个块
    - 自适应压缩率
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ChunkerOutput:
    """
    分块器输出
    """
    chunks: torch.Tensor              # 分块后的表示 [batch, num_chunks, chunk_len, d_model]
    chunk_mask: torch.Tensor          # 有效 chunk 掩码 [batch, num_chunks]
    token_mask: torch.Tensor          # 每个 chunk 内的有效 token 掩码 [batch, num_chunks, chunk_len]
    boundaries: torch.Tensor          # 边界概率 [batch, seq_len]
    num_chunks: torch.Tensor          # 每个样本的 chunk 数 [batch]


class BoundaryPredictor(nn.Module):
    """
    边界预测器
    
    预测每个 token 是否为语义边界。
    """
    
    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: 输入维度
            hidden_dim: 隐藏层维度
            dropout: Dropout 率
        """
        super().__init__()
        
        hidden_dim = hidden_dim or d_model // 4
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        预测边界概率
        
        Args:
            hidden_states: [batch, seq_len, d_model]
            
        Returns:
            boundary_logits: [batch, seq_len]
        """
        logits = self.predictor(hidden_states).squeeze(-1)  # [batch, seq_len]
        return logits


class SemanticChunker(nn.Module):
    """
    语义分块器
    
    使用示例:
        chunker = SemanticChunker(d_model=768, max_chunks=8)
        
        # 分块
        output = chunker(hidden_states, attention_mask)
        chunks = output.chunks  # [batch, num_chunks, chunk_len, d_model]
        
        # 检查实际 chunk 数
        print(output.num_chunks)  # [4, 6, 3, ...]
    """
    
    def __init__(
        self,
        d_model: int,
        max_chunks: int = 8,
        min_chunk_len: int = 4,
        max_chunk_len: int = 64,
        use_learned_boundaries: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: 输入维度
            max_chunks: 最大块数
            min_chunk_len: 最小块长度
            max_chunk_len: 最大块长度
            use_learned_boundaries: 是否使用学习的边界预测
            dropout: Dropout 率
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_chunks = max_chunks
        self.min_chunk_len = min_chunk_len
        self.max_chunk_len = max_chunk_len
        self.use_learned_boundaries = use_learned_boundaries
        
        if use_learned_boundaries:
            self.boundary_predictor = BoundaryPredictor(d_model, dropout=dropout)
        else:
            self.boundary_predictor = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        hard: bool = False,
    ) -> ChunkerOutput:
        """
        前向传播
        
        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: [batch, seq_len] (1 = valid, 0 = padding)
            temperature: Gumbel-Softmax 温度 (训练时)
            hard: 是否使用硬边界 (推理时)
            
        Returns:
            ChunkerOutput
        """
        batch_size, seq_len, d_model = hidden_states.shape
        device = hidden_states.device
        
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        # 预测边界
        if self.use_learned_boundaries and self.boundary_predictor is not None:
            boundary_logits = self.boundary_predictor(hidden_states)  # [batch, seq_len]
            
            if hard or not self.training:
                # 推理时使用硬边界
                boundary_probs = torch.sigmoid(boundary_logits)
                boundaries = (boundary_probs > 0.5).float()
            else:
                # 训练时使用软边界 (Gumbel-Softmax)
                boundaries = torch.sigmoid(boundary_logits / temperature)
        else:
            # 使用固定间隔分块
            boundary_logits = torch.zeros(batch_size, seq_len, device=device)
            boundaries = self._fixed_boundaries(seq_len, attention_mask)
        
        # 根据边界切分
        chunks, chunk_mask, token_mask, num_chunks = self._split_by_boundaries(
            hidden_states, boundaries, attention_mask
        )
        
        return ChunkerOutput(
            chunks=chunks,
            chunk_mask=chunk_mask,
            token_mask=token_mask,
            boundaries=boundaries,
            num_chunks=num_chunks,
        )
    
    def _fixed_boundaries(
        self,
        seq_len: int,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """固定间隔分块"""
        batch_size = attention_mask.shape[0]
        device = attention_mask.device
        
        # 每隔 max_chunk_len 设置一个边界
        boundaries = torch.zeros(batch_size, seq_len, device=device)
        chunk_size = self.max_chunk_len
        
        for i in range(chunk_size, seq_len, chunk_size):
            boundaries[:, i] = 1.0
        
        # 在有效序列的末尾也设置边界
        seq_lens = attention_mask.sum(dim=1).long()
        for b in range(batch_size):
            if seq_lens[b] > 0:
                boundaries[b, seq_lens[b] - 1] = 1.0
        
        return boundaries
    
    def _split_by_boundaries(
        self,
        hidden_states: torch.Tensor,
        boundaries: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        根据边界切分序列
        
        Returns:
            chunks: [batch, max_chunks, max_chunk_len, d_model]
            chunk_mask: [batch, max_chunks]
            token_mask: [batch, max_chunks, max_chunk_len]
            num_chunks: [batch]
        """
        batch_size, seq_len, d_model = hidden_states.shape
        device = hidden_states.device
        
        # 初始化输出
        chunks = torch.zeros(
            batch_size, self.max_chunks, self.max_chunk_len, d_model,
            device=device, dtype=hidden_states.dtype
        )
        chunk_mask = torch.zeros(batch_size, self.max_chunks, device=device)
        token_mask = torch.zeros(
            batch_size, self.max_chunks, self.max_chunk_len,
            device=device
        )
        num_chunks_list = []
        
        for b in range(batch_size):
            # 获取边界位置
            boundary_positions = torch.where(boundaries[b] > 0.5)[0].tolist()
            valid_len = int(attention_mask[b].sum().item())
            
            # 确保末尾有边界
            if len(boundary_positions) == 0 or boundary_positions[-1] != valid_len - 1:
                if valid_len > 0:
                    boundary_positions.append(valid_len - 1)
            
            # 切分
            start = 0
            chunk_idx = 0
            
            for end_pos in boundary_positions:
                if chunk_idx >= self.max_chunks:
                    break
                
                end = end_pos + 1
                chunk_len = min(end - start, self.max_chunk_len)
                
                if chunk_len >= self.min_chunk_len or chunk_idx == 0:
                    # 复制数据
                    actual_len = min(chunk_len, self.max_chunk_len)
                    chunks[b, chunk_idx, :actual_len] = hidden_states[b, start:start + actual_len]
                    token_mask[b, chunk_idx, :actual_len] = 1.0
                    chunk_mask[b, chunk_idx] = 1.0
                    chunk_idx += 1
                
                start = end
            
            num_chunks_list.append(max(1, chunk_idx))
        
        num_chunks = torch.tensor(num_chunks_list, device=device)
        
        return chunks, chunk_mask, token_mask, num_chunks
    
    def get_boundary_loss(
        self,
        boundaries: torch.Tensor,
        attention_mask: torch.Tensor,
        target_num_chunks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算边界正则化损失
        
        鼓励：
        1. 边界稀疏 (不要太多块)
        2. 边界均匀分布
        
        Args:
            boundaries: [batch, seq_len]
            attention_mask: [batch, seq_len]
            target_num_chunks: 目标块数 [batch] (可选)
        """
        # 稀疏性损失：鼓励边界数量适中
        boundary_count = (boundaries * attention_mask).sum(dim=1)
        target_count = target_num_chunks if target_num_chunks is not None else torch.full_like(
            boundary_count, self.max_chunks / 2
        )
        sparsity_loss = F.mse_loss(boundary_count, target_count)
        
        return sparsity_loss


# ============================================================
# 工具函数
# ============================================================

def pool_chunks(
    chunks: torch.Tensor,
    token_mask: torch.Tensor,
    pooling_type: str = "mean",
) -> torch.Tensor:
    """
    对 chunk 内的 token 进行池化
    
    Args:
        chunks: [batch, num_chunks, chunk_len, d_model]
        token_mask: [batch, num_chunks, chunk_len]
        pooling_type: "mean" 或 "first" 或 "last"
        
    Returns:
        pooled: [batch, num_chunks, d_model]
    """
    if pooling_type == "mean":
        mask = token_mask.unsqueeze(-1)  # [batch, num_chunks, chunk_len, 1]
        masked_chunks = chunks * mask
        pooled = masked_chunks.sum(dim=2) / (mask.sum(dim=2) + 1e-8)
    elif pooling_type == "first":
        pooled = chunks[:, :, 0, :]
    elif pooling_type == "last":
        # 获取最后一个有效 token
        batch_size, num_chunks, chunk_len, d_model = chunks.shape
        last_indices = token_mask.sum(dim=-1).long() - 1
        last_indices = last_indices.clamp(min=0)
        
        pooled = torch.zeros(batch_size, num_chunks, d_model, device=chunks.device)
        for b in range(batch_size):
            for c in range(num_chunks):
                idx = last_indices[b, c]
                pooled[b, c] = chunks[b, c, idx]
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")
    
    return pooled
