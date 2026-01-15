"""
Query Retriever - 问题导向检索器

核心功能：根据用户问题动态调整检索策略

设计原理:
1. 将用户问题编码为查询向量
2. 在记忆库中检索相关历史
3. 根据问题类型调整权重
"""

from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn

from ..interfaces.base_module import MemoryItem, LatentVector
from .latent_memory_bank import LatentMemoryBank


class QueryRetriever(nn.Module):
    """
    问题导向检索器
    
    根据用户问题从记忆库中检索相关内容。
    
    使用示例:
        retriever = QueryRetriever(d_latent=512, d_query=768)
        
        # 检索
        query_vec = question_encoder(user_question)
        retrieved = retriever(query_vec, memory_bank, k=5)
    """
    
    def __init__(
        self,
        d_latent: int = 512,
        d_query: int = 768,
        num_heads: int = 8,
    ):
        """
        Args:
            d_latent: 记忆向量维度
            d_query: 查询向量维度
            num_heads: 注意力头数
        """
        super().__init__()
        
        self.d_latent = d_latent
        self.d_query = d_query
        
        # 查询投影
        self.query_proj = nn.Linear(d_query, d_latent)
        
        # 相关性评分
        self.relevance_scorer = nn.Sequential(
            nn.Linear(d_latent * 2, d_latent),
            nn.ReLU(),
            nn.Linear(d_latent, 1),
        )
    
    def forward(
        self,
        query: torch.Tensor,
        memory_bank: LatentMemoryBank,
        k: int = 5,
        current_context: Optional[torch.Tensor] = None,
    ) -> List[MemoryItem]:
        """
        检索相关记忆
        
        Args:
            query: 查询向量 [batch, d_query] 或 [d_query]
            memory_bank: 记忆库
            k: 返回数量
            current_context: 当前上下文 (可用于过滤)
            
        Returns:
            检索到的 MemoryItem 列表
        """
        # 投影到记忆空间
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        query_latent = self.query_proj(query)  # [batch, d_latent]
        
        # 从记忆库检索
        retrieved = memory_bank.retrieve(query_latent.squeeze(0), k=k * 2)
        
        if not retrieved:
            return []
        
        # 重排序
        if len(retrieved) > k:
            retrieved = self._rerank(query_latent.squeeze(0), retrieved, k)
        
        return retrieved[:k]
    
    def _rerank(
        self,
        query: torch.Tensor,
        candidates: List[MemoryItem],
        k: int,
    ) -> List[MemoryItem]:
        """
        使用神经网络重排序
        """
        # 计算每个候选的相关性分数
        scores = []
        for item in candidates:
            combined = torch.cat([query, item.vector], dim=-1)
            score = self.relevance_scorer(combined.unsqueeze(0))
            scores.append(score.item())
        
        # 按分数排序
        sorted_items = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )
        
        return [item for _, item in sorted_items]
    
    def encode_question(
        self,
        question_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        将问题编码为检索查询
        
        Args:
            question_embedding: 问题的文本嵌入
            
        Returns:
            检索查询向量
        """
        return self.query_proj(question_embedding)
