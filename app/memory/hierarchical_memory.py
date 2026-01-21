"""
Hierarchical Memory - 层次化记忆存储

Phase 15: Memory 模块适配

支持 HierarchicalLatent 的存储和检索:
- Global 级别检索
- Chunk 级别细粒度匹配
"""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces.base_module import HierarchicalLatent
from ..interfaces.unified_latent import UnifiedLatent


@dataclass
class MemoryEntry:
    """记忆条目"""
    id: str
    global_vector: torch.Tensor
    chunks: Optional[torch.Tensor] = None
    num_chunks: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class RetrievalResult:
    """检索结果"""
    entries: List[MemoryEntry]
    similarities: List[float]
    matched_chunks: Optional[List[List[int]]] = None


class HierarchicalMemoryStore:
    """
    层次化记忆存储
    
    支持 HierarchicalLatent 的两级存储和检索:
    - Level 1: Global 向量检索 (快速筛选)
    - Level 2: Chunk 级别精细匹配 (精确定位)
    """
    
    def __init__(
        self,
        d_latent: int = 512,
        max_entries: int = 10000,
        global_threshold: float = 0.7,
        chunk_threshold: float = 0.8,
    ):
        self.d_latent = d_latent
        self.max_entries = max_entries
        self.global_threshold = global_threshold
        self.chunk_threshold = chunk_threshold
        
        self._entries: Dict[str, MemoryEntry] = {}
        self._global_index: List[torch.Tensor] = []
        self._entry_ids: List[str] = []
        
        self._counter = 0
    
    def store(
        self,
        latent: HierarchicalLatent,
        entry_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        存储 HierarchicalLatent
        
        Args:
            latent: HierarchicalLatent
            entry_id: 可选的条目 ID
            metadata: 元数据
            
        Returns:
            条目 ID
        """
        import time
        
        if entry_id is None:
            entry_id = f"mem_{self._counter}"
            self._counter += 1
        
        # 创建条目
        entry = MemoryEntry(
            id=entry_id,
            global_vector=latent.global_.squeeze(1).detach().cpu(),
            chunks=latent.chunks.detach().cpu() if latent.chunks is not None else None,
            num_chunks=latent.num_chunks,
            metadata=metadata or {},
            timestamp=time.time(),
        )
        
        # 存储
        self._entries[entry_id] = entry
        self._global_index.append(entry.global_vector)
        self._entry_ids.append(entry_id)
        
        # 限制大小
        if len(self._entries) > self.max_entries:
            oldest_id = self._entry_ids.pop(0)
            del self._entries[oldest_id]
            self._global_index.pop(0)
        
        return entry_id
    
    def retrieve(
        self,
        query: HierarchicalLatent,
        top_k: int = 5,
        use_chunks: bool = True,
    ) -> RetrievalResult:
        """
        检索相似记忆
        
        Args:
            query: 查询 HierarchicalLatent
            top_k: 返回数量
            use_chunks: 是否使用 chunk 级别匹配
            
        Returns:
            RetrievalResult
        """
        if not self._entries:
            return RetrievalResult(entries=[], similarities=[])
        
        # Level 1: Global 检索
        query_global = query.global_.squeeze(1).detach()  # [batch, d_latent]
        
        global_index = torch.stack(self._global_index)  # [num_entries, ...]
        
        # 确保 global_index 是二维的 [num_entries, d_latent]
        if global_index.dim() > 2:
            global_index = global_index.view(global_index.size(0), -1)
        elif global_index.dim() == 1:
            global_index = global_index.unsqueeze(0)
        
        # 确保 query_global 是二维的 [1, d_latent]
        if query_global.dim() == 1:
            query_global = query_global.unsqueeze(0)
        
        # 计算相似度
        query_norm = F.normalize(query_global, dim=-1)  # [1, d_latent]
        index_norm = F.normalize(global_index, dim=-1)  # [num_entries, d_latent]
        similarities = torch.mm(query_norm, index_norm.transpose(-2, -1)).squeeze(0)  # [num_entries]
        
        # 获取 top_k
        top_k = min(top_k, len(self._entries))
        top_scores, top_indices = torch.topk(similarities, top_k)
        
        # 收集结果
        entries = []
        scores = []
        chunk_matches = []
        
        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            entry_id = self._entry_ids[idx]
            entry = self._entries[entry_id]
            entries.append(entry)
            scores.append(score)
            
            # Level 2: Chunk 匹配 (可选)
            if use_chunks and entry.chunks is not None and query.chunks is not None:
                query_chunks = query.chunks.detach()
                matched = self._match_chunks(query_chunks, entry.chunks)
                chunk_matches.append(matched)
            else:
                chunk_matches.append([])
        
        return RetrievalResult(
            entries=entries,
            similarities=scores,
            matched_chunks=chunk_matches if use_chunks else None,
        )
    
    def _match_chunks(
        self,
        query_chunks: torch.Tensor,
        entry_chunks: torch.Tensor,
    ) -> List[int]:
        """匹配 chunks"""
        # query: [batch, num_q, d], entry: [num_e, d]
        query_flat = query_chunks.squeeze(0)  # [num_q, d]
        
        q_norm = F.normalize(query_flat, dim=-1)
        e_norm = F.normalize(entry_chunks.squeeze(0), dim=-1)
        
        sims = torch.mm(q_norm, e_norm.T)  # [num_q, num_e]
        best_matches = sims.argmax(dim=1).tolist()
        
        return best_matches
    
    def store_unified(
        self,
        unified: UnifiedLatent,
        entry_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """存储 UnifiedLatent"""
        return self.store(unified.semantic, entry_id, metadata)
    
    def clear(self) -> None:
        """清空记忆"""
        self._entries.clear()
        self._global_index.clear()
        self._entry_ids.clear()
    
    @property
    def num_entries(self) -> int:
        return len(self._entries)
