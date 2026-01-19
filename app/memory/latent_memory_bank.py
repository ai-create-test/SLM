"""
Latent Memory Bank - 潜向量记忆库

核心功能：存储和检索历史生成的段落潜向量

设计原理:
1. 向量索引 - 使用 FAISS 实现快速近似最近邻搜索
2. 元数据管理 - 存储时间戳、重要度、原文等
3. 容量管理 - LRU/重要度淘汰策略
4. 持久化 - 支持保存和加载
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
import json
from pathlib import Path
import numpy as np
import torch

from ..interfaces.base_module import MemoryItem, LatentVector


@dataclass
class MemoryConfig:
    """记忆库配置"""
    d_latent: int = 512                 # 潜向量维度
    max_size: int = 10000               # 最大条目数
    index_type: str = "flat"            # 索引类型: 'flat', 'ivf', 'hnsw'
    nlist: int = 100                    # IVF 聚类数
    nprobe: int = 10                    # IVF 搜索聚类数
    ef_search: int = 32                 # HNSW 搜索参数
    use_gpu: bool = False               # 是否使用 GPU 加速
    decay_factor: float = 0.99          # 时间衰减因子
    importance_threshold: float = 0.1   # 重要度淘汰阈值


class LatentMemoryBank:
    """
    潜向量记忆库
    
    存储历史生成的段落潜向量，支持高效检索。
    
    使用示例:
        memory = LatentMemoryBank(d_latent=512, max_size=10000)
        
        # 添加记忆
        memory.add(z_vector, content="段落内容", importance=0.8)
        
        # 检索
        retrieved = memory.retrieve(query_vector, k=5)
        for item in retrieved:
            print(item.content, item.importance)
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None, **kwargs):
        """
        Args:
            config: 配置对象
            **kwargs: 覆盖配置的参数
        """
        if config is not None:
            self.config = config
        else:
            self.config = MemoryConfig(**kwargs)
        
        self.d_latent = self.config.d_latent
        self.max_size = self.config.max_size
        
        # 存储
        self._vectors: List[np.ndarray] = []
        self._metadata: List[Dict[str, Any]] = []
        self._timestamps: List[float] = []
        self._importance: List[float] = []
        
        # FAISS 索引 (延迟初始化)
        self._index = None
        self._index_dirty = True
        
        # 统计
        self._total_added = 0
        self._total_retrieved = 0
    
    def add(
        self,
        vector: torch.Tensor,
        content: Optional[str] = None,
        importance: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> int:
        """
        添加新记忆
        
        Args:
            vector: 潜向量 [d_latent]
            content: 原始文本内容
            importance: 重要度分数 (0-1)
            metadata: 额外元数据
            node_id: 可选的 GraphMemory 节点 ID，用于 GraphRAG 检索
            
        Returns:
            新记忆的索引
        """
        # 转换为 numpy
        if isinstance(vector, torch.Tensor):
            vector = vector.detach().cpu().numpy()
        
        vector = vector.astype(np.float32).reshape(-1)
        
        # 容量检查
        if len(self._vectors) >= self.max_size:
            self._evict_oldest()
        
        # 添加
        idx = len(self._vectors)
        self._vectors.append(vector)
        self._timestamps.append(time.time())
        self._importance.append(importance)
        self._metadata.append({
            "content": content,
            "index": idx,
            "node_id": node_id,  # GraphRAG 支持
            **(metadata or {}),
        })
        
        self._index_dirty = True
        self._total_added += 1
        
        return idx
    
    def add_batch(
        self,
        vectors: torch.Tensor,
        contents: Optional[List[str]] = None,
        importances: Optional[List[float]] = None,
    ) -> List[int]:
        """批量添加"""
        batch_size = vectors.shape[0]
        contents = contents or [None] * batch_size
        importances = importances or [1.0] * batch_size
        
        indices = []
        for i in range(batch_size):
            idx = self.add(
                vectors[i],
                content=contents[i],
                importance=importances[i],
            )
            indices.append(idx)
        
        return indices
    
    def retrieve(
        self,
        query: torch.Tensor,
        k: int = 5,
        min_importance: float = 0.0,
    ) -> List[MemoryItem]:
        """
        检索最相关的记忆
        
        Args:
            query: 查询向量 [d_latent]
            k: 返回的记忆数量
            min_importance: 最小重要度阈值
            
        Returns:
            MemoryItem 列表
        """
        if len(self._vectors) == 0:
            return []
        
        # 确保索引是最新的
        self._ensure_index()
        
        # 转换查询
        if isinstance(query, torch.Tensor):
            query = query.detach().cpu().numpy()
        query = query.astype(np.float32).reshape(1, -1)
        
        # 搜索
        k = min(k, len(self._vectors))
        distances, indices = self._index.search(query, k)
        
        # 过滤并构建结果
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            
            importance = self._importance[idx]
            if importance < min_importance:
                continue
            
            results.append(MemoryItem(
                vector=torch.from_numpy(self._vectors[idx]),
                content=self._metadata[idx].get("content"),
                timestamp=self._timestamps[idx],
                importance=importance,
                metadata=self._metadata[idx],
            ))
        
        self._total_retrieved += len(results)
        return results
    
    def retrieve_tensor(
        self,
        query: torch.Tensor,
        k: int = 5,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        检索并返回张量格式
        
        Returns:
            vectors: [k, d_latent]
            indices: 原始索引列表
        """
        items = self.retrieve(query, k)
        
        if not items:
            return torch.zeros(0, self.d_latent), []
        
        vectors = torch.stack([item.vector for item in items])
        indices = [item.metadata.get("index", -1) for item in items]
        
        return vectors, indices
    
    def _ensure_index(self) -> None:
        """确保 FAISS 索引是最新的"""
        if not self._index_dirty and self._index is not None:
            return
        
        try:
            import faiss
            
            vectors = np.array(self._vectors).astype(np.float32)
            
            if self.config.index_type == "flat":
                self._index = faiss.IndexFlatL2(self.d_latent)
            elif self.config.index_type == "ivf":
                quantizer = faiss.IndexFlatL2(self.d_latent)
                nlist = min(self.config.nlist, len(vectors))
                self._index = faiss.IndexIVFFlat(quantizer, self.d_latent, nlist)
                self._index.nprobe = self.config.nprobe
                self._index.train(vectors)
            elif self.config.index_type == "hnsw":
                self._index = faiss.IndexHNSWFlat(self.d_latent, 32)
                self._index.hnsw.efSearch = self.config.ef_search
            else:
                self._index = faiss.IndexFlatL2(self.d_latent)
            
            self._index.add(vectors)
            
        except ImportError:
            # FAISS 不可用，使用简单的余弦相似度
            self._index = SimpleIndex(self._vectors)
        
        self._index_dirty = False
    
    def _evict_oldest(self) -> None:
        """淘汰最老/最不重要的记忆"""
        # 综合时间和重要度
        scores = []
        current_time = time.time()
        
        for i in range(len(self._vectors)):
            age = current_time - self._timestamps[i]
            score = self._importance[i] * (self.config.decay_factor ** (age / 3600))
            scores.append(score)
        
        # 找到最低分的
        min_idx = np.argmin(scores)
        
        # 删除
        del self._vectors[min_idx]
        del self._timestamps[min_idx]
        del self._importance[min_idx]
        del self._metadata[min_idx]
        
        self._index_dirty = True
    
    def update_importance(self, idx: int, importance: float) -> None:
        """更新记忆的重要度"""
        if 0 <= idx < len(self._importance):
            self._importance[idx] = importance
    
    def clear(self) -> None:
        """清空记忆库"""
        self._vectors.clear()
        self._metadata.clear()
        self._timestamps.clear()
        self._importance.clear()
        self._index = None
        self._index_dirty = True
    
    def save(self, path: str) -> None:
        """保存到文件"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # 保存向量
        np.save(path / "vectors.npy", np.array(self._vectors))
        
        # 保存元数据
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({
                "metadata": self._metadata,
                "timestamps": self._timestamps,
                "importance": self._importance,
                "config": {
                    "d_latent": self.d_latent,
                    "max_size": self.max_size,
                },
            }, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "LatentMemoryBank":
        """从文件加载"""
        path = Path(path)
        
        # 加载元数据
        with open(path / "metadata.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 创建实例
        config = MemoryConfig(**data.get("config", {}))
        memory = cls(config)
        
        # 加载向量
        vectors = np.load(path / "vectors.npy")
        memory._vectors = [v for v in vectors]
        memory._metadata = data["metadata"]
        memory._timestamps = data["timestamps"]
        memory._importance = data["importance"]
        memory._index_dirty = True
        
        return memory
    
    def __len__(self) -> int:
        return len(self._vectors)
    
    def __repr__(self) -> str:
        return f"LatentMemoryBank(size={len(self)}, d_latent={self.d_latent})"


class SimpleIndex:
    """简单的向量索引 (FAISS 不可用时的后备)"""
    
    def __init__(self, vectors: List[np.ndarray]):
        self.vectors = np.array(vectors) if vectors else np.array([]).reshape(0, 0)
    
    def add(self, vectors: np.ndarray) -> None:
        self.vectors = vectors
    
    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.vectors) == 0:
            return np.array([[-1] * k]), np.array([[-1] * k])
        
        # 计算余弦相似度
        query_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
        vectors_norm = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-8)
        
        similarities = np.dot(query_norm, vectors_norm.T)[0]
        
        # 取 top-k
        k = min(k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:k]
        top_distances = 1 - similarities[top_indices]  # 转换为距离
        
        return np.array([top_distances]), np.array([top_indices])
