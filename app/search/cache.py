"""
Search Cache - 搜索缓存

缓存搜索结果以减少 API 调用。
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time
import hashlib
import json

from .search_interface import SearchResult


@dataclass
class CacheEntry:
    """缓存条目"""
    results: List[SearchResult]
    timestamp: float
    ttl: int
    
    @property
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl


class SearchCache:
    """
    搜索缓存
    
    使用示例:
        cache = SearchCache(default_ttl=3600)
        
        # 检查缓存
        results = cache.get("python programming")
        
        if results is None:
            results = search(...)
            cache.set("python programming", results)
    """
    
    def __init__(
        self,
        default_ttl: int = 3600,
        max_size: int = 1000,
    ):
        """
        Args:
            default_ttl: 默认过期时间 (秒)
            max_size: 最大缓存条目数
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
    
    def _make_key(self, query: str) -> str:
        """生成缓存键"""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get(
        self,
        query: str,
    ) -> Optional[List[SearchResult]]:
        """
        获取缓存的搜索结果
        
        Args:
            query: 搜索查询
            
        Returns:
            缓存的结果，如果不存在或过期则返回 None
        """
        key = self._make_key(query)
        
        entry = self._cache.get(key)
        if entry is None:
            return None
        
        if entry.is_expired:
            del self._cache[key]
            self._access_order.remove(key)
            return None
        
        # 更新访问顺序
        self._access_order.remove(key)
        self._access_order.append(key)
        
        return entry.results
    
    def set(
        self,
        query: str,
        results: List[SearchResult],
        ttl: Optional[int] = None,
    ) -> None:
        """
        设置缓存
        
        Args:
            query: 搜索查询
            results: 搜索结果
            ttl: 过期时间 (秒)，None 使用默认值
        """
        key = self._make_key(query)
        
        # 容量检查
        while len(self._cache) >= self.max_size:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
        
        # 添加缓存
        self._cache[key] = CacheEntry(
            results=results,
            timestamp=time.time(),
            ttl=ttl or self.default_ttl,
        )
        self._access_order.append(key)
    
    def invalidate(self, query: str) -> None:
        """使特定查询的缓存失效"""
        key = self._make_key(query)
        if key in self._cache:
            del self._cache[key]
            self._access_order.remove(key)
    
    def clear(self) -> None:
        """清空所有缓存"""
        self._cache.clear()
        self._access_order.clear()
    
    def cleanup_expired(self) -> int:
        """清理过期条目"""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired
        ]
        
        for key in expired_keys:
            del self._cache[key]
            self._access_order.remove(key)
        
        return len(expired_keys)
    
    @property
    def size(self) -> int:
        return len(self._cache)
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __contains__(self, query: str) -> bool:
        return self.get(query) is not None
