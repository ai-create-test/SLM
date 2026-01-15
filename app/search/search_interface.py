"""
Search Interface - 搜索抽象接口

定义搜索模块的统一接口。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class SearchResult:
    """搜索结果"""
    title: str                          # 标题
    url: str                            # URL
    snippet: str                        # 摘要
    content: Optional[str] = None       # 完整内容 (可选)
    score: float = 1.0                  # 相关性分数
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"SearchResult(title='{self.title[:30]}...', score={self.score:.2f})"


class SearchInterface(ABC):
    """
    搜索接口抽象类
    
    所有搜索实现必须继承此类。
    
    使用示例:
        class GoogleSearch(SearchInterface):
            async def search(self, query, num_results):
                ...
                return results
    """
    
    @abstractmethod
    async def search(
        self,
        query: str,
        num_results: int = 5,
    ) -> List[SearchResult]:
        """
        执行搜索
        
        Args:
            query: 搜索查询
            num_results: 返回结果数量
            
        Returns:
            SearchResult 列表
        """
        pass
    
    @abstractmethod
    def search_sync(
        self,
        query: str,
        num_results: int = 5,
    ) -> List[SearchResult]:
        """
        同步搜索
        """
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """搜索提供商名称"""
        pass
    
    @property
    def is_available(self) -> bool:
        """检查服务是否可用"""
        return True
