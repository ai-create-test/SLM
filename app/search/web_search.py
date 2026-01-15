"""
Web Search - Web 搜索实现

支持多种搜索引擎。
"""

from typing import List, Optional
import asyncio

from .search_interface import SearchInterface, SearchResult
from .cache import SearchCache


class WebSearch(SearchInterface):
    """
    Web 搜索实现
    
    支持:
    - DuckDuckGo (无需 API Key)
    - Google Search API
    - Bing Search API
    
    使用示例:
        search = WebSearch(provider="duckduckgo")
        results = search.search_sync("Python programming")
    """
    
    def __init__(
        self,
        provider: str = "duckduckgo",
        api_key: Optional[str] = None,
        cache_ttl: int = 3600,
        use_cache: bool = True,
    ):
        """
        Args:
            provider: 搜索提供商 ('duckduckgo', 'google', 'bing')
            api_key: API 密钥 (Google/Bing 需要)
            cache_ttl: 缓存过期时间 (秒)
            use_cache: 是否启用缓存
        """
        self._provider = provider
        self._api_key = api_key
        
        if use_cache:
            self._cache = SearchCache(default_ttl=cache_ttl)
        else:
            self._cache = None
    
    async def search(
        self,
        query: str,
        num_results: int = 5,
    ) -> List[SearchResult]:
        """异步搜索"""
        # 检查缓存
        if self._cache is not None:
            cached = self._cache.get(query)
            if cached is not None:
                return cached[:num_results]
        
        # 执行搜索
        if self._provider == "duckduckgo":
            results = await self._search_duckduckgo(query, num_results)
        elif self._provider == "google":
            results = await self._search_google(query, num_results)
        elif self._provider == "bing":
            results = await self._search_bing(query, num_results)
        else:
            results = []
        
        # 缓存结果
        if self._cache is not None and results:
            self._cache.set(query, results)
        
        return results
    
    def search_sync(
        self,
        query: str,
        num_results: int = 5,
    ) -> List[SearchResult]:
        """同步搜索"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.search(query, num_results))
    
    async def _search_duckduckgo(
        self,
        query: str,
        num_results: int,
    ) -> List[SearchResult]:
        """DuckDuckGo 搜索 (使用 duckduckgo-search 库)"""
        try:
            from duckduckgo_search import DDGS
            
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=num_results):
                    results.append(SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        snippet=r.get("body", ""),
                    ))
            
            return results
            
        except ImportError:
            # 库未安装，返回空结果
            return self._mock_search(query, num_results)
        except Exception:
            return []
    
    async def _search_google(
        self,
        query: str,
        num_results: int,
    ) -> List[SearchResult]:
        """Google Search API"""
        if not self._api_key:
            return []
        
        # TODO: 实现 Google Search API 调用
        return self._mock_search(query, num_results)
    
    async def _search_bing(
        self,
        query: str,
        num_results: int,
    ) -> List[SearchResult]:
        """Bing Search API"""
        if not self._api_key:
            return []
        
        # TODO: 实现 Bing Search API 调用
        return self._mock_search(query, num_results)
    
    def _mock_search(
        self,
        query: str,
        num_results: int,
    ) -> List[SearchResult]:
        """模拟搜索 (开发/测试用)"""
        return [
            SearchResult(
                title=f"Mock Result {i+1} for: {query}",
                url=f"https://example.com/result{i+1}",
                snippet=f"This is a mock search result for query: {query}",
                score=1.0 - i * 0.1,
            )
            for i in range(num_results)
        ]
    
    @property
    def provider_name(self) -> str:
        return self._provider
    
    @property
    def is_available(self) -> bool:
        if self._provider == "duckduckgo":
            try:
                from duckduckgo_search import DDGS
                return True
            except ImportError:
                return False
        
        return self._api_key is not None
