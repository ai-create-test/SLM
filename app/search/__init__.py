"""
Search Package - 联网搜索模块

实时外部知识接入：
- 搜索接口抽象
- Web 搜索实现
- 知识注入器
- 搜索缓存
"""

from .search_interface import SearchInterface, SearchResult
from .web_search import WebSearch
from .knowledge_injector import KnowledgeInjector
from .cache import SearchCache

__all__ = [
    "SearchInterface",
    "SearchResult",
    "WebSearch",
    "KnowledgeInjector",
    "SearchCache",
]
