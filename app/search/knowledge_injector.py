"""
Knowledge Injector - 知识注入器

将搜索结果编码并注入记忆系统。
"""

from typing import List, Optional
import torch

from .search_interface import SearchResult


class KnowledgeInjector:
    """
    知识注入器
    
    将搜索结果转换为潜向量并注入记忆库。
    
    使用示例:
        injector = KnowledgeInjector(encoder, memory_bank)
        
        # 搜索并注入
        results = await search.search("Python programming")
        injector.inject(results)
    """
    
    def __init__(
        self,
        encoder,  # ParagraphEncoder
        memory_bank,  # LatentMemoryBank
        importance_boost: float = 0.5,
    ):
        """
        Args:
            encoder: 段落编码器
            memory_bank: 记忆库
            importance_boost: 外部知识的重要度提升
        """
        self.encoder = encoder
        self.memory_bank = memory_bank
        self.importance_boost = importance_boost
    
    def inject(
        self,
        search_results: List[SearchResult],
        base_importance: float = 0.8,
    ) -> List[int]:
        """
        注入搜索结果
        
        Args:
            search_results: 搜索结果列表
            base_importance: 基础重要度
            
        Returns:
            注入的记忆索引列表
        """
        indices = []
        
        for result in search_results:
            # 提取文本内容
            text = self._extract_text(result)
            if not text:
                continue
            
            # 编码
            with torch.no_grad():
                output = self.encoder(text)
                latent = output.latent.vector
            
            # 计算重要度
            importance = base_importance + result.score * self.importance_boost
            importance = min(importance, 1.0)
            
            # 注入记忆库
            idx = self.memory_bank.add(
                vector=latent,
                content=text,
                importance=importance,
                metadata={
                    "source": "web_search",
                    "url": result.url,
                    "title": result.title,
                },
            )
            indices.append(idx)
        
        return indices
    
    def _extract_text(self, result: SearchResult) -> str:
        """提取搜索结果的文本内容"""
        if result.content:
            return result.content
        
        # 组合标题和摘要
        parts = []
        if result.title:
            parts.append(result.title)
        if result.snippet:
            parts.append(result.snippet)
        
        return ". ".join(parts)
    
    async def inject_from_query(
        self,
        search_engine,  # SearchInterface
        query: str,
        num_results: int = 5,
    ) -> List[int]:
        """
        从查询直接搜索并注入
        """
        results = await search_engine.search(query, num_results)
        return self.inject(results)
