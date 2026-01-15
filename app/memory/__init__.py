"""
Memory Package - 记忆系统 (重构版)

潜变量记忆与检索：
- 潜变量记忆库 (FAISS 索引)
- 问题导向检索
- Cross-Attention 融合
- 类图结构存储
"""

from .latent_memory_bank import LatentMemoryBank, MemoryConfig
from .query_retriever import QueryRetriever
from .cross_attention_fuser import CrossAttentionFuser
from .graph_memory import GraphMemory, GraphNode, GraphEdge

# 保留原有的兼容导入
from .embeddings import TokenEmbedding, RotaryPositionalEmbedding, CombinedEmbedding
from .text_chunker import TextChunker, TextChunk

__all__ = [
    # 新的记忆系统
    "LatentMemoryBank",
    "MemoryConfig",
    "QueryRetriever",
    "CrossAttentionFuser",
    "GraphMemory",
    "GraphNode",
    "GraphEdge",
    # 保留的原有组件
    "TokenEmbedding",
    "RotaryPositionalEmbedding",
    "CombinedEmbedding",
    "TextChunker",
    "TextChunk",
]
