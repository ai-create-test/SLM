"""
Memory Module - 语义嵌入与长文本处理

核心功能:
1. 长文本分块处理 (TextChunker)
2. Token嵌入与位置编码 (TokenEmbedding + RoPE)
3. 统一编码入口 (MemoryEncoder)
4. 上下文融合 - 情感/场景注入 (ContextFusion)

导出类:
- TextChunker: 文本分块器
- TokenAwareChunker: Token感知分块器
- TextChunk: 文本块数据结构
- CombinedEmbedding: 组合嵌入层
- MemoryEncoder: 整合编码器
- create_memory_encoder: 预配置工厂函数
- ContextAwareEmbedding: 上下文感知嵌入层
- ContextFusion: 门控融合层
"""

from .text_chunker import (
    TextChunker,
    TokenAwareChunker,
    TextChunk,
    ChunkConfig,
    ChunkStrategy,
    BaseChunker,
)

from .embeddings import (
    TokenEmbedding,
    PositionalEncoding,
    RotaryPositionalEmbedding,
    CombinedEmbedding,
    EmbeddingConfig,
)

from .memory_encoder import (
    MemoryEncoder,
    EncodingResult,
    create_memory_encoder,
)

from .fusion import (
    EmotionEmbedding,
    SceneEmbedding,
    ContextFusion,
    ContextAwareEmbedding,
    RMSNorm,
    FusionConfig,
    create_context_aware_embedding,
    EMOTION_IDS,
    SCENE_IDS,
)


__all__ = [
    # 分块
    "TextChunker",
    "TokenAwareChunker",
    "TextChunk",
    "ChunkConfig",
    "ChunkStrategy",
    "BaseChunker",
    # 嵌入
    "TokenEmbedding",
    "PositionalEncoding",
    "RotaryPositionalEmbedding",
    "CombinedEmbedding",
    "EmbeddingConfig",
    # 编码器
    "MemoryEncoder",
    "EncodingResult",
    "create_memory_encoder",
    # 上下文融合
    "EmotionEmbedding",
    "SceneEmbedding",
    "ContextFusion",
    "ContextAwareEmbedding",
    "RMSNorm",
    "FusionConfig",
    "create_context_aware_embedding",
    "EMOTION_IDS",
    "SCENE_IDS",
]

__version__ = "0.2.0"

