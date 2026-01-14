"""
Core Module - 核心LLM架构
包含Tokenizer和其他核心组件

导出说明：
- BaseTokenizer: 分词器抽象基类
- TokenizerResult: 分词结果容器
- BatchTokenizerResult: 批量分词结果
- BPETokenizer: BPE分词器实现
- TokenizerFactory: 分词器工厂
- get_tokenizer: 便捷函数

问题导向注意力：
- TokenAttentionMixin: 注意力权重计算混入类
- AttentionConfig: 注意力配置
- AttentionStrategy: 计算策略枚举
- QueryFocusedAttention: 独立的注意力计算器
"""

from .tokenizer_base import (
    BaseTokenizer,
    TokenizerResult,
    BatchTokenizerResult,
    PaddingStrategy,
    TruncationStrategy,
)

from .tokenizer_attention import (
    TokenAttentionMixin,
    AttentionConfig,
    AttentionStrategy,
    QueryFocusedAttention,
)

from .bpe_tokenizer import BPETokenizer

from .tokenizer_factory import (
    TokenizerFactory,
    get_tokenizer,
)


__all__ = [
    # 基础类
    "BaseTokenizer",
    "TokenizerResult",
    "BatchTokenizerResult",
    "PaddingStrategy",
    "TruncationStrategy",
    # 注意力
    "TokenAttentionMixin",
    "AttentionConfig",
    "AttentionStrategy",
    "QueryFocusedAttention",
    # BPE实现
    "BPETokenizer",
    # 工厂
    "TokenizerFactory",
    "get_tokenizer",
]

__version__ = "0.1.0"
