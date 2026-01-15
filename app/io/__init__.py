"""
IO Package - 输入/输出层

语义压缩机核心模块：
- 段落级编码 (VQ-VAE)
- 段落级解码
- 向量量化码本
- 语义段落分割
- Tokenizer 包装器
- 预训练语言模型
"""

from .paragraph_encoder import ParagraphEncoder
from .paragraph_decoder import ParagraphDecoder
from .vq_codebook import VQCodebook, VQOutput
from .semantic_segmenter import SemanticSegmenter
from .tokenizer_wrapper import TokenizerWrapper, FallbackTokenizer, get_tokenizer, TokenizerOutput
from .base_lm import BaseLM, FallbackLM, get_base_lm, LMOutput

__all__ = [
    "ParagraphEncoder",
    "ParagraphDecoder",
    "VQCodebook",
    "VQOutput",
    "SemanticSegmenter",
    "TokenizerWrapper",
    "FallbackTokenizer",
    "get_tokenizer",
    "TokenizerOutput",
    "BaseLM",
    "FallbackLM",
    "get_base_lm",
    "LMOutput",
]


