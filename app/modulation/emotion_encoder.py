"""
Emotion Encoder - 情感状态编码器

本模块已重构。请使用新的 SemanticEmotionEncoder:

    from app.modulation import SemanticEmotionEncoder
    encoder = SemanticEmotionEncoder(d_emotion=128)
    vec = encoder("happy")     # 英文
    vec = encoder("开心")      # 中文

向后兼容说明：
- EmotionEncoder 现在是 SemanticEmotionEncoder 的别名
- EMOTION_IDS 保留用于 legacy 代码
"""

# 重导出新模块 (向后兼容)
from .emotion import (
    VADCoordinate,
    VADLexicon,
    VADEncoder,
    SemanticEmotionEncoder,
    UnifiedEmotionEncoder,
    get_default_lexicon,
    get_vad,
    LEGACY_EMOTION_IDS,
)

# Backward compatibility aliases
EmotionEncoder = SemanticEmotionEncoder
EMOTION_IDS = LEGACY_EMOTION_IDS
ID_TO_EMOTION = {v: k for k, v in EMOTION_IDS.items()}

__all__ = [
    "SemanticEmotionEncoder",
    "UnifiedEmotionEncoder",
    "VADEncoder",
    "VADCoordinate",
    "VADLexicon",
    "get_default_lexicon",
    "get_vad",
    # Legacy aliases
    "EmotionEncoder",
    "EMOTION_IDS",
    "ID_TO_EMOTION",
    "LEGACY_EMOTION_IDS",
]
