"""
Modulation Package - 控制与调制层

深度情感/场景调制：
- AdaLN 自适应层归一化
- FiLM 特征线性调制
- 情感编码 (VAD-based)
- 场景编码
"""

from .adaln import AdaptiveLayerNorm, AdaLNBlock
from .film import FiLM, FiLMBlock
from .scene_encoder import SceneEncoder, SCENE_IDS

# 新版情感编码系统
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

# Alias for backward compatibility
EmotionEncoder = SemanticEmotionEncoder
EMOTION_IDS = LEGACY_EMOTION_IDS

__all__ = [
    # AdaLN
    "AdaptiveLayerNorm",
    "AdaLNBlock",
    # FiLM
    "FiLM",
    "FiLMBlock",
    # Emotion (New - Semantic)
    "SemanticEmotionEncoder",
    "UnifiedEmotionEncoder",
    "VADEncoder",
    "VADCoordinate",
    "VADLexicon",
    "get_default_lexicon",
    "get_vad",
    # Emotion (Legacy - Alias)
    "EmotionEncoder",
    "EMOTION_IDS",
    "LEGACY_EMOTION_IDS",
    # Scene
    "SceneEncoder",
    "SCENE_IDS",
]
