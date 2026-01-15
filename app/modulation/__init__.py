"""
Modulation Package - 控制与调制层

深度情感/场景调制：
- AdaLN 自适应层归一化
- FiLM 特征线性调制
- 情感/场景编码器
"""

from .adaln import AdaptiveLayerNorm, AdaLNBlock
from .film import FiLM, FiLMBlock
from .emotion_encoder import EmotionEncoder, EMOTION_IDS
from .scene_encoder import SceneEncoder, SCENE_IDS

__all__ = [
    "AdaptiveLayerNorm",
    "AdaLNBlock",
    "FiLM",
    "FiLMBlock",
    "EmotionEncoder",
    "SceneEncoder",
    "EMOTION_IDS",
    "SCENE_IDS",
]
