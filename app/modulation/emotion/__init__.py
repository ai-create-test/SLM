"""
Emotion Module - 情感编码系统

统一的情感编码接口，支持任意自然语言情感描述。
"""

from .vad_lexicon import VADCoordinate, VADLexicon, get_default_lexicon, get_vad
from .vad_encoder import VADEncoder
from .semantic_retriever import SemanticRetriever, RetrievalResult
from .trainable_encoder import (
    TrainableEmotionEncoder,
    EmotionEncoderTrainer,
    TrainingConfig,
    CharacterCNN,
)
from .unified import SemanticEmotionEncoder, UnifiedEmotionEncoder, LEGACY_EMOTION_IDS

__all__ = [
    # Core classes
    "VADCoordinate",
    "VADLexicon",
    "VADEncoder",
    "SemanticEmotionEncoder",
    "SemanticRetriever",
    "RetrievalResult",
    # Trainable (Tier 3)
    "TrainableEmotionEncoder",
    "EmotionEncoderTrainer",
    "TrainingConfig",
    "CharacterCNN",
    # Aliases
    "UnifiedEmotionEncoder",
    # Utilities
    "get_default_lexicon",
    "get_vad",
    # Legacy
    "LEGACY_EMOTION_IDS",
]
