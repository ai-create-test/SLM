"""
Inference Module - 推理接口

Phase 16: 推理接口
"""

from .amhvq_inference import (
    AMHVQInference,
    InferenceConfig,
    InferenceOutput,
)

__all__ = [
    "AMHVQInference",
    "InferenceConfig",
    "InferenceOutput",
]
