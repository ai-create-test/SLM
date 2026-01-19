"""
Interfaces Package - 统一接口定义

提供所有模块必须遵循的基类和配置系统。
"""

from .base_module import BaseModule, ModuleOutput, LatentVector, MemoryItem
from .config import Config, ModelConfig, TrainingConfig, PipelineConfig
from .config_loader import ConfigLoader, CloudConfig, ExtendedConfig
from .registry import Registry

__all__ = [
    "BaseModule",
    "ModuleOutput",
    "LatentVector",
    "MemoryItem",
    "Config",
    "ModelConfig",
    "TrainingConfig",
    "PipelineConfig",
    "Registry",
]

