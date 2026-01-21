"""
Interfaces Package - 统一接口定义

提供所有模块必须遵循的基类和配置系统。
"""

from .base_module import BaseModule, ModuleOutput, LatentVector, MemoryItem, HierarchicalLatent
from .config import Config, ModelConfig, TrainingConfig, PipelineConfig
from .config_loader import ConfigLoader, CloudConfig, ExtendedConfig
from .registry import Registry
from .unified_latent import (
    UnifiedLatent,
    SymbolAnchor,
    SymbolAnchors,
    StructureRef,
    StructureSlot,
    PrecisionConfig,
    detect_latent_type,
    to_unified,
    to_hierarchical,
    to_legacy,
)

__all__ = [
    # Base
    "BaseModule",
    "ModuleOutput",
    "LatentVector",
    "MemoryItem",
    # AMHVQ+
    "HierarchicalLatent",
    "UnifiedLatent",
    "SymbolAnchor",
    "SymbolAnchors",
    "StructureRef",
    "StructureSlot",
    "PrecisionConfig",
    # Config
    "Config",
    "ModelConfig",
    "TrainingConfig",
    "PipelineConfig",
    "ConfigLoader",
    "CloudConfig",
    "ExtendedConfig",
    # Registry
    "Registry",
    # Utils
    "detect_latent_type",
    "to_unified",
    "to_hierarchical",
    "to_legacy",
]
