"""
Training Package - 训练基础设施

包含:
- 数据加载器
- VQ-VAE 训练器
- Dynamics 训练器
- 统一训练器
- 训练阶段
"""

from .data_loader import ParagraphDataset, SequenceDataset, ParagraphDataLoader
from .vqvae_trainer import VQVAETrainer
from .dynamics_trainer import DynamicsTrainer
from .unified_trainer import UnifiedTrainer, train_model
from .training_stages import (
    VQVAEStage,
    DynamicsStage,
    EmotionStage,
    FinetuneStage,
    StageResult,
)

__all__ = [
    # Data
    "ParagraphDataset",
    "SequenceDataset",
    "ParagraphDataLoader",
    # Legacy trainers
    "VQVAETrainer",
    "DynamicsTrainer",
    # Unified training
    "UnifiedTrainer",
    "train_model",
    # Stages
    "VQVAEStage",
    "DynamicsStage",
    "EmotionStage",
    "FinetuneStage",
    "StageResult",
]
