"""
NeuralFlow LLM Framework

一个前沿的 LLM 架构实验框架，实现段落级语义推理。

核心特性:
- 段落级 VQ-VAE 语义压缩
- Mamba 动态动力学模型
- 自适应计算时间 (ACT)
- 潜空间记忆与检索
- 深度情感/场景调制 (AdaLN)
- 自我回溯与反思
- 联网搜索知识注入
"""

__version__ = "0.1.0"
__author__ = "NeuralFlow Team"

# 核心接口
from .interfaces import (
    BaseModule,
    ModuleOutput,
    Config,
    ModelConfig,
    TrainingConfig,
    PipelineConfig,
    Registry,
)

from .interfaces.base_module import LatentVector, MemoryItem

# IO 层
from .io import (
    ParagraphEncoder,
    ParagraphDecoder,
    VQCodebook,
    SemanticSegmenter,
)

# 核心大脑
from .brain import (
    DynamicsModel,
    MambaBlock,
    ACTController,
    HaltUnit,
    ReasoningLoop,
)

# 记忆系统
from .memory import (
    LatentMemoryBank,
    QueryRetriever,
    CrossAttentionFuser,
    GraphMemory,
)

# 调制层
from .modulation import (
    AdaptiveLayerNorm,
    FiLM,
    EmotionEncoder,
    SceneEncoder,
)

# 自省模块
from .reflection import (
    TrajectoryLogger,
    Backtracker,
    SelfCritic,
)

# 搜索模块
from .search import (
    SearchInterface,
    WebSearch,
    KnowledgeInjector,
)

# 流水线
from .pipeline import NeuralFlowPipeline, PipelineOutput

# 保留的原有组件
from .core import BPETokenizer, TokenizerFactory

__all__ = [
    # 版本
    "__version__",
    
    # 接口
    "BaseModule",
    "ModuleOutput",
    "Config",
    "ModelConfig",
    "TrainingConfig",
    "PipelineConfig",
    "Registry",
    "LatentVector",
    "MemoryItem",
    
    # IO
    "ParagraphEncoder",
    "ParagraphDecoder",
    "VQCodebook",
    "SemanticSegmenter",
    
    # Brain
    "DynamicsModel",
    "MambaBlock",
    "ACTController",
    "HaltUnit",
    "ReasoningLoop",
    
    # Memory
    "LatentMemoryBank",
    "QueryRetriever",
    "CrossAttentionFuser",
    "GraphMemory",
    
    # Modulation
    "AdaptiveLayerNorm",
    "FiLM",
    "EmotionEncoder",
    "SceneEncoder",
    
    # Reflection
    "TrajectoryLogger",
    "Backtracker",
    "SelfCritic",
    
    # Search
    "SearchInterface",
    "WebSearch",
    "KnowledgeInjector",
    
    # Pipeline
    "NeuralFlowPipeline",
    "PipelineOutput",
    
    # Legacy
    "BPETokenizer",
    "TokenizerFactory",
]
