"""
Brain Package - 核心大脑模块

动态动力学模型与自适应计算：
- Mamba/GRU 动力学预测
- ACT 自适应计算时间
- Halt Unit 停止决策
- 思考循环协调
- 情感调制 Mamba
- 记忆增强 Mamba (新增)
"""

from .dynamics_model import DynamicsModel, MambaBlock
from .act_controller import ACTController
from .halt_unit import HaltUnit
from .reasoning_loop import ReasoningLoop
from .modulated_mamba import (
    ModulatedMambaBlock,
    ModulatedMambaStack,
    MultiConditionModulatedMamba,
    MemoryAugmentedMambaBlock,
    MemoryAugmentedMambaStack,
)
from .modulated_dynamics import ModulatedDynamicsModel
from .modulated_reasoning_loop import ModulatedReasoningLoop

__all__ = [
    "DynamicsModel",
    "MambaBlock",
    "ACTController",
    "HaltUnit",
    "ReasoningLoop",
    # 情感调制
    "ModulatedMambaBlock",
    "ModulatedMambaStack",
    "MultiConditionModulatedMamba",
    "ModulatedDynamicsModel",
    "ModulatedReasoningLoop",
    # 记忆增强
    "MemoryAugmentedMambaBlock",
    "MemoryAugmentedMambaStack",
]
