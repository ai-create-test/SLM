"""
Brain Package - 核心大脑模块

动态动力学模型与自适应计算：
- Mamba/GRU 动力学预测
- ACT 自适应计算时间
- Halt Unit 停止决策
- 思考循环协调
"""

from .dynamics_model import DynamicsModel, MambaBlock
from .act_controller import ACTController
from .halt_unit import HaltUnit
from .reasoning_loop import ReasoningLoop

__all__ = [
    "DynamicsModel",
    "MambaBlock",
    "ACTController",
    "HaltUnit",
    "ReasoningLoop",
]
