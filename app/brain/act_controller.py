"""
ACT Controller - 自适应计算时间控制器

核心功能：根据问题难度动态分配计算资源

参考论文: Adaptive Computation Time for Recurrent Neural Networks (Graves, 2016)

设计原理:
1. 简单问题：1次思考 (System 1, 直觉)
2. 复杂问题：N次思考 (System 2, 慢思考)
3. 模型自动学习何时停止
"""

from typing import Tuple, Optional, List
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces.base_module import BaseModule, ModuleOutput
from ..interfaces.config import ModelConfig
from ..interfaces.registry import Registry
from .halt_unit import HaltUnit


@dataclass
class ACTOutput(ModuleOutput):
    """ACT 输出"""
    final_state: torch.Tensor         # 最终状态 [batch, d_model]
    num_steps: torch.Tensor           # 实际思考步数 [batch]
    halt_probs: List[torch.Tensor]    # 每步的停止概率
    remainders: torch.Tensor          # 剩余概率 (用于正则化)
    ponder_cost: torch.Tensor         # 思考代价 (正则化损失)


@Registry.register("brain", "act")
class ACTController(BaseModule):
    """
    自适应计算时间控制器
    
    实现动态长度的思考循环。
    
    核心机制:
    1. 每步计算停止概率 p_t
    2. 累积概率 R = Σp_t
    3. 当 R ≥ 1-ε 时停止
    4. 最终输出是各步状态的加权和
    
    使用示例:
        act = ACTController(
            d_model=768,
            max_steps=10,
        )
        
        # 执行 ACT
        initial_state = encoder(x)
        output = act(initial_state, step_fn=dynamics.step)
        final = output.final_state  # 动态深度处理后的状态
    """
    
    MODULE_TYPE = "brain"
    
    def __init__(
        self,
        d_model: int = 768,
        max_steps: int = 10,
        halt_threshold: float = 0.99,
        epsilon: float = 0.01,
        ponder_cost: float = 0.01,
    ):
        """
        Args:
            d_model: 状态维度
            max_steps: 最大思考步数
            halt_threshold: 停止阈值 (累积概率超过此值停止)
            epsilon: 最小剩余概率
            ponder_cost: 思考代价系数 (正则化，鼓励效率)
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_steps = max_steps
        self.halt_threshold = halt_threshold
        self.epsilon = epsilon
        self.ponder_cost_coef = ponder_cost
        
        # 停止决策单元
        self.halt_unit = HaltUnit(d_model)
    
    def forward(
        self,
        initial_state: torch.Tensor,
        step_fn: callable,
        context: Optional[torch.Tensor] = None,
    ) -> ACTOutput:
        """
        执行自适应计算时间循环
        
        Args:
            initial_state: 初始状态 [batch, d_model]
            step_fn: 单步计算函数 f(state, context) -> new_state
            context: 可选的上下文信息 (如情感、场景)
            
        Returns:
            ACTOutput
        """
        batch_size = initial_state.shape[0]
        device = initial_state.device
        dtype = initial_state.dtype
        
        # 初始化
        state = initial_state
        halted = torch.zeros(batch_size, dtype=torch.bool, device=device)
        cumulative_prob = torch.zeros(batch_size, 1, device=device)
        
        # 加权状态累加器
        weighted_state = torch.zeros_like(initial_state)
        
        # 记录
        halt_probs = []
        step_counts = torch.zeros(batch_size, device=device)
        remainders = torch.ones(batch_size, 1, device=device)
        
        for step in range(self.max_steps):
            # 获取当前停止概率
            halt_prob = self.halt_unit(state)  # [batch, 1]
            halt_probs.append(halt_prob)
            
            # 更新累积概率
            new_cumulative = cumulative_prob + halt_prob
            
            # 判断哪些样本应该停止
            should_halt = (new_cumulative >= self.halt_threshold) & (~halted)
            
            # 计算权重
            # 对于继续的样本：使用 halt_prob
            # 对于停止的样本：使用剩余概率
            weight = torch.where(
                should_halt.unsqueeze(-1).expand_as(halt_prob),
                remainders,  # 停止时使用剩余概率
                halt_prob,   # 继续时使用当前概率
            )
            
            # 累加加权状态
            weighted_state = weighted_state + weight * state
            
            # 更新记录
            step_counts = torch.where(
                ~halted,
                torch.full_like(step_counts, step + 1),
                step_counts,
            )
            
            # 更新停止状态
            halted = halted | should_halt.squeeze(-1)
            
            # 更新剩余概率
            remainders = 1.0 - new_cumulative
            cumulative_prob = new_cumulative
            
            # 如果所有样本都停止，提前退出
            if halted.all():
                break
            
            # 执行一步计算
            if context is not None:
                state = step_fn(state, context)
            else:
                state = step_fn(state)
        
        # 处理未停止的样本（使用最后的剩余概率）
        if not halted.all():
            weighted_state = weighted_state + remainders * state
        
        # 计算思考代价 (正则化项)
        # 鼓励模型尽早停止
        ponder_cost = self.ponder_cost_coef * (
            step_counts.float().mean() + remainders.abs().mean()
        )
        
        return ACTOutput(
            data=weighted_state,
            final_state=weighted_state,
            num_steps=step_counts,
            halt_probs=halt_probs,
            remainders=remainders,
            ponder_cost=ponder_cost,
        )
    
    def get_ponder_loss(self, output: ACTOutput) -> torch.Tensor:
        """
        获取思考代价损失
        
        用于训练时正则化，防止模型总是使用最大步数。
        """
        return output.ponder_cost
    
    @classmethod
    def from_config(cls, config: ModelConfig, **kwargs) -> "ACTController":
        """从配置创建实例"""
        return cls(
            d_model=config.d_model,
            max_steps=config.max_think_steps,
            halt_threshold=config.halt_threshold,
            ponder_cost=config.ponder_cost,
            **kwargs,
        )


class SimpleThinkingStep(nn.Module):
    """
    简单的思考步骤
    
    用于与 ACT 配合的单步计算。
    """
    
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        
        d_ff = d_ff or d_model * 4
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
    
    def forward(
        self, 
        state: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            state: 当前状态 [batch, d_model]
            context: 可选上下文 [batch, d_context]
        """
        # Pre-LN 风格
        x = self.norm1(state)
        
        # 融合上下文 (如果提供)
        if context is not None:
            x = x + context
        
        # FFN
        x = state + self.ff(self.norm2(x))
        
        return x
