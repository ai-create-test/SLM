"""
Halt Unit - 停止决策神经元

核心功能：决定何时停止思考循环

设计原理:
- 输出 [0, 1] 的停止概率
- 累积概率超过阈值时停止
- 实现 System 1 (快) vs System 2 (慢) 的认知
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaltUnit(nn.Module):
    """
    停止决策神经元
    
    根据当前思维状态输出停止概率。
    
    公式:
        p_halt = σ(W · state + b)
    
    停止条件:
        Σ p_halt > threshold  或  step > max_steps
    
    使用示例:
        halt_unit = HaltUnit(d_model=768)
        
        cumulative_prob = 0.0
        for step in range(max_steps):
            state = think(state)
            halt_prob = halt_unit(state)
            cumulative_prob += halt_prob
            
            if cumulative_prob > 0.99:
                break
    """
    
    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        bias_init: float = 1.0,
    ):
        """
        Args:
            d_model: 输入状态维度
            hidden_dim: 隐藏层维度 (None 表示直接投影)
            bias_init: 偏置初始值 (正值倾向于继续思考)
        """
        super().__init__()
        
        if hidden_dim is not None:
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.net = nn.Linear(d_model, 1)
        
        # 初始化偏置使模型倾向于继续思考
        with torch.no_grad():
            if isinstance(self.net, nn.Linear):
                self.net.bias.fill_(-bias_init)
            else:
                self.net[-1].bias.fill_(-bias_init)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        计算停止概率
        
        Args:
            state: 当前思维状态 [batch, d_model]
            
        Returns:
            halt_prob: 停止概率 [batch, 1]
        """
        logit = self.net(state)
        halt_prob = torch.sigmoid(logit)
        return halt_prob
    
    def should_halt(
        self,
        state: torch.Tensor,
        cumulative_prob: torch.Tensor,
        threshold: float = 0.99,
    ) -> tuple:
        """
        判断是否应该停止
        
        Args:
            state: 当前状态
            cumulative_prob: 累积停止概率
            threshold: 停止阈值
            
        Returns:
            should_halt: 是否停止
            new_cumulative: 新的累积概率
            halt_prob: 当前步的停止概率
        """
        halt_prob = self.forward(state)
        new_cumulative = cumulative_prob + halt_prob
        should_halt = (new_cumulative >= threshold).squeeze(-1)
        
        return should_halt, new_cumulative, halt_prob


# 用于类型提示
from typing import Optional
