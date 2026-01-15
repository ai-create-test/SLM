"""
Backtracker - 回溯执行器

当检测到推理错误时，回退到之前的状态重新推理。
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from .trajectory_logger import TrajectoryLogger, StepRecord


class Backtracker(nn.Module):
    """
    回溯执行器
    
    功能：
    1. 检测是否需要回溯
    2. 选择回溯目标步骤
    3. 恢复状态并重新推理
    
    使用示例:
        backtracker = Backtracker(d_model=768)
        
        # 检测是否需要回溯
        if backtracker.should_backtrack(output, self_score):
            step = backtracker.find_backtrack_target(trajectory)
            state = trajectory.get_state(step)
            # 从 state 重新推理
    """
    
    def __init__(
        self,
        d_model: int = 768,
        backtrack_threshold: float = 0.3,
        max_backtracks: int = 3,
    ):
        """
        Args:
            d_model: 状态维度
            backtrack_threshold: 回溯阈值 (自评分低于此值时回溯)
            max_backtracks: 最大回溯次数
        """
        super().__init__()
        
        self.d_model = d_model
        self.backtrack_threshold = backtrack_threshold
        self.max_backtracks = max_backtracks
        
        self._backtrack_count = 0
        
        # 回溯点选择网络
        self.target_selector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
    
    def should_backtrack(
        self,
        current_output: Optional[torch.Tensor] = None,
        self_eval_score: Optional[float] = None,
        error_detected: bool = False,
    ) -> bool:
        """
        判断是否应该回溯
        
        Args:
            current_output: 当前输出状态
            self_eval_score: 自我评估分数 (0-1)
            error_detected: 是否检测到错误
            
        Returns:
            是否应该回溯
        """
        # 检查回溯次数限制
        if self._backtrack_count >= self.max_backtracks:
            return False
        
        # 明确检测到错误
        if error_detected:
            return True
        
        # 自评分过低
        if self_eval_score is not None and self_eval_score < self.backtrack_threshold:
            return True
        
        return False
    
    def find_backtrack_target(
        self,
        trajectory: TrajectoryLogger,
        current_state: Optional[torch.Tensor] = None,
    ) -> int:
        """
        找到回溯的目标步骤
        
        Args:
            trajectory: 推理轨迹
            current_state: 当前状态 (用于计算目标)
            
        Returns:
            目标步骤编号
        """
        if len(trajectory) == 0:
            return 0
        
        records = trajectory.get_trajectory()
        
        if current_state is not None and len(records) > 1:
            # 使用神经网络选择
            scores = []
            for record in records[:-1]:  # 不包括最后一步
                score = self.target_selector(record.state)
                scores.append(score.item())
            
            # 选择分数最高的
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            return records[best_idx].step
        
        # 简单策略：回退到上一步
        return records[-2].step if len(records) > 1 else 0
    
    def backtrack_to(
        self,
        trajectory: TrajectoryLogger,
        target_step: int,
    ) -> Optional[torch.Tensor]:
        """
        回溯到指定步骤
        
        Args:
            trajectory: 推理轨迹
            target_step: 目标步骤
            
        Returns:
            目标步骤的状态
        """
        state = trajectory.get_state(target_step)
        
        if state is not None:
            self._backtrack_count += 1
        
        return state
    
    def reset(self) -> None:
        """重置回溯计数"""
        self._backtrack_count = 0
    
    @property
    def backtrack_count(self) -> int:
        return self._backtrack_count
