"""
Trajectory Logger - 推理轨迹记录器

记录推理过程的每一步，用于回溯和分析。
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time
import torch


@dataclass
class StepRecord:
    """单步记录"""
    step: int                           # 步骤编号
    state: torch.Tensor                 # 状态快照
    halt_prob: float                    # 停止概率
    memories_used: List[int]            # 使用的记忆索引
    timestamp: float                    # 时间戳
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "halt_prob": self.halt_prob,
            "memories_used": self.memories_used,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class TrajectoryLogger:
    """
    推理轨迹记录器
    
    记录推理过程中的所有状态，用于：
    1. 回溯到之前的状态
    2. 分析推理路径
    3. 调试和可视化
    
    使用示例:
        logger = TrajectoryLogger()
        
        # 记录每一步
        for step in range(max_steps):
            state = brain.step(state)
            logger.log_step(step, state, halt_prob, memories)
        
        # 获取轨迹
        trajectory = logger.get_trajectory()
        
        # 回溯到特定步骤
        old_state = logger.get_state(step=3)
    """
    
    def __init__(self, max_length: int = 100):
        """
        Args:
            max_length: 最大记录长度 (防止内存溢出)
        """
        self.max_length = max_length
        self._records: List[StepRecord] = []
        self._start_time: Optional[float] = None
    
    def start(self) -> None:
        """开始新的推理轨迹"""
        self._records.clear()
        self._start_time = time.time()
    
    def log_step(
        self,
        step: int,
        state: torch.Tensor,
        halt_prob: float = 0.0,
        memories_used: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        记录一步
        
        Args:
            step: 步骤编号
            state: 当前状态
            halt_prob: 停止概率
            memories_used: 使用的记忆索引
            metadata: 额外元数据
        """
        record = StepRecord(
            step=step,
            state=state.detach().clone(),
            halt_prob=halt_prob,
            memories_used=memories_used or [],
            timestamp=time.time(),
            metadata=metadata or {},
        )
        
        self._records.append(record)
        
        # 限制长度
        if len(self._records) > self.max_length:
            self._records.pop(0)
    
    def get_trajectory(self) -> List[StepRecord]:
        """获取完整轨迹"""
        return self._records.copy()
    
    def get_state(self, step: int) -> Optional[torch.Tensor]:
        """
        获取特定步骤的状态
        
        Args:
            step: 步骤编号
            
        Returns:
            状态张量，如果不存在则返回 None
        """
        for record in self._records:
            if record.step == step:
                return record.state.clone()
        return None
    
    def get_last_state(self) -> Optional[torch.Tensor]:
        """获取最后一步的状态"""
        if self._records:
            return self._records[-1].state.clone()
        return None
    
    def get_step_count(self) -> int:
        """获取总步数"""
        return len(self._records)
    
    def get_duration(self) -> float:
        """获取总耗时"""
        if not self._records or self._start_time is None:
            return 0.0
        return self._records[-1].timestamp - self._start_time
    
    def get_average_halt_prob(self) -> float:
        """获取平均停止概率"""
        if not self._records:
            return 0.0
        return sum(r.halt_prob for r in self._records) / len(self._records)
    
    def clear(self) -> None:
        """清空记录"""
        self._records.clear()
        self._start_time = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "step_count": self.get_step_count(),
            "duration": self.get_duration(),
            "average_halt_prob": self.get_average_halt_prob(),
            "steps": [r.to_dict() for r in self._records],
        }
    
    def __len__(self) -> int:
        return len(self._records)
