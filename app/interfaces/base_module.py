"""
Base Module - 所有模块的抽象基类

设计原则:
1. 统一的初始化接口
2. 标准化的输入输出格式
3. 可序列化的状态管理
4. 训练/推理模式切换
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, TypeVar, Generic
import torch
import torch.nn as nn


@dataclass
class ModuleOutput:
    """
    模块输出的标准容器
    
    所有模块的输出都应该返回此类或其子类，
    确保流水线中数据流的一致性。
    """
    data: torch.Tensor                          # 主要输出数据
    auxiliary: Dict[str, Any] = field(default_factory=dict)  # 辅助信息
    metadata: Dict[str, Any] = field(default_factory=dict)   # 元数据
    
    def to_device(self, device: torch.device) -> "ModuleOutput":
        """移动到指定设备"""
        return ModuleOutput(
            data=self.data.to(device),
            auxiliary={
                k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in self.auxiliary.items()
            },
            metadata=self.metadata,
        )


class BaseModule(nn.Module, ABC):
    """
    所有模块的抽象基类
    
    定义了统一的接口规范，确保模块间的互操作性。
    
    使用示例:
        class MyModule(BaseModule):
            def __init__(self, config):
                super().__init__(config)
                self.layer = nn.Linear(config.d_model, config.d_model)
            
            def forward(self, x, **kwargs) -> ModuleOutput:
                return ModuleOutput(data=self.layer(x))
            
            @classmethod
            def from_config(cls, config):
                return cls(config)
    """
    
    # 模块类型标识
    MODULE_TYPE: str = "base"
    
    def __init__(self, config: Any = None, **kwargs):
        super().__init__()
        self.config = config
        self._is_training = True
        self._device = None
        
    @abstractmethod
    def forward(self, *args, **kwargs) -> ModuleOutput:
        """
        前向传播
        
        所有子类必须实现此方法，并返回 ModuleOutput。
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: Any) -> "BaseModule":
        """
        从配置创建模块实例
        
        支持序列化和反序列化。
        """
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """
        获取模块状态（用于保存）
        
        子类可以覆盖此方法添加额外状态。
        """
        return {
            "module_type": self.MODULE_TYPE,
            "config": self.config,
            "state_dict": self.state_dict(),
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """从状态恢复"""
        self.load_state_dict(state["state_dict"])
    
    def set_training_mode(self, mode: bool) -> None:
        """设置训练/推理模式"""
        self._is_training = mode
        self.train(mode)
    
    @property
    def device(self) -> torch.device:
        """获取当前设备"""
        if self._device is None:
            # 尝试从参数推断设备
            try:
                param = next(self.parameters())
                self._device = param.device
            except StopIteration:
                self._device = torch.device("cpu")
        return self._device
    
    def to(self, device: torch.device) -> "BaseModule":
        """移动到指定设备"""
        self._device = device
        return super().to(device)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.MODULE_TYPE})"


class LatentVector:
    """
    潜向量容器
    
    封装段落级的潜在表示，包含向量和相关元数据。
    """
    
    def __init__(
        self,
        vector: torch.Tensor,
        codebook_indices: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            vector: 潜向量 [batch, d_latent] 或 [d_latent]
            codebook_indices: VQ 码本索引 (如果使用 VQ-VAE)
            metadata: 元数据 (如原始文本位置、时间戳等)
        """
        self.vector = vector
        self.codebook_indices = codebook_indices
        self.metadata = metadata or {}
    
    @property
    def shape(self):
        return self.vector.shape
    
    @property
    def device(self):
        return self.vector.device
    
    def to(self, device: torch.device) -> "LatentVector":
        return LatentVector(
            vector=self.vector.to(device),
            codebook_indices=self.codebook_indices.to(device) if self.codebook_indices is not None else None,
            metadata=self.metadata,
        )
    
    def __repr__(self) -> str:
        return f"LatentVector(shape={self.shape}, device={self.device})"


class MemoryItem:
    """
    记忆项容器
    
    存储在记忆库中的单个条目。
    """
    
    def __init__(
        self,
        vector: torch.Tensor,
        content: Optional[str] = None,
        timestamp: Optional[float] = None,
        importance: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.vector = vector
        self.content = content
        self.timestamp = timestamp
        self.importance = importance
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        preview = self.content[:30] + "..." if self.content and len(self.content) > 30 else self.content
        return f"MemoryItem(content='{preview}', importance={self.importance:.2f})"
