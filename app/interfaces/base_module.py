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


@dataclass
class HierarchicalLatent:
    """
    分层潜向量容器 (AMHVQ+)
    
    支持多层级语义表示：
    - global_: 全局语义 [batch, 1, d_latent]
    - chunks: 语义块 [batch, num_chunks, d_latent]
    - detail: 细节残差 [batch, num_detail, d_latent//2] (可选)
    
    使用示例:
        latent = HierarchicalLatent(
            global_=torch.randn(1, 1, 512),
            chunks=torch.randn(1, 4, 512),
        )
        
        # 获取总 token 数
        print(latent.num_tokens)  # 5
        
        # 扁平化
        flat = latent.flatten()  # [batch, 5, 512]
        
        # 转为单向量 (兼容旧接口)
        vec = latent.to_single_vector()  # [batch, 512]
    """
    global_: torch.Tensor  # [batch, 1, d_latent]
    chunks: torch.Tensor   # [batch, num_chunks, d_latent]
    detail: Optional[torch.Tensor] = None  # [batch, num_detail, d_latent//2]
    indices: Optional[torch.Tensor] = None  # VQ 码本索引
    chunk_mask: Optional[torch.Tensor] = None  # 有效 chunk 掩码 [batch, num_chunks]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_tokens(self) -> int:
        """总 latent token 数"""
        count = self.global_.shape[1] + self.chunks.shape[1]
        if self.detail is not None:
            count += self.detail.shape[1]
        return count
    
    @property
    def num_chunks(self) -> int:
        """语义块数量"""
        return self.chunks.shape[1]
    
    @property
    def d_latent(self) -> int:
        """潜空间维度"""
        return self.global_.shape[-1]
    
    @property
    def batch_size(self) -> int:
        return self.global_.shape[0]
    
    @property
    def device(self) -> torch.device:
        return self.global_.device
    
    def flatten(self) -> torch.Tensor:
        """
        扁平化所有层级为单一序列
        
        Returns:
            [batch, num_tokens, d_latent]
        """
        components = [self.global_, self.chunks]
        if self.detail is not None:
            # detail 维度可能不同，需要投影或 padding
            if self.detail.shape[-1] != self.d_latent:
                # 简单 padding 到相同维度
                padded = torch.zeros(
                    self.batch_size, self.detail.shape[1], self.d_latent,
                    device=self.device, dtype=self.detail.dtype
                )
                padded[..., :self.detail.shape[-1]] = self.detail
                components.append(padded)
            else:
                components.append(self.detail)
        return torch.cat(components, dim=1)
    
    def to_single_vector(self) -> torch.Tensor:
        """
        转换为单向量 (兼容旧接口)
        
        使用 global 向量的均值，确保向后兼容。
        
        Returns:
            [batch, d_latent]
        """
        return self.global_.squeeze(1)
    
    def to_legacy(self) -> "LatentVector":
        """转换为旧版 LatentVector 格式"""
        return LatentVector(
            vector=self.to_single_vector(),
            codebook_indices=self.indices,
            metadata={**self.metadata, "_hierarchical": True}
        )
    
    def to(self, device: torch.device) -> "HierarchicalLatent":
        """移动到指定设备"""
        return HierarchicalLatent(
            global_=self.global_.to(device),
            chunks=self.chunks.to(device),
            detail=self.detail.to(device) if self.detail is not None else None,
            indices=self.indices.to(device) if self.indices is not None else None,
            chunk_mask=self.chunk_mask.to(device) if self.chunk_mask is not None else None,
            metadata=self.metadata,
        )
    
    @classmethod
    def from_single_vector(
        cls,
        vector: torch.Tensor,
        num_chunks: int = 1,
    ) -> "HierarchicalLatent":
        """
        从单向量创建 (兼容旧数据)
        
        Args:
            vector: [batch, d_latent]
            num_chunks: 复制为多少个 chunk
        """
        batch_size, d_latent = vector.shape
        global_ = vector.unsqueeze(1)  # [batch, 1, d_latent]
        chunks = vector.unsqueeze(1).expand(-1, num_chunks, -1).clone()
        return cls(global_=global_, chunks=chunks)
    
    @classmethod
    def from_flat(
        cls,
        flat_tensor: torch.Tensor,
        num_chunks: int,
        has_detail: bool = False,
        detail_dim: Optional[int] = None,
    ) -> "HierarchicalLatent":
        """
        从扁平化张量重建 HierarchicalLatent
        
        Args:
            flat_tensor: [batch, num_tokens, d_latent]
            num_chunks: chunk 数量
            has_detail: 是否包含 detail
            detail_dim: detail 的维度 (如果不同于 d_latent)
            
        Returns:
            HierarchicalLatent
        """
        batch_size, num_tokens, d_latent = flat_tensor.shape
        
        # 分离 global (第一个 token)
        global_ = flat_tensor[:, :1, :]  # [batch, 1, d_latent]
        
        # 分离 chunks
        chunks_end = 1 + num_chunks
        chunks = flat_tensor[:, 1:chunks_end, :]  # [batch, num_chunks, d_latent]
        
        # 分离 detail (如果有)
        detail = None
        if has_detail and num_tokens > chunks_end:
            detail = flat_tensor[:, chunks_end:, :]
            if detail_dim is not None and detail_dim != d_latent:
                detail = detail[..., :detail_dim]
        
        return cls(global_=global_, chunks=chunks, detail=detail)
    
    def __repr__(self) -> str:
        detail_info = f", detail={self.detail.shape}" if self.detail is not None else ""
        return (
            f"HierarchicalLatent(global={self.global_.shape}, "
            f"chunks={self.chunks.shape}{detail_info}, "
            f"num_tokens={self.num_tokens})"
        )


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
