"""
Latent Utils - 潜空间兼容性工具

提供不同潜空间表示之间的转换工具。
"""

from typing import Union, Any
import torch

from .base_module import LatentVector, HierarchicalLatent
from .unified_latent import (
    UnifiedLatent,
    detect_latent_type,
    to_unified,
    to_hierarchical,
    to_legacy,
)

# Re-export from unified_latent for convenience
__all__ = [
    "detect_latent_type",
    "to_unified",
    "to_hierarchical",
    "to_legacy",
    "ensure_legacy",
    "ensure_hierarchical",
    "ensure_unified",
    "latent_to_tensor",
    "tensor_to_latent",
]


def ensure_legacy(
    latent: Union[UnifiedLatent, HierarchicalLatent, LatentVector, torch.Tensor],
) -> LatentVector:
    """
    确保输入为 LatentVector 格式
    
    用于与旧接口兼容。
    """
    return to_legacy(latent)


def ensure_hierarchical(
    latent: Union[UnifiedLatent, HierarchicalLatent, LatentVector, torch.Tensor],
    num_chunks: int = 4,
) -> HierarchicalLatent:
    """
    确保输入为 HierarchicalLatent 格式
    
    Args:
        latent: 任意潜空间类型
        num_chunks: 如果从单向量创建，使用的 chunk 数量
    """
    latent_type = detect_latent_type(latent)
    
    if latent_type == "hierarchical":
        return latent
    elif latent_type == "unified":
        return latent.semantic
    elif latent_type == "legacy":
        return HierarchicalLatent.from_single_vector(latent.vector, num_chunks)
    elif latent_type == "tensor":
        if latent.dim() == 2:
            return HierarchicalLatent.from_single_vector(latent, num_chunks)
        elif latent.dim() == 3:
            # 假设是 [batch, num_tokens, d_latent]，尝试解析
            batch_size, num_tokens, d_latent = latent.shape
            if num_tokens == 1:
                return HierarchicalLatent.from_single_vector(latent.squeeze(1), num_chunks)
            else:
                # 假设第一个是 global，其余是 chunks
                return HierarchicalLatent.from_flat(latent, num_chunks=num_tokens - 1)
        else:
            raise ValueError(f"Cannot convert tensor with {latent.dim()} dims")
    else:
        raise ValueError(f"Unknown latent type: {type(latent)}")


def ensure_unified(
    latent: Union[UnifiedLatent, HierarchicalLatent, LatentVector, torch.Tensor],
    scene: str = "chat",
) -> UnifiedLatent:
    """
    确保输入为 UnifiedLatent 格式
    """
    return to_unified(latent, scene)


def latent_to_tensor(
    latent: Union[UnifiedLatent, HierarchicalLatent, LatentVector, torch.Tensor],
    mode: str = "single",
) -> torch.Tensor:
    """
    将任意潜空间转换为张量
    
    Args:
        latent: 潜空间
        mode: "single" (单向量), "flat" (扁平化), "global" (仅全局)
        
    Returns:
        张量
    """
    latent_type = detect_latent_type(latent)
    
    if latent_type == "tensor":
        return latent
    
    if latent_type == "legacy":
        return latent.vector
    
    if latent_type == "unified":
        latent = latent.semantic
        latent_type = "hierarchical"
    
    if latent_type == "hierarchical":
        if mode == "single":
            return latent.to_single_vector()
        elif mode == "flat":
            return latent.flatten()
        elif mode == "global":
            return latent.global_.squeeze(1)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    raise ValueError(f"Cannot convert {type(latent)} to tensor")


def tensor_to_latent(
    tensor: torch.Tensor,
    target_type: str = "legacy",
    num_chunks: int = 4,
    scene: str = "chat",
) -> Union[LatentVector, HierarchicalLatent, UnifiedLatent]:
    """
    将张量转换为指定的潜空间类型
    
    Args:
        tensor: 输入张量 [batch, d_latent] 或 [batch, num_tokens, d_latent]
        target_type: "legacy", "hierarchical", "unified"
        num_chunks: chunk 数量 (用于 hierarchical)
        scene: 场景 (用于 unified)
        
    Returns:
        指定类型的潜空间
    """
    if target_type == "legacy":
        if tensor.dim() == 3:
            tensor = tensor.mean(dim=1)
        return LatentVector(vector=tensor)
    
    elif target_type == "hierarchical":
        if tensor.dim() == 2:
            return HierarchicalLatent.from_single_vector(tensor, num_chunks)
        else:
            return HierarchicalLatent.from_flat(tensor, num_chunks=tensor.shape[1] - 1)
    
    elif target_type == "unified":
        hierarchical = tensor_to_latent(tensor, "hierarchical", num_chunks)
        return UnifiedLatent.from_hierarchical(hierarchical, scene)
    
    else:
        raise ValueError(f"Unknown target type: {target_type}")
