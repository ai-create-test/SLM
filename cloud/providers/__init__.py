"""
Cloud Providers Package

支持的平台:
- RunPod
- Modal
- Lambda Labs
"""

from .base import CloudProvider, JobStatus
from .runpod import RunPodProvider
from .modal_provider import ModalProvider
from .lambda_labs import LambdaLabsProvider


# 平台注册表
PROVIDERS = {
    "runpod": RunPodProvider,
    "modal": ModalProvider,
    "lambda": LambdaLabsProvider,
    "lambda_labs": LambdaLabsProvider,
}


def get_provider(name: str, **kwargs) -> CloudProvider:
    """
    获取云平台实例
    
    Args:
        name: 平台名称 (runpod, modal, lambda)
        **kwargs: 传递给 Provider 构造函数
        
    Returns:
        CloudProvider 实例
    """
    name = name.lower()
    
    if name not in PROVIDERS:
        available = list(PROVIDERS.keys())
        raise ValueError(f"Unknown provider: {name}. Available: {available}")
    
    return PROVIDERS[name](**kwargs)


__all__ = [
    "CloudProvider",
    "JobStatus",
    "RunPodProvider",
    "ModalProvider",
    "LambdaLabsProvider",
    "get_provider",
]
