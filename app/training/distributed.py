"""
Distributed Training - 分布式训练支持

功能:
- DDP (DistributedDataParallel) 封装
- 梯度累积
- 混合精度训练 (AMP)
- 分布式工具函数
"""

from typing import Optional, Dict, Any, List
import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


# ============================================================================
# 分布式环境
# ============================================================================

def is_distributed() -> bool:
    """检查是否在分布式环境中"""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """获取当前进程排名"""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """获取总进程数"""
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """是否是主进程"""
    return get_rank() == 0


def setup_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
) -> None:
    """
    初始化分布式环境
    
    通常由 torchrun 自动设置环境变量。
    
    Args:
        backend: 后端 ("nccl" for GPU, "gloo" for CPU)
        init_method: 初始化方法
    """
    if not dist.is_available():
        print("Distributed not available")
        return
    
    if dist.is_initialized():
        return
    
    # 检查环境变量 (torchrun 会设置)
    if "RANK" not in os.environ:
        print("RANK not set, running in single process mode")
        return
    
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # 设置设备
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    # 初始化进程组
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    
    # 同步
    dist.barrier()
    
    if is_main_process():
        print(f"Distributed initialized: rank {rank}/{world_size}")


def cleanup_distributed() -> None:
    """清理分布式环境"""
    if is_distributed():
        dist.destroy_process_group()


# ============================================================================
# DDP 封装
# ============================================================================

def wrap_ddp(
    model: nn.Module,
    device_ids: Optional[List[int]] = None,
    find_unused_parameters: bool = True,
) -> nn.Module:
    """
    用 DDP 封装模型
    
    Args:
        model: 原始模型
        device_ids: GPU 设备 ID
        find_unused_parameters: 是否查找未使用的参数
        
    Returns:
        DDP 封装的模型
    """
    if not is_distributed():
        return model
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if device_ids is None and torch.cuda.is_available():
        device_ids = [local_rank]
    
    return DDP(
        model,
        device_ids=device_ids,
        find_unused_parameters=find_unused_parameters,
    )


def unwrap_model(model: nn.Module) -> nn.Module:
    """获取 DDP 封装的原始模型"""
    if isinstance(model, DDP):
        return model.module
    return model


# ============================================================================
# 分布式数据加载
# ============================================================================

def make_distributed_dataloader(
    dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    collate_fn=None,
) -> DataLoader:
    """
    创建分布式数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批次大小 (每个进程)
        shuffle: 是否打乱
        num_workers: 工作进程数
        pin_memory: 是否固定内存
        collate_fn: 批次整理函数
        
    Returns:
        DataLoader
    """
    sampler = None
    
    if is_distributed():
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
        )
        shuffle = False  # sampler 会处理
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )


# ============================================================================
# 混合精度训练 (AMP)
# ============================================================================

class AMPWrapper:
    """
    混合精度训练封装
    
    使用示例:
        amp = AMPWrapper(enabled=True)
        
        for batch in dataloader:
            with amp.autocast():
                loss = model(batch)
            
            amp.backward(loss, optimizer)
    """
    
    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Args:
            enabled: 是否启用
            dtype: 混合精度类型 (float16 或 bfloat16)
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.dtype = dtype
        
        if self.enabled:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None
    
    def autocast(self):
        """获取 autocast 上下文"""
        if self.enabled:
            return torch.amp.autocast("cuda", dtype=self.dtype)
        else:
            return torch.amp.autocast("cuda", enabled=False)
    
    def backward(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: Optional[float] = None,
        parameters=None,
    ) -> None:
        """
        反向传播 (带梯度缩放)
        
        Args:
            loss: 损失
            optimizer: 优化器
            max_grad_norm: 梯度裁剪
            parameters: 参数 (用于梯度裁剪)
        """
        if self.enabled and self.scaler is not None:
            self.scaler.scale(loss).backward()
            
            if max_grad_norm and parameters:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            if max_grad_norm and parameters:
                torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            
            optimizer.step()
        
        optimizer.zero_grad()


# ============================================================================
# 梯度累积
# ============================================================================

class GradientAccumulator:
    """
    梯度累积封装
    
    使用示例:
        accumulator = GradientAccumulator(steps=4)
        
        for batch in dataloader:
            loss = model(batch)
            
            if accumulator.should_sync():
                # 最后一个累积步骤，同步梯度
                loss.backward()
            else:
                # 中间步骤，不同步
                with model.no_sync():
                    loss.backward()
            
            if accumulator.should_step():
                optimizer.step()
                optimizer.zero_grad()
            
            accumulator.step()
    """
    
    def __init__(self, steps: int = 1):
        """
        Args:
            steps: 累积步数
        """
        self.steps = steps
        self.current_step = 0
    
    def step(self) -> None:
        """累积一步"""
        self.current_step = (self.current_step + 1) % self.steps
    
    def should_step(self) -> bool:
        """是否应该更新参数"""
        return self.current_step == 0
    
    def should_sync(self) -> bool:
        """是否应该同步梯度 (DDP)"""
        return (self.current_step + 1) % self.steps == 0
    
    def reset(self) -> None:
        """重置"""
        self.current_step = 0


# ============================================================================
# 分布式工具
# ============================================================================

def all_reduce(
    tensor: torch.Tensor,
    op: str = "sum",
) -> torch.Tensor:
    """
    全局规约
    
    Args:
        tensor: 输入张量
        op: 操作 ("sum", "mean", "max", "min")
        
    Returns:
        规约后的张量
    """
    if not is_distributed():
        return tensor
    
    tensor = tensor.clone()
    
    if op == "sum":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    elif op == "mean":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= get_world_size()
    elif op == "max":
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    elif op == "min":
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    
    return tensor


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """从源进程广播张量"""
    if not is_distributed():
        return tensor
    
    tensor = tensor.clone()
    dist.broadcast(tensor, src=src)
    return tensor


def sync_print(*args, **kwargs) -> None:
    """只在主进程打印"""
    if is_main_process():
        print(*args, **kwargs)
