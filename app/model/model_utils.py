"""
Model Utilities - 模型保存/加载工具

支持 SafeTensors 和 PyTorch 原生格式。
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from collections import OrderedDict

import torch
import torch.nn as nn


# ============================================================================
# SafeTensors 支持 (可选依赖)
# ============================================================================

_SAFETENSORS_AVAILABLE = False
try:
    from safetensors.torch import save_file, load_file
    _SAFETENSORS_AVAILABLE = True
except ImportError:
    pass


def is_safetensors_available() -> bool:
    """检查 SafeTensors 是否可用"""
    return _SAFETENSORS_AVAILABLE


def save_safetensors(
    state_dict: Dict[str, torch.Tensor],
    path: str,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """
    保存为 SafeTensors 格式
    
    Args:
        state_dict: 模型状态字典
        path: 保存路径
        metadata: 可选元数据
    """
    if _SAFETENSORS_AVAILABLE:
        save_file(state_dict, path, metadata=metadata)
    else:
        # 回退到 PyTorch 格式
        torch.save(state_dict, path.replace('.safetensors', '.pt'))


def load_safetensors(
    path: str,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    加载 SafeTensors 文件
    
    Args:
        path: 文件路径
        device: 目标设备
        
    Returns:
        状态字典
    """
    if _SAFETENSORS_AVAILABLE and path.endswith('.safetensors'):
        return load_file(path, device=device)
    else:
        # 回退到 PyTorch 格式
        pt_path = path.replace('.safetensors', '.pt')
        if os.path.exists(pt_path):
            return torch.load(pt_path, map_location=device)
        return torch.load(path, map_location=device)


# ============================================================================
# 配置保存/加载
# ============================================================================

def save_config(config: Any, path: str) -> None:
    """
    保存配置到 JSON
    
    Args:
        config: 配置对象 (需要有 to_dict 方法)
        path: 保存路径
    """
    if hasattr(config, 'to_dict'):
        data = config.to_dict()
    elif hasattr(config, '__dict__'):
        data = config.__dict__
    else:
        data = dict(config)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_config(path: str) -> Dict[str, Any]:
    """
    从 JSON 加载配置
    
    Args:
        path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================================
# 模型目录结构
# ============================================================================

class ModelDirectory:
    """
    模型目录管理
    
    标准目录结构:
        model_dir/
        ├── config.json
        ├── model.safetensors (或 model.pt)
        ├── tokenizer/
        ├── emotion/
        ├── memory/
        └── training_state.json
    """
    
    CONFIG_FILE = "config.json"
    WEIGHTS_FILE = "model.safetensors"
    WEIGHTS_FILE_PT = "model.pt"
    TOKENIZER_DIR = "tokenizer"
    EMOTION_DIR = "emotion"
    MEMORY_DIR = "memory"
    TRAINING_STATE_FILE = "training_state.json"
    
    def __init__(self, path: str):
        self.path = Path(path)
    
    def ensure_exists(self) -> None:
        """确保目录存在"""
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / self.TOKENIZER_DIR).mkdir(exist_ok=True)
        (self.path / self.EMOTION_DIR).mkdir(exist_ok=True)
        (self.path / self.MEMORY_DIR).mkdir(exist_ok=True)
    
    @property
    def config_path(self) -> Path:
        return self.path / self.CONFIG_FILE
    
    @property
    def weights_path(self) -> Path:
        """返回存在的权重文件路径"""
        safetensors_path = self.path / self.WEIGHTS_FILE
        if safetensors_path.exists():
            return safetensors_path
        return self.path / self.WEIGHTS_FILE_PT
    
    @property
    def training_state_path(self) -> Path:
        return self.path / self.TRAINING_STATE_FILE
    
    def exists(self) -> bool:
        """检查是否是有效的模型目录"""
        return self.config_path.exists() and (
            (self.path / self.WEIGHTS_FILE).exists() or
            (self.path / self.WEIGHTS_FILE_PT).exists()
        )


# ============================================================================
# 训练状态
# ============================================================================

def save_training_state(
    path: str,
    global_step: int,
    epoch: int,
    completed_stages: list,
    losses: Dict[str, float],
    **kwargs,
) -> None:
    """
    保存训练状态
    
    Args:
        path: 保存路径
        global_step: 全局步数
        epoch: 当前epoch
        completed_stages: 已完成的训练阶段
        losses: 各阶段最终损失
        **kwargs: 其他信息
    """
    state = {
        "global_step": global_step,
        "epoch": epoch,
        "completed_stages": completed_stages,
        "losses": losses,
        **kwargs,
    }
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)


def load_training_state(path: str) -> Dict[str, Any]:
    """加载训练状态"""
    if not os.path.exists(path):
        return {
            "global_step": 0,
            "epoch": 0,
            "completed_stages": [],
            "losses": {},
        }
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================================
# 安全加载
# ============================================================================

def safe_load_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = False,
) -> tuple:
    """
    安全加载状态字典 (允许部分匹配)
    
    Args:
        model: 目标模型
        state_dict: 状态字典
        strict: 是否严格匹配
        
    Returns:
        (missing_keys, unexpected_keys)
    """
    model_state = model.state_dict()
    
    # 过滤匹配的键
    filtered_state = {}
    unexpected_keys = []
    
    for k, v in state_dict.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered_state[k] = v
            else:
                print(f"Shape mismatch for {k}: {model_state[k].shape} vs {v.shape}")
                unexpected_keys.append(k)
        else:
            unexpected_keys.append(k)
    
    missing_keys = [k for k in model_state if k not in filtered_state]
    
    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(f"Missing: {missing_keys}, Unexpected: {unexpected_keys}")
    
    model.load_state_dict(filtered_state, strict=False)
    
    return missing_keys, unexpected_keys
