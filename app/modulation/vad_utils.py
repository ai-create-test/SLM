"""
VAD Utilities - Valence-Arousal-Dominance 情感空间工具

VAD 模型是心理学中描述情感状态的三维连续空间:
- Valence (效价): -1.0 (不愉快) → +1.0 (愉快)
- Arousal (唤醒度): -1.0 (平静/困倦) → +1.0 (激动/警觉)
- Dominance (支配度): -1.0 (被控制/顺从) → +1.0 (掌控/主导)

参考文献:
- Russell, J. A. (1980). A circumplex model of affect.
- Mehrabian, A. (1996). Pleasure-arousal-dominance: A general framework...
"""

from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import torch


@dataclass
class VADCoordinate:
    """VAD 坐标点"""
    valence: float      # [-1.0, 1.0]
    arousal: float      # [-1.0, 1.0]
    dominance: float    # [-1.0, 1.0]
    
    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """转换为 PyTorch 张量 [3]"""
        t = torch.tensor([self.valence, self.arousal, self.dominance], dtype=torch.float32)
        if device is not None:
            t = t.to(device)
        return t
    
    def __add__(self, other: "VADCoordinate") -> "VADCoordinate":
        """支持 VAD 坐标加法 (用于混合情感)"""
        return VADCoordinate(
            valence=(self.valence + other.valence) / 2,
            arousal=(self.arousal + other.arousal) / 2,
            dominance=(self.dominance + other.dominance) / 2,
        )
    
    def blend(self, other: "VADCoordinate", weight: float = 0.5) -> "VADCoordinate":
        """加权混合两个 VAD 坐标"""
        w = max(0.0, min(1.0, weight))  # Clamp to [0, 1]
        return VADCoordinate(
            valence=self.valence * (1 - w) + other.valence * w,
            arousal=self.arousal * (1 - w) + other.arousal * w,
            dominance=self.dominance * (1 - w) + other.dominance * w,
        )


# ============================================================================
# 标准 VAD 预设 (基于心理学文献)
# ============================================================================

VAD_PRESETS: Dict[str, VADCoordinate] = {
    # 基本情绪 (Ekman's 6 + 扩展)
    "neutral":      VADCoordinate(0.0, 0.0, 0.0),
    "happy":        VADCoordinate(0.8, 0.5, 0.6),
    "sad":          VADCoordinate(-0.7, -0.3, -0.5),
    "angry":        VADCoordinate(-0.6, 0.8, 0.7),
    "afraid":       VADCoordinate(-0.7, 0.7, -0.6),
    "surprised":    VADCoordinate(0.4, 0.8, 0.2),
    "disgusted":    VADCoordinate(-0.6, 0.3, 0.4),
    "contempt":     VADCoordinate(-0.4, 0.1, 0.6),
    
    # 复杂/混合情绪
    "anxious":      VADCoordinate(-0.5, 0.7, -0.4),
    "calm":         VADCoordinate(0.3, -0.7, 0.3),
    "excited":      VADCoordinate(0.7, 0.9, 0.5),
    "bored":        VADCoordinate(-0.3, -0.6, -0.2),
    "content":      VADCoordinate(0.6, -0.2, 0.4),
    "frustrated":   VADCoordinate(-0.5, 0.6, -0.3),
    "hopeful":      VADCoordinate(0.5, 0.3, 0.2),
    "melancholy":   VADCoordinate(-0.4, -0.4, -0.3),
    "proud":        VADCoordinate(0.7, 0.4, 0.8),
    "ashamed":      VADCoordinate(-0.6, 0.2, -0.7),
    "curious":      VADCoordinate(0.4, 0.5, 0.3),
    "nostalgic":    VADCoordinate(0.2, -0.3, -0.2),
    
    # 文学/细腻情感
    "bittersweet":  VADCoordinate(0.1, -0.1, -0.1),   # 苦乐参半
    "wistful":      VADCoordinate(0.0, -0.2, -0.3),   # 惆怅
    "serene":       VADCoordinate(0.6, -0.8, 0.5),    # 宁静
    "pensive":      VADCoordinate(-0.1, -0.3, 0.1),   # 沉思
    "yearning":     VADCoordinate(0.3, 0.2, -0.4),    # 渴望
}

# 反向映射: 用于从 legacy emotion_id 转换
EMOTION_ID_TO_VAD: Dict[int, VADCoordinate] = {
    0: VAD_PRESETS["neutral"],
    1: VAD_PRESETS["happy"],
    2: VAD_PRESETS["sad"],
    3: VAD_PRESETS["angry"],
    4: VAD_PRESETS["afraid"],
    5: VAD_PRESETS["surprised"],
    6: VAD_PRESETS["disgusted"],
    7: VAD_PRESETS["contempt"],
}


def get_vad(emotion: str) -> VADCoordinate:
    """
    获取情感的 VAD 坐标
    
    Args:
        emotion: 情感名称 (如 "happy", "bittersweet")
        
    Returns:
        VADCoordinate 对象
        
    Raises:
        KeyError: 如果情感名称未定义
    """
    key = emotion.lower().strip()
    if key not in VAD_PRESETS:
        raise KeyError(f"Unknown emotion: '{emotion}'. Available: {list(VAD_PRESETS.keys())}")
    return VAD_PRESETS[key]


def emotion_id_to_vad(emotion_id: int) -> VADCoordinate:
    """
    将旧版 emotion_id (0-7) 转换为 VAD 坐标
    
    用于向后兼容旧代码
    """
    return EMOTION_ID_TO_VAD.get(emotion_id, VAD_PRESETS["neutral"])


def vad_to_tensor(
    valence: float,
    arousal: float,
    dominance: float,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    直接从数值创建 VAD 张量
    
    Args:
        valence, arousal, dominance: 各维度值 [-1.0, 1.0]
        device: 目标设备
        
    Returns:
        张量 [3]
    """
    t = torch.tensor([valence, arousal, dominance], dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t


def batch_vad_to_tensor(
    emotions: List[str],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    批量转换情感名称为 VAD 张量
    
    Args:
        emotions: 情感名称列表
        device: 目标设备
        
    Returns:
        张量 [batch, 3]
    """
    vads = [get_vad(e).to_tensor() for e in emotions]
    batch = torch.stack(vads)
    if device is not None:
        batch = batch.to(device)
    return batch


def blend_emotions(
    emotion1: str,
    emotion2: str,
    weight: float = 0.5,
) -> VADCoordinate:
    """
    混合两种情感
    
    Args:
        emotion1: 第一种情感
        emotion2: 第二种情感
        weight: emotion2 的权重 [0.0, 1.0]
        
    Returns:
        混合后的 VADCoordinate
        
    Example:
        >>> blend_emotions("happy", "sad", 0.3)  # 70% happy + 30% sad
    """
    vad1 = get_vad(emotion1)
    vad2 = get_vad(emotion2)
    return vad1.blend(vad2, weight)
