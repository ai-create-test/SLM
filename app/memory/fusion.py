"""
Context Fusion Module - 场景与情感注入

核心功能:
1. EmotionEmbedding - 情感状态查找表
2. SceneEmbedding - 场景模式查找表
3. ContextFusion - 可学习门控融合层
4. ContextAwareEmbedding - 整合embedding + 融合

设计原理:
- 使用可学习门控标量 (nn.Parameter) 控制情感/场景注入强度
- Zero-Initialization: 门控初始值为0，模型初期只看文本
- 维度广播: 将2D控制向量扩展至3D以匹配token embedding
- Post-Fusion RMSNorm: 稳定数值分布，防止梯度爆炸
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import CombinedEmbedding


# ============================================================================
# 常量定义
# ============================================================================

# 情感ID映射
EMOTION_IDS: Dict[str, int] = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
    "afraid": 4,
    "surprised": 5,
}

# 场景ID映射
SCENE_IDS: Dict[str, int] = {
    "chat": 0,
    "coding": 1,
    "debate": 2,
    "creative": 3,
    "analysis": 4,
}

# 反向映射
ID_TO_EMOTION = {v: k for k, v in EMOTION_IDS.items()}
ID_TO_SCENE = {v: k for k, v in SCENE_IDS.items()}


# ============================================================================
# RMSNorm (Root Mean Square Layer Normalization)
# ============================================================================

class RMSNorm(nn.Module):
    """
    RMS Layer Normalization
    
    相比LayerNorm，RMSNorm不减去均值，计算更高效。
    被LLaMA等现代LLM广泛采用。
    
    公式: x * rsqrt(mean(x^2) + eps) * weight
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算RMS
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ============================================================================
# 控制向量嵌入层
# ============================================================================

class EmotionEmbedding(nn.Module):
    """
    情感状态嵌入层
    
    将离散的情感ID转换为稠密向量。
    
    使用示例:
        emotion_emb = EmotionEmbedding(d_model=768)
        emotion_id = torch.tensor([3])  # Angry
        vector = emotion_emb(emotion_id)  # [1, 768]
    """
    
    def __init__(
        self,
        d_model: int,
        num_emotions: int = 6,
        init_std: float = 0.02,
    ):
        """
        Args:
            d_model: 嵌入维度 (必须与主模型一致)
            num_emotions: 情感类别数
            init_std: 权重初始化标准差
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_emotions = num_emotions
        
        # 嵌入层
        self.embedding = nn.Embedding(
            num_embeddings=num_emotions,
            embedding_dim=d_model,
        )
        
        # 初始化权重
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)
    
    def forward(self, emotion_id: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            emotion_id: 情感ID [batch] 或 [batch, 1]
            
        Returns:
            情感向量 [batch, d_model]
        """
        # 确保是1D张量
        if emotion_id.dim() > 1:
            emotion_id = emotion_id.squeeze(-1)
        
        return self.embedding(emotion_id)
    
    def get_emotion_name(self, emotion_id: int) -> str:
        """获取情感名称"""
        return ID_TO_EMOTION.get(emotion_id, f"unknown_{emotion_id}")


class SceneEmbedding(nn.Module):
    """
    场景模式嵌入层
    
    将离散的场景ID转换为稠密向量。
    
    使用示例:
        scene_emb = SceneEmbedding(d_model=768)
        scene_id = torch.tensor([1])  # Coding
        vector = scene_emb(scene_id)  # [1, 768]
    """
    
    def __init__(
        self,
        d_model: int,
        num_scenes: int = 5,
        init_std: float = 0.02,
    ):
        """
        Args:
            d_model: 嵌入维度 (必须与主模型一致)
            num_scenes: 场景类别数
            init_std: 权重初始化标准差
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_scenes = num_scenes
        
        # 嵌入层
        self.embedding = nn.Embedding(
            num_embeddings=num_scenes,
            embedding_dim=d_model,
        )
        
        # 初始化权重
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)
    
    def forward(self, scene_id: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            scene_id: 场景ID [batch] 或 [batch, 1]
            
        Returns:
            场景向量 [batch, d_model]
        """
        # 确保是1D张量
        if scene_id.dim() > 1:
            scene_id = scene_id.squeeze(-1)
        
        return self.embedding(scene_id)
    
    def get_scene_name(self, scene_id: int) -> str:
        """获取场景名称"""
        return ID_TO_SCENE.get(scene_id, f"unknown_{scene_id}")


# ============================================================================
# 门控融合层
# ============================================================================

class ContextFusion(nn.Module):
    """
    上下文融合层 - 可学习门控注入机制
    
    核心公式:
        Output = Token + (gate_emotion * Emotion) + (gate_scene * Scene)
    
    关键设计:
    - gate_emotion/gate_scene 是可学习标量参数 (nn.Parameter)
    - Zero-Initialization: 初始值为0或极小值
    - 维度广播: 2D控制向量 → 3D (与token embedding对齐)
    - Post-Fusion RMSNorm: 稳定输出分布
    
    使用示例:
        fusion = ContextFusion(d_model=768)
        output = fusion(
            token_embedding,      # [batch, seq_len, d_model]
            emotion_vector,       # [batch, d_model]
            scene_vector          # [batch, d_model]
        )
    """
    
    def __init__(
        self,
        d_model: int,
        gate_init_value: float = 0.0,
        use_post_norm: bool = True,
    ):
        """
        Args:
            d_model: 嵌入维度
            gate_init_value: 门控初始值 (建议0.0或极小值如0.01)
            use_post_norm: 是否使用融合后归一化
        """
        super().__init__()
        
        self.d_model = d_model
        self.use_post_norm = use_post_norm
        
        # 可学习门控参数
        # 初始化为0，使模型初期只看文本
        self.gate_emotion = nn.Parameter(
            torch.tensor(gate_init_value, dtype=torch.float32)
        )
        self.gate_scene = nn.Parameter(
            torch.tensor(gate_init_value, dtype=torch.float32)
        )
        
        # Post-Fusion RMSNorm
        if use_post_norm:
            self.post_norm = RMSNorm(d_model)
    
    def forward(
        self,
        token_embedding: torch.Tensor,
        emotion_vector: Optional[torch.Tensor] = None,
        scene_vector: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        融合token embedding与上下文控制向量
        
        Args:
            token_embedding: Token嵌入 [batch, seq_len, d_model]
            emotion_vector: 情感向量 [batch, d_model] (可选)
            scene_vector: 场景向量 [batch, d_model] (可选)
            
        Returns:
            融合后的向量 [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = token_embedding.shape
        
        # 从token embedding开始
        output = token_embedding
        
        # 情感注入
        if emotion_vector is not None:
            # 维度广播: [batch, d_model] → [batch, seq_len, d_model]
            emotion_expanded = emotion_vector.unsqueeze(1).expand(
                batch_size, seq_len, d_model
            )
            output = output + self.gate_emotion * emotion_expanded
        
        # 场景注入
        if scene_vector is not None:
            # 维度广播: [batch, d_model] → [batch, seq_len, d_model]
            scene_expanded = scene_vector.unsqueeze(1).expand(
                batch_size, seq_len, d_model
            )
            output = output + self.gate_scene * scene_expanded
        
        # Post-Fusion归一化
        if self.use_post_norm:
            output = self.post_norm(output)
        
        return output
    
    def get_gate_values(self) -> Tuple[float, float]:
        """获取当前门控值"""
        return (
            self.gate_emotion.item(),
            self.gate_scene.item(),
        )


# ============================================================================
# 整合层
# ============================================================================

class ContextAwareEmbedding(nn.Module):
    """
    上下文感知嵌入层
    
    整合:
    - Token Embedding + Position Encoding (RoPE)
    - Emotion Embedding
    - Scene Embedding
    - Context Fusion (可学习门控)
    
    使用示例:
        embedding = ContextAwareEmbedding(
            vocab_size=100000,
            d_model=768,
        )
        
        # 基础使用 (无情感/场景)
        output = embedding(token_ids)
        
        # 带情感使用
        output = embedding(token_ids, emotion_id=3)  # Angry
        
        # 完整使用
        output = embedding(
            token_ids,
            emotion_id=torch.tensor([3]),
            scene_id=torch.tensor([1])
        )
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 2048,
        position_encoding: str = "rope",
        dropout: float = 0.1,
        padding_idx: int = 0,
        num_emotions: int = 6,
        num_scenes: int = 5,
        gate_init_value: float = 0.0,
        use_post_norm: bool = True,
        **kwargs,
    ):
        """
        Args:
            vocab_size: 词表大小
            d_model: 嵌入维度
            max_seq_len: 最大序列长度
            position_encoding: 位置编码类型 ('rope' 或 'sinusoidal')
            dropout: Dropout率
            padding_idx: 填充token ID
            num_emotions: 情感类别数
            num_scenes: 场景类别数
            gate_init_value: 门控初始值
            use_post_norm: 是否使用Post-Fusion归一化
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_emotions = num_emotions
        self.num_scenes = num_scenes
        
        # 基础Token + Position Embedding
        self.token_embedding = CombinedEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            position_encoding=position_encoding,
            dropout=dropout,
            padding_idx=padding_idx,
            **kwargs,
        )
        
        # 控制向量嵌入
        self.emotion_embedding = EmotionEmbedding(
            d_model=d_model,
            num_emotions=num_emotions,
        )
        
        self.scene_embedding = SceneEmbedding(
            d_model=d_model,
            num_scenes=num_scenes,
        )
        
        # 融合层
        self.fusion = ContextFusion(
            d_model=d_model,
            gate_init_value=gate_init_value,
            use_post_norm=use_post_norm,
        )
    
    def forward(
        self,
        token_ids: torch.Tensor,
        emotion_id: Optional[Union[torch.Tensor, int]] = None,
        scene_id: Optional[Union[torch.Tensor, int]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            token_ids: Token ID [batch, seq_len]
            emotion_id: 情感ID (int 或 [batch] tensor)，默认0=Neutral
            scene_id: 场景ID (int 或 [batch] tensor)，默认0=Chat
            position_ids: 位置ID (可选)
            
        Returns:
            融合后的嵌入 [batch, seq_len, d_model]
        """
        batch_size = token_ids.size(0)
        device = token_ids.device
        
        # 1. Token + Position Embedding
        token_emb = self.token_embedding(token_ids, position_ids)
        
        # 2. 处理情感ID
        if emotion_id is None:
            emotion_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        elif isinstance(emotion_id, int):
            emotion_id = torch.full(
                (batch_size,), emotion_id, dtype=torch.long, device=device
            )
        else:
            emotion_id = emotion_id.to(device)
        
        # 3. 处理场景ID
        if scene_id is None:
            scene_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        elif isinstance(scene_id, int):
            scene_id = torch.full(
                (batch_size,), scene_id, dtype=torch.long, device=device
            )
        else:
            scene_id = scene_id.to(device)
        
        # 4. 获取控制向量
        emotion_vec = self.emotion_embedding(emotion_id)  # [batch, d_model]
        scene_vec = self.scene_embedding(scene_id)        # [batch, d_model]
        
        # 5. 融合
        output = self.fusion(token_emb, emotion_vec, scene_vec)
        
        return output
    
    def get_gate_values(self) -> Tuple[float, float]:
        """获取门控值"""
        return self.fusion.get_gate_values()
    
    @property
    def embedding_weight(self) -> torch.Tensor:
        """获取token嵌入权重"""
        return self.token_embedding.embedding_weight


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class FusionConfig:
    """融合层配置"""
    
    # 基础参数
    vocab_size: int = 100000
    d_model: int = 768
    max_seq_len: int = 2048
    position_encoding: str = "rope"
    dropout: float = 0.1
    padding_idx: int = 0
    
    # 控制向量参数
    num_emotions: int = 6
    num_scenes: int = 5
    
    # 融合参数
    gate_init_value: float = 0.0
    use_post_norm: bool = True


def create_context_aware_embedding(
    preset: str = "base",
    **kwargs,
) -> ContextAwareEmbedding:
    """
    创建预配置的ContextAwareEmbedding
    
    Args:
        preset: 预设名 ('small', 'base', 'large')
        **kwargs: 覆盖参数
        
    Returns:
        ContextAwareEmbedding实例
    """
    presets = {
        "small": {
            "vocab_size": 110000,
            "d_model": 256,
            "max_seq_len": 256,
        },
        "base": {
            "vocab_size": 110000,
            "d_model": 768,
            "max_seq_len": 512,
        },
        "large": {
            "vocab_size": 150000,
            "d_model": 1024,
            "max_seq_len": 1024,
        },
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    config = {**presets[preset], **kwargs}
    return ContextAwareEmbedding(**config)
