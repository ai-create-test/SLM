"""
Emotion Encoder - 情感状态编码器

将离散的情感标签编码为连续向量，用于调制推理过程。
"""

from typing import Dict, Optional, Union
import torch
import torch.nn as nn


# 情感 ID 映射
EMOTION_IDS: Dict[str, int] = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
    "afraid": 4,
    "surprised": 5,
    "disgusted": 6,
    "contempt": 7,
}

# 反向映射
ID_TO_EMOTION = {v: k for k, v in EMOTION_IDS.items()}


class EmotionEncoder(nn.Module):
    """
    情感状态编码器
    
    将离散的情感 ID 编码为稠密向量。
    
    使用示例:
        encoder = EmotionEncoder(d_emotion=128)
        
        # 从 ID 编码
        emotion_vec = encoder(torch.tensor([3]))  # Angry
        
        # 从名称编码
        emotion_vec = encoder.encode_name("angry")
    """
    
    def __init__(
        self,
        d_emotion: int = 128,
        num_emotions: int = 8,
        init_std: float = 0.02,
    ):
        """
        Args:
            d_emotion: 情感向量维度
            num_emotions: 情感类别数
            init_std: 权重初始化标准差
        """
        super().__init__()
        
        self.d_emotion = d_emotion
        self.num_emotions = num_emotions
        
        # 情感嵌入层
        self.embedding = nn.Embedding(
            num_embeddings=num_emotions,
            embedding_dim=d_emotion,
        )
        
        # 初始化
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)
    
    def forward(
        self,
        emotion_id: Union[torch.Tensor, int],
    ) -> torch.Tensor:
        """
        编码情感 ID
        
        Args:
            emotion_id: 情感 ID [batch] 或 单个 int
            
        Returns:
            情感向量 [batch, d_emotion]
        """
        if isinstance(emotion_id, int):
            emotion_id = torch.tensor([emotion_id], device=self.embedding.weight.device)
        
        if emotion_id.dim() > 1:
            emotion_id = emotion_id.squeeze(-1)
        
        return self.embedding(emotion_id)
    
    def encode_name(
        self,
        emotion_name: str,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        从情感名称编码
        
        Args:
            emotion_name: 情感名称 (如 "happy", "angry")
            device: 目标设备
        """
        emotion_id = EMOTION_IDS.get(emotion_name.lower(), 0)
        device = device or self.embedding.weight.device
        return self.forward(torch.tensor([emotion_id], device=device))
    
    def get_emotion_name(self, emotion_id: int) -> str:
        """获取情感名称"""
        return ID_TO_EMOTION.get(emotion_id, f"unknown_{emotion_id}")
    
    @property
    def emotion_names(self) -> list:
        """返回所有情感名称"""
        return list(EMOTION_IDS.keys())


class LearnedEmotionEncoder(nn.Module):
    """
    可学习的情感编码器
    
    不依赖预定义的情感类别，而是从情感描述文本中学习编码。
    
    预留实现：可以使用预训练的文本编码器来编码情感描述。
    """
    
    def __init__(
        self,
        d_emotion: int = 128,
        d_text: int = 768,
    ):
        """
        Args:
            d_emotion: 输出情感向量维度
            d_text: 文本编码器的输出维度
        """
        super().__init__()
        
        self.d_emotion = d_emotion
        
        # 从文本嵌入投影到情感空间
        self.projection = nn.Sequential(
            nn.Linear(d_text, d_emotion),
            nn.LayerNorm(d_emotion),
            nn.Tanh(),  # 限制范围
        )
    
    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        从文本嵌入编码情感
        
        Args:
            text_embedding: 情感描述的文本嵌入 [batch, d_text]
            
        Returns:
            情感向量 [batch, d_emotion]
        """
        return self.projection(text_embedding)
