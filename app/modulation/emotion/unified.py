"""
Unified Emotion Encoder - 统一情感编码接口

Tier 1: 词典精确匹配
Tier 2: 语义相似度检索 (TODO)
Tier 3: 可训练神经网络 (Future)
"""

from typing import Optional, Union, List
import torch
import torch.nn as nn

from .vad_lexicon import VADLexicon, VADCoordinate, get_default_lexicon
from .vad_encoder import VADEncoder
from .semantic_retriever import SemanticRetriever


# Legacy emotion ID 映射 (向后兼容)
LEGACY_EMOTION_IDS = {
    "neutral": 0, "happy": 1, "sad": 2, "angry": 3,
    "afraid": 4, "surprised": 5, "disgusted": 6, "contempt": 7,
}
LEGACY_ID_TO_EMOTION = {v: k for k, v in LEGACY_EMOTION_IDS.items()}


class SemanticEmotionEncoder(nn.Module):
    """
    统一情感编码器 (Semantic Emotion Encoder)
    
    支持任意自然语言情感描述的编码:
    - 英文: "happy", "excited", "bittersweet"
    - 中文: "开心", "焦虑", "三分讥笑三分薄凉"
    - VAD 张量: torch.tensor([0.8, 0.5, 0.6])
    - Legacy ID: 0-7 (向后兼容)
    
    用法:
        encoder = SemanticEmotionEncoder(d_emotion=128)
        
        vec = encoder("happy")              # 英文
        vec = encoder("开心")               # 中文
        vec = encoder("三分讥笑三分薄凉")   # 复杂表达
        vec = encoder(["happy", "sad"])     # 批量
        vec = encoder(1)                    # Legacy ID
    """
    
    def __init__(
        self,
        d_emotion: int = 128,
        lexicon: Optional[VADLexicon] = None,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_emotion: 输出情感向量维度
            lexicon: 自定义词典 (默认使用全局词典)
            hidden_dim: MLP 隐藏层维度
            dropout: Dropout 率
        """
        super().__init__()
        
        self.d_emotion = d_emotion
        self.lexicon = lexicon or get_default_lexicon()
        
        # Core encoder
        self.vad_encoder = VADEncoder(
            d_emotion=d_emotion,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        
        # Tier 2: Semantic retriever for unknown words
        self._retriever: Optional[SemanticRetriever] = None
        
        # Default VAD for when all else fails
        self._default_vad = VADCoordinate(0.0, 0.0, 0.0, source="default")
    
    def forward(
        self,
        emotion: Union[str, int, torch.Tensor, List[str]],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        编码情感 (自动识别输入类型)
        
        Args:
            emotion: 情感输入，支持:
                - str: 情感词/短语
                - int: legacy emotion_id (0-7)
                - torch.Tensor: VAD 坐标 [3] 或 [batch, 3]
                - List[str]: 批量情感词
            device: 目标设备
            
        Returns:
            情感向量 [batch, d_emotion]
        """
        device = device or next(self.parameters()).device
        
        # 处理不同输入类型
        if isinstance(emotion, int):
            return self._encode_legacy_id(emotion, device)
        
        elif isinstance(emotion, str):
            return self._encode_text(emotion, device)
        
        elif isinstance(emotion, list):
            return self._encode_batch(emotion, device)
        
        elif isinstance(emotion, torch.Tensor):
            if emotion.shape[-1] == 3:
                # VAD tensor
                return self.vad_encoder(emotion.to(device))
            else:
                # Assume legacy emotion_id tensor
                return self._encode_legacy_id(int(emotion.item()), device)
        
        else:
            raise TypeError(f"Unsupported emotion type: {type(emotion)}")
    
    def _encode_text(self, text: str, device: torch.device) -> torch.Tensor:
        """编码单个文本"""
        # Tier 1: 词典精确匹配
        vad = self.lexicon.get(text)
        
        if vad is None:
            # Tier 2: 语义检索
            vad = self._get_semantic_vad(text)
        
        return self.vad_encoder.encode_coordinate(vad, device)
    
    def _get_semantic_vad(self, text: str) -> VADCoordinate:
        """使用语义检索获取 VAD"""
        # 懒加载 retriever
        if self._retriever is None:
            self._retriever = SemanticRetriever(self.lexicon)
        
        result = self._retriever.retrieve(text, k=5)
        
        # 如果检索结果太差 (低相似度)，返回 default
        if result.source == "fallback":
            return self._default_vad
        
        return result
    
    def _encode_batch(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """批量编码"""
        vads = []
        for text in texts:
            vad = self.lexicon.get(text)
            if vad is None:
                vad = self._default_vad
            vads.append(vad)
        
        return self.vad_encoder.encode_batch(vads, device)
    
    def _encode_legacy_id(self, emotion_id: int, device: torch.device) -> torch.Tensor:
        """编码 legacy emotion_id"""
        emotion_name = LEGACY_ID_TO_EMOTION.get(emotion_id, "neutral")
        return self._encode_text(emotion_name, device)
    
    def encode_name(self, name: str, device: Optional[torch.device] = None) -> torch.Tensor:
        """便捷方法: 按名称编码 (API 兼容)"""
        device = device or next(self.parameters()).device
        return self._encode_text(name, device)
    
    def encode_blend(
        self,
        emotion1: str,
        emotion2: str,
        weight: float = 0.5,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """混合两种情感"""
        device = device or next(self.parameters()).device
        
        vad1 = self.lexicon.get(emotion1) or self._default_vad
        vad2 = self.lexicon.get(emotion2) or self._default_vad
        blended = vad1.blend(vad2, weight)
        
        return self.vad_encoder.encode_coordinate(blended, device)
    
    @property
    def emotion_count(self) -> int:
        """词典中的情感词数量"""
        return len(self.lexicon)
    
    @property
    def supported_emotions(self) -> List[str]:
        """返回所有支持的情感词"""
        return self.lexicon.all_words()


# Alias for backward compatibility
UnifiedEmotionEncoder = SemanticEmotionEncoder
