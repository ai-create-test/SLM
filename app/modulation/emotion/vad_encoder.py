"""
VAD Encoder - VAD 坐标到情感向量的投影

将 3 维 VAD 空间投影到 d_emotion 维的稠密向量。
"""

from typing import Optional, List
import torch
import torch.nn as nn

from .vad_lexicon import VADCoordinate


class VADEncoder(nn.Module):
    """
    VAD (Valence-Arousal-Dominance) 情感编码器
    
    使用 MLP 将 3 维 VAD 坐标投影到 d_emotion 维稠密向量。
    
    用法:
        encoder = VADEncoder(d_emotion=128)
        
        # 从 VAD 张量编码
        vad = torch.tensor([[0.8, 0.5, 0.6]])
        vec = encoder(vad)  # [1, 128]
        
        # 从 VADCoordinate 编码
        vad_coord = VADCoordinate(0.8, 0.5, 0.6)
        vec = encoder.encode_coordinate(vad_coord)
    """
    
    def __init__(
        self,
        d_emotion: int = 128,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_emotion: 输出情感向量维度
            hidden_dim: MLP 隐藏层维度，默认为 max(32, d_emotion // 2)
            dropout: Dropout 率
        """
        super().__init__()
        
        self.d_emotion = d_emotion
        self.hidden_dim = hidden_dim or max(32, d_emotion // 2)
        
        # VAD (3维) → d_emotion 的 MLP 投影
        self.projection = nn.Sequential(
            nn.Linear(3, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, d_emotion),
            nn.LayerNorm(d_emotion),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, vad: torch.Tensor) -> torch.Tensor:
        """
        编码 VAD 向量
        
        Args:
            vad: VAD 坐标张量 [batch, 3] 或 [3]
                 每个维度范围 [-1.0, 1.0]
            
        Returns:
            情感向量 [batch, d_emotion]
        """
        if vad.dim() == 1:
            vad = vad.unsqueeze(0)  # [3] → [1, 3]
        
        return self.projection(vad)
    
    def encode_coordinate(
        self,
        vad: VADCoordinate,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        从 VADCoordinate 编码
        
        Args:
            vad: VADCoordinate 对象
            device: 目标设备
            
        Returns:
            情感向量 [1, d_emotion]
        """
        device = device or next(self.parameters()).device
        vad_tensor = vad.to_tensor(device).unsqueeze(0)
        return self.forward(vad_tensor)
    
    def encode_batch(
        self,
        vads: List[VADCoordinate],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        批量编码 VADCoordinate 列表
        
        Args:
            vads: VADCoordinate 对象列表
            device: 目标设备
            
        Returns:
            情感向量 [batch, d_emotion]
        """
        device = device or next(self.parameters()).device
        tensors = [v.to_tensor() for v in vads]
        vad_tensor = torch.stack(tensors).to(device)
        return self.forward(vad_tensor)
