"""
Trainable Emotion Encoder - Tier 3 可训练神经网络

将自然语言情感描述直接映射到 VAD 坐标的神经网络编码器。

支持两种模式:
1. 简化模式 (无 transformers): 使用字符级 CNN + MLP
2. 完整模式 (有 transformers): 使用预训练 BERT/多语言模型

训练流程:
1. 使用 annotation tool 收集 <text, VAD> 标注数据
2. 加载数据训练 text → VAD 网络
3. 将训练好的模型集成到 SemanticEmotionEncoder
"""

from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vad_lexicon import VADCoordinate
from .vad_encoder import VADEncoder


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 20
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_length: int = 64


# ============================================================================
# 简化版: 字符级 CNN 编码器 (无需 transformers)
# ============================================================================

class CharacterCNN(nn.Module):
    """
    字符级 CNN 文本编码器
    
    将文本转换为固定维度向量，支持任意语言。
    """
    
    # 支持的字符集 (ASCII + 常用中文)
    CHARSET = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        " .,!?;:'-\""
        "的一是不了人我在有他这为之大来以个中上们到"  # 常用中文
    )
    
    def __init__(
        self,
        d_model: int = 256,
        max_length: int = 64,
        num_filters: int = 128,
        kernel_sizes: Tuple[int, ...] = (2, 3, 4, 5),
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_length = max_length
        
        # 字符嵌入
        vocab_size = len(self.CHARSET) + 2  # +2 for PAD and UNK
        self.char_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # 多尺度卷积
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        # 输出投影
        total_filters = num_filters * len(kernel_sizes)
        self.output_proj = nn.Sequential(
            nn.Linear(total_filters, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        
        # 字符到索引映射
        self._char_to_idx = {c: i+2 for i, c in enumerate(self.CHARSET)}
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """将文本转换为字符索引"""
        indices = []
        for c in text[:self.max_length]:
            idx = self._char_to_idx.get(c, 1)  # 1 = UNK
            indices.append(idx)
        
        # 填充到 max_length
        while len(indices) < self.max_length:
            indices.append(0)  # 0 = PAD
        
        return torch.tensor(indices, dtype=torch.long)
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        编码文本列表
        
        Args:
            texts: 文本列表
            
        Returns:
            文本嵌入 [batch, d_model]
        """
        device = next(self.parameters()).device
        
        # Tokenize
        tokens = torch.stack([self._tokenize(t) for t in texts]).to(device)
        
        # Embed: [batch, seq_len, d_model]
        x = self.char_embedding(tokens)
        
        # Conv expects [batch, channels, seq_len]
        x = x.transpose(1, 2)
        
        # Multi-scale convolution + max pooling
        conv_outputs = []
        for conv in self.convs:
            h = conv(x)
            h = F.relu(h)
            h = F.max_pool1d(h, h.size(2)).squeeze(2)
            conv_outputs.append(h)
        
        # Concatenate
        x = torch.cat(conv_outputs, dim=1)
        
        # Project
        return self.output_proj(x)


# ============================================================================
# 完整版: Transformer 编码器 (需要 transformers 库)
# ============================================================================

class TransformerTextEncoder(nn.Module):
    """
    基于预训练 Transformer 的文本编码器
    
    使用 HuggingFace transformers 库加载预训练模型。
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        d_model: int = 256,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        try:
            from transformers import AutoModel, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name)
            self.encoder_dim = self.encoder.config.hidden_size
            
            if freeze_encoder:
                for p in self.encoder.parameters():
                    p.requires_grad = False
            
            # 投影到目标维度
            self.output_proj = nn.Linear(self.encoder_dim, d_model)
            
            self._available = True
            
        except ImportError:
            self._available = False
            self.tokenizer = None
            self.encoder = None
            self.output_proj = None
    
    @property
    def is_available(self) -> bool:
        return self._available
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """编码文本列表"""
        if not self._available:
            raise RuntimeError("transformers library not available")
        
        device = next(self.parameters()).device
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        
        # Encode
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.encoder(**inputs)
        
        # Use [CLS] token
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Project
        return self.output_proj(cls_embedding)


# ============================================================================
# Trainable Emotion Encoder (主类)
# ============================================================================

class TrainableEmotionEncoder(nn.Module):
    """
    可训练的情感编码器 (Tier 3)
    
    将任意自然语言文本直接映射到 VAD 坐标。
    
    用法:
        # 创建编码器
        encoder = TrainableEmotionEncoder(d_emotion=128)
        
        # 训练
        trainer = EmotionEncoderTrainer(encoder)
        trainer.train(dataset)
        
        # 推理
        vad = encoder.predict_vad("非常开心")  # Returns VADCoordinate
        vec = encoder("非常开心")  # Returns [1, d_emotion]
    """
    
    def __init__(
        self,
        d_emotion: int = 128,
        d_hidden: int = 256,
        use_transformer: bool = True,
        transformer_model: str = "bert-base-multilingual-cased",
        freeze_transformer: bool = True,
    ):
        """
        Args:
            d_emotion: 输出情感向量维度
            d_hidden: 隐藏层维度
            use_transformer: 是否尝试使用 Transformer
            transformer_model: 预训练模型名称
            freeze_transformer: 是否冻结预训练参数
        """
        super().__init__()
        
        self.d_emotion = d_emotion
        self.d_hidden = d_hidden
        
        # 尝试使用 Transformer，否则回退到 CharCNN
        self._use_transformer = False
        if use_transformer:
            self.text_encoder = TransformerTextEncoder(
                model_name=transformer_model,
                d_model=d_hidden,
                freeze_encoder=freeze_transformer,
            )
            if self.text_encoder.is_available:
                self._use_transformer = True
        
        if not self._use_transformer:
            self.text_encoder = CharacterCNN(d_model=d_hidden)
        
        # Text embedding → VAD
        self.to_vad = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden // 2, 3),
            nn.Tanh(),  # VAD 范围 [-1, 1]
        )
        
        # VAD → d_emotion
        self.vad_encoder = VADEncoder(d_emotion=d_emotion, hidden_dim=d_hidden // 2)
    
    def forward(
        self,
        texts: List[str],
        return_vad: bool = False,
    ) -> torch.Tensor:
        """
        编码文本列表
        
        Args:
            texts: 文本列表
            return_vad: 是否返回 VAD 而不是最终向量
            
        Returns:
            情感向量 [batch, d_emotion] 或 VAD [batch, 3]
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Text → hidden
        h = self.text_encoder(texts)
        
        # Hidden → VAD
        vad = self.to_vad(h)
        
        if return_vad:
            return vad
        
        # VAD → emotion vector
        return self.vad_encoder(vad)
    
    def predict_vad(self, text: str) -> VADCoordinate:
        """
        预测单个文本的 VAD
        
        Args:
            text: 输入文本
            
        Returns:
            VADCoordinate
        """
        with torch.no_grad():
            vad = self.forward([text], return_vad=True)[0]
        
        return VADCoordinate(
            valence=float(vad[0]),
            arousal=float(vad[1]),
            dominance=float(vad[2]),
            source="neural",
        )
    
    def predict_batch(self, texts: List[str]) -> List[VADCoordinate]:
        """批量预测 VAD"""
        with torch.no_grad():
            vads = self.forward(texts, return_vad=True)
        
        results = []
        for vad in vads:
            results.append(VADCoordinate(
                valence=float(vad[0]),
                arousal=float(vad[1]),
                dominance=float(vad[2]),
                source="neural",
            ))
        return results
    
    @property
    def encoder_type(self) -> str:
        """返回使用的编码器类型"""
        return "transformer" if self._use_transformer else "char_cnn"
    
    def save(self, path: str) -> None:
        """保存模型"""
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "d_emotion": self.d_emotion,
                "d_hidden": self.d_hidden,
                "encoder_type": self.encoder_type,
            }
        }, path)
    
    @classmethod
    def load(cls, path: str) -> "TrainableEmotionEncoder":
        """加载模型"""
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["config"]
        
        model = cls(
            d_emotion=config["d_emotion"],
            d_hidden=config["d_hidden"],
            use_transformer=(config["encoder_type"] == "transformer"),
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model


# ============================================================================
# 训练器
# ============================================================================

class EmotionEncoderTrainer:
    """
    情感编码器训练器
    
    用法:
        encoder = TrainableEmotionEncoder(d_emotion=128)
        trainer = EmotionEncoderTrainer(encoder)
        
        # 从 JSON 加载训练数据
        trainer.load_data("data/vad/custom_emotions.json")
        
        # 训练
        trainer.train(epochs=20)
        
        # 保存
        encoder.save("models/emotion_encoder.pt")
    """
    
    def __init__(
        self,
        encoder: TrainableEmotionEncoder,
        config: Optional[TrainingConfig] = None,
    ):
        self.encoder = encoder
        self.config = config or TrainingConfig()
        
        self._texts: List[str] = []
        self._vads: List[Tuple[float, float, float]] = []
    
    def load_data(self, path: str) -> int:
        """
        从 JSON 文件加载训练数据
        
        Args:
            path: JSON 文件路径
            
        Returns:
            加载的样本数
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        count = 0
        for text, values in data.items():
            if text.startswith("_"):
                continue
            
            v = values.get("v", values.get("valence", 0.0))
            a = values.get("a", values.get("arousal", 0.0))
            d = values.get("d", values.get("dominance", 0.0))
            
            self._texts.append(text)
            self._vads.append((v, a, d))
            count += 1
        
        print(f"Loaded {count} training samples from {path}")
        return count
    
    def add_sample(self, text: str, vad: VADCoordinate) -> None:
        """添加单个训练样本"""
        self._texts.append(text)
        self._vads.append((vad.valence, vad.arousal, vad.dominance))
    
    def train(
        self,
        epochs: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        训练模型
        
        Args:
            epochs: 训练轮数
            verbose: 是否打印进度
            
        Returns:
            训练历史 {"loss": [...]}
        """
        epochs = epochs or self.config.epochs
        
        if len(self._texts) == 0:
            raise ValueError("No training data. Call load_data() first.")
        
        # 准备数据
        device = next(self.encoder.parameters()).device
        target_vads = torch.tensor(self._vads, dtype=torch.float32, device=device)
        
        # 优化器
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.encoder.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # 训练
        self.encoder.train()
        history = {"loss": []}
        
        batch_size = min(self.config.batch_size, len(self._texts))
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # 简单的全批次训练 (数据量小)
            for i in range(0, len(self._texts), batch_size):
                batch_texts = self._texts[i:i+batch_size]
                batch_vads = target_vads[i:i+batch_size]
                
                optimizer.zero_grad()
                
                # 前向
                pred_vads = self.encoder(batch_texts, return_vad=True)
                
                # 损失
                loss = F.mse_loss(pred_vads, batch_vads)
                
                # 反向
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            history["loss"].append(avg_loss)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
        self.encoder.eval()
        
        if verbose:
            print(f"\nTraining complete. Final loss: {history['loss'][-1]:.4f}")
        
        return history
    
    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        self.encoder.eval()
        device = next(self.encoder.parameters()).device
        
        target_vads = torch.tensor(self._vads, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            pred_vads = self.encoder(self._texts, return_vad=True)
            mse = F.mse_loss(pred_vads, target_vads).item()
            mae = F.l1_loss(pred_vads, target_vads).item()
        
        return {
            "mse": mse,
            "mae": mae,
            "rmse": mse ** 0.5,
        }
