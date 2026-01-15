"""
Base Language Model - 预训练语言模型包装器

提供统一的接口封装 HuggingFace 预训练模型。

支持的模型:
- BERT / RoBERTa / DistilBERT (编码器)
- GPT2 / GPT-Neo (解码器)
- T5 / BART (编解码器)

设计原则:
1. 与 HuggingFace transformers 库兼容
2. 提供简单的 encode 接口返回 hidden states
3. 支持冻结/微调切换
4. 自动处理模型下载和缓存
"""

from typing import Optional, List, Union, Dict, Any, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn

# 尝试导入 transformers
try:
    from transformers import (
        AutoModel,
        AutoConfig,
        PreTrainedModel,
        PretrainedConfig,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    PreTrainedModel = nn.Module
    PretrainedConfig = object


@dataclass
class LMOutput:
    """
    语言模型输出
    """
    last_hidden_state: torch.Tensor    # 最后一层隐藏状态 [batch, seq_len, d_model]
    pooler_output: Optional[torch.Tensor] = None  # 池化输出 [batch, d_model]
    all_hidden_states: Optional[Tuple[torch.Tensor, ...]] = None  # 所有层的隐藏状态
    attentions: Optional[Tuple[torch.Tensor, ...]] = None  # 注意力权重


class BaseLM(nn.Module):
    """
    预训练语言模型包装器
    
    提供统一的接口，封装 HuggingFace 预训练模型。
    
    使用示例:
        # 创建
        lm = BaseLM.from_pretrained("bert-base-uncased")
        
        # 编码
        output = lm(input_ids, attention_mask)
        hidden_states = output.last_hidden_state  # [batch, seq_len, 768]
        
        # 冻结/解冻
        lm.freeze()
        lm.unfreeze()
    """
    
    # 预设模型映射
    PRESET_MODELS: Dict[str, Tuple[str, int]] = {
        # 模型名 -> (HuggingFace 名称, 隐藏维度)
        "bert": ("bert-base-uncased", 768),
        "bert-chinese": ("bert-base-chinese", 768),
        "bert-large": ("bert-large-uncased", 1024),
        "roberta": ("roberta-base", 768),
        "distilbert": ("distilbert-base-uncased", 768),
        "albert": ("albert-base-v2", 768),
        "electra": ("google/electra-base-discriminator", 768),
    }
    
    def __init__(
        self,
        model: PreTrainedModel,
        d_model: int,
        model_name: str = "unknown",
        freeze: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ):
        """
        Args:
            model: HuggingFace 预训练模型实例
            d_model: 隐藏维度
            model_name: 模型名称 (用于记录)
            freeze: 是否冻结参数
            output_hidden_states: 是否输出所有层隐藏状态
            output_attentions: 是否输出注意力权重
        """
        super().__init__()
        
        self.model = model
        self.d_model = d_model
        self.model_name = model_name
        self._frozen = False
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        
        if freeze:
            self.freeze()
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        freeze: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> "BaseLM":
        """
        从预训练模型加载
        
        Args:
            model_name_or_path: 模型名称或路径
                - HuggingFace 模型名: "bert-base-uncased"
                - 本地路径: "./my_model"
                - 预设名称: "bert", "roberta"
            freeze: 是否冻结参数
            output_hidden_states: 是否输出所有层隐藏状态
            output_attentions: 是否输出注意力权重
            **kwargs: 传递给 AutoModel 的额外参数
            
        Returns:
            BaseLM 实例
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers 库未安装。请运行: pip install transformers"
            )
        
        # 检查预设
        if model_name_or_path in cls.PRESET_MODELS:
            hf_name, d_model = cls.PRESET_MODELS[model_name_or_path]
        else:
            hf_name = model_name_or_path
            d_model = None
        
        # 加载配置
        config = AutoConfig.from_pretrained(hf_name, **kwargs)
        
        # 获取隐藏维度
        if d_model is None:
            d_model = getattr(config, "hidden_size", 768)
        
        # 加载模型
        model = AutoModel.from_pretrained(
            hf_name,
            config=config,
            **kwargs,
        )
        
        return cls(
            model=model,
            d_model=d_model,
            model_name=model_name_or_path,
            freeze=freeze,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> LMOutput:
        """
        前向传播
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: 注意力掩码 [batch, seq_len]
            token_type_ids: Segment IDs [batch, seq_len]
            
        Returns:
            LMOutput
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=self.output_hidden_states,
            output_attentions=self.output_attentions,
            **kwargs,
        )
        
        return LMOutput(
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=getattr(outputs, "pooler_output", None),
            all_hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )
    
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        简化的编码接口
        
        Returns:
            last_hidden_state [batch, seq_len, d_model]
        """
        output = self.forward(input_ids, attention_mask)
        return output.last_hidden_state
    
    def freeze(self) -> None:
        """冻结所有参数"""
        for param in self.model.parameters():
            param.requires_grad = False
        self._frozen = True
    
    def unfreeze(self) -> None:
        """解冻所有参数"""
        for param in self.model.parameters():
            param.requires_grad = True
        self._frozen = False
    
    def unfreeze_last_n_layers(self, n: int) -> None:
        """
        只解冻最后 n 层
        
        常用于微调策略。
        """
        # 首先冻结所有
        self.freeze()
        
        # 获取 encoder 层
        encoder = getattr(self.model, "encoder", None)
        if encoder is None:
            encoder = getattr(self.model, "transformer", None)
        
        if encoder is not None:
            layers = getattr(encoder, "layer", None)
            if layers is None:
                layers = getattr(encoder, "layers", None)
            
            if layers is not None and len(layers) >= n:
                for layer in layers[-n:]:
                    for param in layer.parameters():
                        param.requires_grad = True
        
        # 始终解冻 pooler (如果存在)
        pooler = getattr(self.model, "pooler", None)
        if pooler is not None:
            for param in pooler.parameters():
                param.requires_grad = True
    
    @property
    def is_frozen(self) -> bool:
        """是否已冻结"""
        return self._frozen
    
    @property
    def num_parameters(self) -> int:
        """总参数数量"""
        return sum(p.numel() for p in self.model.parameters())
    
    @property
    def num_trainable_parameters(self) -> int:
        """可训练参数数量"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_input_embeddings(self) -> nn.Module:
        """获取输入嵌入层"""
        return self.model.get_input_embeddings()
    
    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        """调整词表大小"""
        self.model.resize_token_embeddings(new_num_tokens)
    
    def __repr__(self) -> str:
        frozen_str = "frozen" if self._frozen else "trainable"
        return f"BaseLM(model={self.model_name}, d_model={self.d_model}, {frozen_str})"


class FallbackLM(nn.Module):
    """
    备用语言模型 (当 transformers 不可用时)
    
    使用简单的 Transformer Encoder 作为占位符。
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        d_model: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 3072,
        dropout: float = 0.1,
        max_position_embeddings: int = 512,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.model_name = "fallback"
        self._frozen = False
        
        # 词嵌入
        self.word_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        
        # LayerNorm 和 Dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> LMOutput:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 位置 IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # 嵌入
        word_emb = self.word_embeddings(input_ids)
        pos_emb = self.position_embeddings(position_ids)
        embeddings = word_emb + pos_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # 转换 attention_mask 为 key_padding_mask 格式
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        
        # Encoder
        hidden_states = self.encoder(embeddings, src_key_padding_mask=key_padding_mask)
        
        # 池化 (使用 [CLS] 位置)
        pooler_output = hidden_states[:, 0, :]
        
        return LMOutput(
            last_hidden_state=hidden_states,
            pooler_output=pooler_output,
        )
    
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = self.forward(input_ids, attention_mask)
        return output.last_hidden_state
    
    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        self._frozen = True
    
    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
        self._frozen = False
    
    @property
    def is_frozen(self) -> bool:
        return self._frozen


def get_base_lm(
    model_name: str = "bert-base-uncased",
    freeze: bool = True,
    fallback: bool = True,
    **kwargs,
) -> Union[BaseLM, FallbackLM]:
    """
    获取基础语言模型的统一入口
    
    Args:
        model_name: 模型名称
        freeze: 是否冻结参数
        fallback: 是否在加载失败时使用备用方案
        **kwargs: 额外参数
        
    Returns:
        BaseLM 或 FallbackLM 实例
    """
    if HAS_TRANSFORMERS:
        try:
            return BaseLM.from_pretrained(model_name, freeze=freeze, **kwargs)
        except Exception as e:
            if fallback:
                print(f"Warning: Failed to load {model_name}, using fallback LM. Error: {e}")
                return FallbackLM(**kwargs)
            raise
    else:
        if fallback:
            print("Warning: transformers not installed, using fallback LM")
            return FallbackLM(**kwargs)
        raise ImportError("transformers 库未安装")
