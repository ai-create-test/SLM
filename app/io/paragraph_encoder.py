"""
Paragraph Encoder - 段落级语义编码器

核心功能：将段落压缩为固定维度的潜向量

架构:
    Paragraph Text
        ↓
    Pre-trained LM (frozen) → Contextual Embeddings [seq_len, d_model]
        ↓
    Pooling Layer → Paragraph Vector [d_model]
        ↓
    Projection → Pre-quantized [d_latent]
        ↓
    VQ Codebook → Discrete Latent Code [d_latent]
"""

from typing import Optional, List, Union, Tuple, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces.base_module import BaseModule, ModuleOutput, LatentVector
from ..interfaces.config import ModelConfig
from ..interfaces.registry import Registry
from .vq_codebook import VQCodebook, VQOutput
from .tokenizer_wrapper import TokenizerWrapper, FallbackTokenizer, get_tokenizer, TokenizerOutput
from .base_lm import BaseLM, FallbackLM, get_base_lm, LMOutput


@dataclass
class EncoderOutput(ModuleOutput):
    """编码器输出"""
    latent: LatentVector = None          # 潜向量
    vq_output: Optional[VQOutput] = None # VQ 输出 (包含损失)
    pooled: torch.Tensor = None          # 池化后的向量 (量化前)


class AttentionPooling(nn.Module):
    """
    注意力池化层
    
    使用可学习的 query 向量对序列进行加权池化。
    比简单的 mean pooling 更能捕捉关键信息。
    """
    
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: 序列表示 [batch, seq_len, d_model]
            mask: 注意力掩码 [batch, seq_len]
            
        Returns:
            池化后的向量 [batch, d_model]
        """
        batch_size = x.shape[0]
        query = self.query.expand(batch_size, -1, -1)
        
        # Cross-attention: query attends to sequence
        attn_out, _ = self.attention(query, x, x, key_padding_mask=mask)
        
        # 归一化并取出
        pooled = self.norm(attn_out.squeeze(1))
        return pooled


class PerceiverPooling(nn.Module):
    """
    Perceiver 风格的池化层
    
    使用固定数量的 latent 向量来压缩变长序列。
    """
    
    def __init__(
        self,
        d_model: int,
        num_latents: int = 4,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, d_model))
        
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: 序列表示 [batch, seq_len, d_model]
            
        Returns:
            池化后的向量 [batch, d_model]
        """
        batch_size = x.shape[0]
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        
        for cross_attn, norm in zip(self.cross_attns, self.norms):
            attn_out, _ = cross_attn(latents, x, x, key_padding_mask=mask)
            latents = norm(latents + attn_out)
        
        # 对所有 latent 取平均
        pooled = self.final_norm(latents.mean(dim=1))
        return pooled


@Registry.register("encoder", "paragraph")
class ParagraphEncoder(BaseModule):
    """
    段落级语义编码器
    
    将自然语言段落压缩为固定维度的潜向量。
    
    使用示例:
        encoder = ParagraphEncoder.from_config(config)
        
        # 编码段落
        output = encoder("这是一个段落文本...")
        z = output.latent.vector  # [d_latent]
        
        # 批量编码
        output = encoder(["段落1", "段落2", "段落3"])
        z = output.latent.vector  # [batch, d_latent]
    """
    
    MODULE_TYPE = "encoder"
    
    def __init__(
        self,
        d_model: int = 768,
        d_latent: int = 512,
        codebook_size: int = 8192,
        num_codebooks: int = 4,
        commitment_cost: float = 0.25,
        pooling_type: str = "attention",
        use_vq: bool = True,
        base_model_name: str = "bert-base-uncased",
        freeze_base: bool = True,
        dropout: float = 0.1,
        max_length: int = 256,
        tokenizer: Optional[Any] = None,
    ):
        """
        Args:
            d_model: 基础模型的隐藏维度
            d_latent: 潜空间维度
            codebook_size: VQ 码本大小
            num_codebooks: 多头 VQ 数量
            commitment_cost: VQ 承诺损失系数
            pooling_type: 池化类型 ('attention', 'perceiver', 'mean')
            use_vq: 是否使用向量量化
            base_model_name: 基础预训练模型名称
            freeze_base: 是否冻结基础模型
            dropout: Dropout 率
            max_length: Tokenizer 最大长度
            tokenizer: 可选的预加载 Tokenizer
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_latent = d_latent
        self.use_vq = use_vq
        self.base_model_name = base_model_name
        self.freeze_base = freeze_base
        self.max_length = max_length
        
        # ========== Tokenizer ==========
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            # 尝试加载 HuggingFace tokenizer，失败则使用备用
            self.tokenizer = get_tokenizer(
                model_name=base_model_name,
                max_length=max_length,
                fallback=True,
            )
        
        # ========== 基础语言模型 ==========
        # 使用真实的预训练模型
        self.base_model = get_base_lm(
            model_name=base_model_name,
            freeze=freeze_base,
            fallback=True,
        )
        
        # 确保 d_model 匹配
        actual_d_model = getattr(self.base_model, 'd_model', d_model)
        if actual_d_model != d_model:
            # 如果预训练模型的维度不同，添加投影层
            self.model_proj = nn.Linear(actual_d_model, d_model)
            self._actual_d_model = actual_d_model
        else:
            self.model_proj = None
            self._actual_d_model = d_model
        
        # ========== 池化层 ==========
        if pooling_type == "attention":
            self.pooling = AttentionPooling(d_model)
        elif pooling_type == "perceiver":
            self.pooling = PerceiverPooling(d_model)
        else:  # mean
            self.pooling = None
        
        # ========== 投影层 ==========
        self.pre_proj = nn.Sequential(
            nn.Linear(d_model, d_latent),
            nn.LayerNorm(d_latent),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_latent, d_latent),
        )
        
        # ========== VQ 码本 ==========
        if use_vq:
            self.vq_codebook = VQCodebook(
                d_latent=d_latent,
                codebook_size=codebook_size,
                num_codebooks=num_codebooks,
                commitment_cost=commitment_cost,
            )
        else:
            self.vq_codebook = None
    
    def get_base_model_info(self) -> dict:
        """
        获取基础模型信息
        """
        return {
            "model_name": getattr(self.base_model, 'model_name', 'unknown'),
            "d_model": getattr(self.base_model, 'd_model', self.d_model),
            "is_frozen": getattr(self.base_model, 'is_frozen', self.freeze_base),
        }
    
    def forward(
        self,
        text: Union[str, List[str], torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> EncoderOutput:
        """
        编码段落
        
        Args:
            text: 输入文本 (字符串、字符串列表或 token IDs)
            attention_mask: 注意力掩码
            
        Returns:
            EncoderOutput
        """
        # 处理文本输入
        if isinstance(text, str):
            text = [text]
        
        if isinstance(text, list):
            # 使用真实的 Tokenizer
            tokenizer_output = self.tokenizer.encode_batch(text, max_length=self.max_length)
            token_ids = tokenizer_output.input_ids.to(self.device)
            if attention_mask is None:
                attn_mask = tokenizer_output.attention_mask.to(self.device)
            else:
                attn_mask = attention_mask
        else:
            token_ids = text
            attn_mask = attention_mask
        
        # 基础模型编码
        # [batch, seq_len, d_model]
        lm_output = self.base_model(token_ids, attention_mask=attn_mask)
        
        # 提取 hidden states
        if hasattr(lm_output, 'last_hidden_state'):
            hidden_states = lm_output.last_hidden_state
        else:
            hidden_states = lm_output
        
        # 如果需要投影维度
        if self.model_proj is not None:
            hidden_states = self.model_proj(hidden_states)
        
        # 将 attention_mask 转换为 key_padding_mask 格式给池化层
        if attn_mask is not None:
            padding_mask = (attn_mask == 0)
        else:
            padding_mask = None
        
        # 池化
        if self.pooling is not None:
            pooled = self.pooling(hidden_states, padding_mask)
        else:
            # Mean pooling
            if attn_mask is not None:
                # attn_mask: 1 = real token, 0 = padding
                mask = attn_mask.unsqueeze(-1).float()
                pooled = (hidden_states * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            else:
                pooled = hidden_states.mean(dim=1)
        
        # 投影到潜空间
        z_pre = self.pre_proj(pooled)  # [batch, d_latent]
        
        # 向量量化
        if self.use_vq and self.vq_codebook is not None:
            vq_output = self.vq_codebook(z_pre)
            z = vq_output.quantized
            codebook_indices = vq_output.indices
        else:
            vq_output = None
            z = z_pre
            codebook_indices = None
        
        # 构建 LatentVector
        latent = LatentVector(
            vector=z,
            codebook_indices=codebook_indices,
            metadata={"pooling_type": type(self.pooling).__name__ if self.pooling else "mean"},
        )
        
        return EncoderOutput(
            data=z,
            latent=latent,
            vq_output=vq_output,
            pooled=pooled,
        )
    
    def tokenize(self, texts: Union[str, List[str]]) -> TokenizerOutput:
        """
        对文本进行分词
        
        Args:
            texts: 单个文本或文本列表
            
        Returns:
            TokenizerOutput 包含 input_ids 和 attention_mask
        """
        if isinstance(texts, str):
            texts = [texts]
        return self.tokenizer.encode_batch(texts, max_length=self.max_length)
    
    @property
    def vocab_size(self) -> int:
        """获取词表大小"""
        return self.tokenizer.vocab_size
    
    def encode_text(self, text: Union[str, List[str]]) -> LatentVector:
        """
        便捷方法：直接返回 LatentVector
        """
        output = self.forward(text)
        return output.latent
    
    @classmethod
    def from_config(cls, config: ModelConfig, **kwargs) -> "ParagraphEncoder":
        """从配置创建实例"""
        return cls(
            d_model=config.d_model,
            d_latent=config.d_latent,
            codebook_size=config.codebook_size,
            num_codebooks=config.num_codebooks,
            commitment_cost=config.commitment_cost,
            dropout=config.dropout,
            **kwargs,
        )
    
    def get_loss(self, output: EncoderOutput) -> torch.Tensor:
        """
        获取编码器相关的损失
        
        主要是 VQ 的 commitment loss 和 codebook loss。
        """
        if output.vq_output is None:
            return torch.tensor(0.0)
        
        return output.vq_output.commitment_loss + output.vq_output.codebook_loss
