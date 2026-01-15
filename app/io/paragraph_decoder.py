"""
Paragraph Decoder - 潜向量到段落的解码器

核心功能：从潜向量生成自然语言段落

架构:
    Latent Vector [d_latent]
        ↓
    Projection → [d_model]
        ↓
    Conditional Cross-Attention (Emotion, Scene)
        ↓
    Autoregressive Decoder
        ↓
    Output Paragraph Text
"""

from typing import Optional, List, Union, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces.base_module import BaseModule, ModuleOutput, LatentVector
from ..interfaces.config import ModelConfig
from ..interfaces.registry import Registry


@dataclass
class DecoderOutput(ModuleOutput):
    """解码器输出"""
    text: Optional[List[str]]          # 生成的文本 (推理时)
    logits: torch.Tensor               # 输出 logits [batch, seq_len, vocab_size]
    loss: Optional[torch.Tensor]       # 损失 (训练时)


@Registry.register("decoder", "paragraph")
class ParagraphDecoder(BaseModule):
    """
    段落解码器
    
    将潜向量解码为自然语言段落。
    支持条件生成 (情感/场景调制)。
    
    使用示例:
        decoder = ParagraphDecoder.from_config(config)
        
        # 推理生成
        output = decoder.generate(latent_vector, max_length=256)
        text = output.text  # ["生成的段落..."]
        
        # 训练时计算损失
        output = decoder(latent_vector, target_ids, emotion=emotion_vec)
        loss = output.loss
    """
    
    MODULE_TYPE = "decoder"
    
    def __init__(
        self,
        d_latent: int = 512,
        d_model: int = 768,
        vocab_size: int = 50000,
        max_length: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        use_rope: bool = True,
    ):
        """
        Args:
            d_latent: 潜空间维度
            d_model: 解码器隐藏维度
            vocab_size: 词表大小
            max_length: 最大生成长度
            num_layers: Transformer 层数
            num_heads: 注意力头数
            d_ff: FFN 隐藏维度
            dropout: Dropout 率
            use_rope: 是否使用 RoPE 位置编码
        """
        super().__init__()
        
        self.d_latent = d_latent
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # ========== 潜向量投影 ==========
        self.latent_proj = nn.Sequential(
            nn.Linear(d_latent, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        # ========== Token 嵌入 ==========
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # ========== 位置编码 ==========
        if use_rope:
            self.position_encoding = None  # RoPE 在 attention 中应用
            self.use_rope = True
        else:
            self.position_encoding = self._create_sinusoidal_pe(max_length, d_model)
            self.use_rope = False
        
        # ========== Transformer 解码器 ==========
        # 自注意力 + 交叉注意力 (对潜向量)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # ========== 输出层 ==========
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重绑定 (可选)
        # self.output_proj.weight = self.token_embedding.weight
        
        # ========== Dropout ==========
        self.dropout = nn.Dropout(dropout)
    
    def _create_sinusoidal_pe(self, max_len: int, d_model: int) -> nn.Parameter:
        """创建正弦位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(
        self,
        latent: Union[LatentVector, torch.Tensor],
        target_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        emotion: Optional[torch.Tensor] = None,
        scene: Optional[torch.Tensor] = None,
    ) -> DecoderOutput:
        """
        前向传播
        
        Args:
            latent: 潜向量 [batch, d_latent] 或 LatentVector
            target_ids: 目标 token IDs [batch, seq_len] (训练时)
            attention_mask: 注意力掩码
            emotion: 情感向量 [batch, d_emotion] (可选)
            scene: 场景向量 [batch, d_scene] (可选)
            
        Returns:
            DecoderOutput
        """
        # 提取潜向量
        if isinstance(latent, LatentVector):
            z = latent.vector
        else:
            z = latent
        
        batch_size = z.shape[0]
        
        # 投影潜向量作为 memory
        # [batch, 1, d_model]
        memory = self.latent_proj(z).unsqueeze(1)
        
        # TODO: 融合情感/场景条件 (后续由 modulation 模块处理)
        
        if target_ids is not None:
            # 训练模式：teacher forcing
            return self._forward_train(memory, target_ids, attention_mask)
        else:
            # 推理模式：自回归生成
            return self._forward_inference(memory, batch_size)
    
    def _forward_train(
        self,
        memory: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> DecoderOutput:
        """训练模式前向传播"""
        batch_size, seq_len = target_ids.shape
        
        # Token 嵌入
        token_emb = self.token_embedding(target_ids)  # [batch, seq_len, d_model]
        
        # 位置编码
        if self.position_encoding is not None:
            token_emb = token_emb + self.position_encoding[:, :seq_len, :]
        
        token_emb = self.dropout(token_emb)
        
        # 因果掩码
        causal_mask = self._generate_causal_mask(seq_len, target_ids.device)
        
        # Transformer 解码
        hidden = self.decoder(
            tgt=token_emb,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=attention_mask,
        )
        
        # 输出层
        hidden = self.output_norm(hidden)
        logits = self.output_proj(hidden)  # [batch, seq_len, vocab_size]
        
        # 计算损失 (shift by 1)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=0,  # padding token
        )
        
        return DecoderOutput(
            data=hidden,
            text=None,
            logits=logits,
            loss=loss,
        )
    
    def _forward_inference(
        self,
        memory: torch.Tensor,
        batch_size: int,
    ) -> DecoderOutput:
        """推理模式：自回归生成"""
        device = memory.device
        
        # 初始化为 BOS token
        generated = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        
        all_logits = []
        
        for _ in range(self.max_length - 1):
            # Token 嵌入
            token_emb = self.token_embedding(generated)
            
            # 位置编码
            if self.position_encoding is not None:
                seq_len = generated.shape[1]
                token_emb = token_emb + self.position_encoding[:, :seq_len, :]
            
            # 因果掩码
            causal_mask = self._generate_causal_mask(generated.shape[1], device)
            
            # 解码
            hidden = self.decoder(
                tgt=token_emb,
                memory=memory,
                tgt_mask=causal_mask,
            )
            
            hidden = self.output_norm(hidden)
            logits = self.output_proj(hidden[:, -1:, :])  # 只取最后一个位置
            all_logits.append(logits)
            
            # 采样下一个 token
            next_token = logits.argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)
            
            # 检查是否全部生成 EOS
            if (next_token == 2).all():  # 2 = EOS
                break
        
        # 拼接所有 logits
        full_logits = torch.cat(all_logits, dim=1)
        
        # 解码为文本 (占位)
        text = self._decode_tokens_placeholder(generated)
        
        return DecoderOutput(
            data=generated,
            text=text,
            logits=full_logits,
            loss=None,
        )
    
    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """生成因果注意力掩码"""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def _decode_tokens_placeholder(self, token_ids: torch.Tensor) -> List[str]:
        """
        占位解码器
        
        实际实现应使用真实的 tokenizer。
        """
        texts = []
        for ids in token_ids:
            chars = [chr(int(i) % 128) for i in ids if i > 2]  # 跳过特殊 token
            texts.append(''.join(chars))
        return texts
    
    def generate(
        self,
        latent: Union[LatentVector, torch.Tensor],
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        emotion: Optional[torch.Tensor] = None,
        scene: Optional[torch.Tensor] = None,
    ) -> DecoderOutput:
        """
        生成文本
        
        支持更多采样参数。
        """
        if max_length is not None:
            old_max = self.max_length
            self.max_length = max_length
        
        output = self.forward(latent, emotion=emotion, scene=scene)
        
        if max_length is not None:
            self.max_length = old_max
        
        return output
    
    @classmethod
    def from_config(cls, config: ModelConfig, **kwargs) -> "ParagraphDecoder":
        """从配置创建实例"""
        return cls(
            d_latent=config.d_latent,
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            max_length=config.max_paragraph_len,
            dropout=config.dropout,
            **kwargs,
        )
