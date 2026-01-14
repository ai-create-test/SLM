"""
Memory Encoder Module - 整合入口

将长文本转化为模型可计算的稠密向量张量。
整合TextChunker、Tokenizer和Embedding功能。

使用示例：
    encoder = MemoryEncoder(vocab_size=100000, d_model=768)
    tensors = encoder.encode("超长文本...")  # List[Tensor]
    
    # 或获取合并结果
    result = encoder.encode_with_chunks("超长文本...")
    print(result.embeddings.shape)  # [num_chunks, max_seq, d_model]
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Any, Dict
import torch
import torch.nn as nn

from .text_chunker import TextChunker, TokenAwareChunker, TextChunk, ChunkStrategy
from .embeddings import CombinedEmbedding, EmbeddingConfig

# 尝试导入Tokenizer模块
try:
    from ..core import get_tokenizer, BPETokenizer, BaseTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False


@dataclass
class EncodingResult:
    """
    编码结果容器
    
    包含分块文本、Token ID和嵌入向量。
    """
    chunks: List[TextChunk]                    # 文本块列表
    token_ids: List[torch.Tensor]              # 每块的token ID
    embeddings: torch.Tensor                   # 嵌入张量 [num_chunks, seq_len, d_model]
    attention_mask: Optional[torch.Tensor] = None  # 注意力掩码
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_chunks(self) -> int:
        return len(self.chunks)
    
    @property
    def total_tokens(self) -> int:
        return sum(t.numel() for t in self.token_ids)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "num_chunks": self.num_chunks,
            "total_tokens": self.total_tokens,
            "embedding_shape": list(self.embeddings.shape),
            "metadata": self.metadata,
        }


class MemoryEncoder(nn.Module):
    """
    记忆编码器
    
    将长文本转化为稠密向量张量的完整流水线：
    1. 长文本分块
    2. Token化
    3. 嵌入 + 位置编码
    
    使用示例：
        encoder = MemoryEncoder(vocab_size=100000, d_model=768)
        
        # 简单使用
        embeddings = encoder.encode("超长文本...")
        
        # 获取详细结果
        result = encoder.encode_with_chunks("超长文本...")
    """
    
    def __init__(
        self,
        vocab_size: int = 100000,
        d_model: int = 768,
        max_seq_len: int = 512,
        max_chunk_size: int = 2000,
        overlap_size: int = 200,
        position_encoding: str = 'rope',
        dropout: float = 0.1,
        padding_idx: int = 0,
        tokenizer: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        初始化记忆编码器
        
        Args:
            vocab_size: 词表大小
            d_model: 嵌入维度
            max_seq_len: 每块最大序列长度（token数）
            max_chunk_size: 分块最大字符数
            overlap_size: 分块重叠字符数
            position_encoding: 位置编码类型 ('rope' 或 'sinusoidal')
            dropout: Dropout率
            padding_idx: 填充token ID
            tokenizer: 可选的Tokenizer实例
            device: 计算设备
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化Tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif TOKENIZER_AVAILABLE:
            self.tokenizer = get_tokenizer("gpt4")
        else:
            self.tokenizer = None
        
        # 初始化分块器
        if self.tokenizer is not None:
            self.chunker = TokenAwareChunker(
                tokenizer=self.tokenizer,
                max_tokens=max_seq_len,
                overlap_tokens=overlap_size // 4,
                max_chunk_size=max_chunk_size,
                overlap_size=overlap_size,
            )
        else:
            self.chunker = TextChunker(
                max_chunk_size=max_chunk_size,
                overlap_size=overlap_size,
                strategy=ChunkStrategy.SENTENCE,
            )
        
        # 初始化嵌入层
        self.embedding = CombinedEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            position_encoding=position_encoding,
            dropout=dropout,
            padding_idx=padding_idx,
        )
        
        self.to(self.device)
    
    def encode(
        self,
        text: str,
        return_tensors: bool = True,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        编码文本为张量
        
        Args:
            text: 输入文本
            return_tensors: 是否返回合并的张量
            
        Returns:
            如果return_tensors=True，返回合并的张量 [num_chunks, seq_len, d_model]
            否则返回张量列表
        """
        result = self.encode_with_chunks(text)
        
        if return_tensors:
            return result.embeddings
        else:
            return [result.embeddings[i] for i in range(result.embeddings.size(0))]
    
    def encode_with_chunks(self, text: str) -> EncodingResult:
        """
        编码文本并返回详细结果
        
        Args:
            text: 输入文本
            
        Returns:
            EncodingResult包含chunks、token_ids和embeddings
        """
        # 1. 分块
        chunks = self.chunker.chunk(text)
        
        if not chunks:
            # 空文本处理
            empty_embedding = torch.zeros(1, 1, self.d_model, device=self.device)
            return EncodingResult(
                chunks=[],
                token_ids=[torch.tensor([0], device=self.device)],
                embeddings=empty_embedding,
                metadata={"empty": True},
            )
        
        # 2. Token化每个块
        token_ids_list = []
        max_len = 0
        
        for chunk in chunks:
            if self.tokenizer is not None:
                result = self.tokenizer.encode(
                    chunk.content,
                    add_special_tokens=True,
                    max_length=self.max_seq_len,
                    truncation=True,
                )
                # 过滤负数ID（特殊标记）
                ids = [tid for tid in result.token_ids if tid >= 0]
                ids = torch.tensor(ids, dtype=torch.long, device=self.device)
            else:
                # 没有tokenizer时使用简单的字符级编码
                ids = torch.tensor(
                    [ord(c) % self.vocab_size for c in chunk.content[:self.max_seq_len]],
                    dtype=torch.long,
                    device=self.device,
                )
            
            token_ids_list.append(ids)
            max_len = max(max_len, len(ids))
        
        # 3. 填充至相同长度
        padded_ids = []
        attention_masks = []
        
        for ids in token_ids_list:
            pad_len = max_len - len(ids)
            if pad_len > 0:
                padded = torch.cat([
                    ids,
                    torch.zeros(pad_len, dtype=torch.long, device=self.device)
                ])
                mask = torch.cat([
                    torch.ones(len(ids), dtype=torch.long, device=self.device),
                    torch.zeros(pad_len, dtype=torch.long, device=self.device)
                ])
            else:
                padded = ids
                mask = torch.ones(len(ids), dtype=torch.long, device=self.device)
            
            padded_ids.append(padded)
            attention_masks.append(mask)
        
        # 堆叠为批次
        batch_ids = torch.stack(padded_ids)  # [num_chunks, max_len]
        batch_mask = torch.stack(attention_masks)  # [num_chunks, max_len]
        
        # 4. 嵌入
        with torch.no_grad():
            embeddings = self.embedding(batch_ids)  # [num_chunks, max_len, d_model]
        
        return EncodingResult(
            chunks=chunks,
            token_ids=token_ids_list,
            embeddings=embeddings,
            attention_mask=batch_mask,
            metadata={
                "max_seq_len": max_len,
                "d_model": self.d_model,
                "device": str(self.device),
            },
        )
    
    def encode_batch(
        self,
        texts: List[str],
    ) -> List[EncodingResult]:
        """
        批量编码多个文本
        
        Args:
            texts: 文本列表
            
        Returns:
            EncodingResult列表
        """
        return [self.encode_with_chunks(text) for text in texts]
    
    def get_embedding_dim(self) -> int:
        """获取嵌入维度"""
        return self.d_model
    
    def get_max_seq_len(self) -> int:
        """获取最大序列长度"""
        return self.max_seq_len
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        直接前向传播（用于训练时）
        
        Args:
            token_ids: Token ID张量 [batch, seq_len]
            
        Returns:
            嵌入张量 [batch, seq_len, d_model]
        """
        return self.embedding(token_ids)


def create_memory_encoder(
    preset: str = 'base',
    **kwargs
) -> MemoryEncoder:
    """
    创建预配置的MemoryEncoder
    
    Args:
        preset: 预设配置名 ('base', 'small', 'large')
        **kwargs: 覆盖默认参数
        
    Returns:
        MemoryEncoder实例
    """
    presets = {
        'small': {
            'vocab_size': 110000,  # 需要大于tiktoken的词表大小(~100277)
            'd_model': 256,
            'max_seq_len': 256,
        },
        'base': {
            'vocab_size': 110000,
            'd_model': 768,
            'max_seq_len': 512,
        },
        'large': {
            'vocab_size': 150000,
            'd_model': 1024,
            'max_seq_len': 1024,
        },
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    config = {**presets[preset], **kwargs}
    return MemoryEncoder(**config)
