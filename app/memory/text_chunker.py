"""
Text Chunker Module - 长文本分块处理

核心功能：
1. 将超长文本分割为可处理的块
2. 保持语义边界完整（按句子/段落）
3. 支持重叠窗口保持上下文连贯

设计原则：
- 与Tokenizer模块兼容
- 支持多种分块策略
- 为后续retrieval模块预留接口
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Iterator, Union
from enum import Enum


class ChunkStrategy(Enum):
    """分块策略枚举"""
    FIXED_SIZE = "fixed_size"          # 固定大小
    SENTENCE = "sentence"              # 按句子边界
    PARAGRAPH = "paragraph"            # 按段落边界
    SEMANTIC = "semantic"              # 语义分块（预留）


@dataclass
class TextChunk:
    """
    文本块数据结构
    
    包含块内容及其元数据，便于后续处理和溯源。
    """
    content: str                           # 块内容
    start_pos: int                         # 在原文中的起始位置
    end_pos: int                           # 在原文中的结束位置
    chunk_index: int                       # 块索引
    total_chunks: int                      # 总块数
    overlap_with_prev: int = 0             # 与前一块重叠的字符数
    metadata: dict = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.content)
    
    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"TextChunk({self.chunk_index}/{self.total_chunks}, len={len(self)}, '{preview}')"


@dataclass 
class ChunkConfig:
    """分块配置"""
    strategy: ChunkStrategy = ChunkStrategy.SENTENCE
    max_chunk_size: int = 512              # 最大块大小（字符数）
    min_chunk_size: int = 100              # 最小块大小
    overlap_size: int = 50                 # 重叠窗口大小
    respect_sentence_boundary: bool = True # 是否尊重句子边界


class BaseChunker(ABC):
    """分块器抽象基类"""
    
    @abstractmethod
    def chunk(self, text: str) -> List[TextChunk]:
        """将文本分块"""
        pass
    
    @abstractmethod
    def chunk_iter(self, text: str) -> Iterator[TextChunk]:
        """迭代式分块（节省内存）"""
        pass


class TextChunker(BaseChunker):
    """
    文本分块器
    
    将长文本按配置分割为多个语义完整的块。
    
    使用示例：
        chunker = TextChunker(max_chunk_size=512, overlap_size=50)
        chunks = chunker.chunk("非常长的文本...")
        
        for chunk in chunks:
            process(chunk.content)
    """
    
    # 句子边界正则
    SENTENCE_PATTERN = re.compile(
        r'(?<=[.!?。！？])\s+|'  # 标点后空格
        r'(?<=[.!?。！？])(?=[A-Z\u4e00-\u9fff])|'  # 标点后大写/中文
        r'\n\n+'  # 多个换行
    )
    
    # 段落边界正则
    PARAGRAPH_PATTERN = re.compile(r'\n\n+|\r\n\r\n+')
    
    def __init__(
        self,
        max_chunk_size: int = 512,
        min_chunk_size: int = 100,
        overlap_size: int = 50,
        strategy: ChunkStrategy = ChunkStrategy.SENTENCE,
        respect_sentence_boundary: bool = True,
    ):
        """
        初始化文本分块器
        
        Args:
            max_chunk_size: 最大块大小（字符）
            min_chunk_size: 最小块大小
            overlap_size: 块之间的重叠大小
            strategy: 分块策略
            respect_sentence_boundary: 是否尊重句子边界
        """
        self.config = ChunkConfig(
            strategy=strategy,
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
            overlap_size=overlap_size,
            respect_sentence_boundary=respect_sentence_boundary,
        )
    
    def chunk(self, text: str) -> List[TextChunk]:
        """
        将文本分块
        
        Args:
            text: 输入文本
            
        Returns:
            TextChunk列表
        """
        return list(self.chunk_iter(text))
    
    def chunk_iter(self, text: str) -> Iterator[TextChunk]:
        """
        迭代式分块
        
        对于超长文本，使用迭代器节省内存。
        """
        if not text or not text.strip():
            return
        
        text = text.strip()
        
        if self.config.strategy == ChunkStrategy.FIXED_SIZE:
            yield from self._chunk_fixed_size(text)
        elif self.config.strategy == ChunkStrategy.SENTENCE:
            yield from self._chunk_by_sentence(text)
        elif self.config.strategy == ChunkStrategy.PARAGRAPH:
            yield from self._chunk_by_paragraph(text)
        else:
            # 默认使用句子分块
            yield from self._chunk_by_sentence(text)
    
    def _chunk_fixed_size(self, text: str) -> Iterator[TextChunk]:
        """固定大小分块"""
        total_length = len(text)
        chunk_size = self.config.max_chunk_size
        overlap = self.config.overlap_size
        
        # 计算总块数
        if total_length <= chunk_size:
            total_chunks = 1
        else:
            effective_size = chunk_size - overlap
            total_chunks = 1 + (total_length - chunk_size + effective_size - 1) // effective_size
        
        start = 0
        chunk_index = 0
        
        while start < total_length:
            end = min(start + chunk_size, total_length)
            
            # 确定重叠
            overlap_with_prev = 0 if chunk_index == 0 else overlap
            
            yield TextChunk(
                content=text[start:end],
                start_pos=start,
                end_pos=end,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                overlap_with_prev=overlap_with_prev,
            )
            
            # 移动窗口
            start = end - overlap if end < total_length else total_length
            chunk_index += 1
    
    def _chunk_by_sentence(self, text: str) -> Iterator[TextChunk]:
        """按句子边界分块"""
        # 先按句子分割
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return
        
        chunks_data = []
        current_chunk = []
        current_length = 0
        current_start = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # 如果单个句子超过最大块大小，需要强制分割
            if sentence_len > self.config.max_chunk_size:
                # 先保存当前块
                if current_chunk:
                    chunks_data.append((current_start, ''.join(current_chunk)))
                    current_chunk = []
                    current_length = 0
                
                # 强制分割长句子
                for sub_chunk in self._force_split(sentence, current_start):
                    chunks_data.append((sub_chunk[0], sub_chunk[1]))
                current_start += sentence_len
                continue
            
            # 检查是否需要开始新块
            if current_length + sentence_len > self.config.max_chunk_size and current_chunk:
                chunks_data.append((current_start, ''.join(current_chunk)))
                
                # 重叠处理：保留一些内容到下一块
                overlap_content = self._get_overlap_content(current_chunk)
                current_start = current_start + current_length - len(overlap_content)
                current_chunk = [overlap_content] if overlap_content else []
                current_length = len(overlap_content)
            
            current_chunk.append(sentence)
            current_length += sentence_len
        
        # 保存最后一块
        if current_chunk:
            chunks_data.append((current_start, ''.join(current_chunk)))
        
        # 生成TextChunk对象
        total_chunks = len(chunks_data)
        for i, (start_pos, content) in enumerate(chunks_data):
            yield TextChunk(
                content=content,
                start_pos=start_pos,
                end_pos=start_pos + len(content),
                chunk_index=i,
                total_chunks=total_chunks,
                overlap_with_prev=self.config.overlap_size if i > 0 else 0,
            )
    
    def _chunk_by_paragraph(self, text: str) -> Iterator[TextChunk]:
        """按段落边界分块"""
        paragraphs = self.PARAGRAPH_PATTERN.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            return
        
        chunks_data = []
        current_chunk = []
        current_length = 0
        current_start = 0
        
        for para in paragraphs:
            para_len = len(para)
            
            if current_length + para_len + 2 > self.config.max_chunk_size and current_chunk:
                chunks_data.append((current_start, '\n\n'.join(current_chunk)))
                current_start += current_length
                current_chunk = []
                current_length = 0
            
            current_chunk.append(para)
            current_length += para_len + 2  # +2 for '\n\n'
        
        if current_chunk:
            chunks_data.append((current_start, '\n\n'.join(current_chunk)))
        
        total_chunks = len(chunks_data)
        for i, (start_pos, content) in enumerate(chunks_data):
            yield TextChunk(
                content=content,
                start_pos=start_pos,
                end_pos=start_pos + len(content),
                chunk_index=i,
                total_chunks=total_chunks,
            )
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割为句子"""
        # 使用正则分割，但保留分隔符
        parts = self.SENTENCE_PATTERN.split(text)
        
        sentences = []
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part + ' ')
        
        return sentences
    
    def _force_split(self, text: str, start_offset: int) -> Iterator[Tuple[int, str]]:
        """强制分割超长文本"""
        chunk_size = self.config.max_chunk_size
        for i in range(0, len(text), chunk_size):
            yield (start_offset + i, text[i:i + chunk_size])
    
    def _get_overlap_content(self, chunks: List[str]) -> str:
        """获取重叠内容（从chunk末尾取）"""
        if not chunks:
            return ""
        
        combined = ''.join(chunks)
        overlap_size = min(self.config.overlap_size, len(combined))
        
        if overlap_size <= 0:
            return ""
        
        return combined[-overlap_size:]
    
    def estimate_chunks(self, text: str) -> int:
        """估算文本会产生多少块"""
        text_len = len(text)
        if text_len <= self.config.max_chunk_size:
            return 1
        
        effective_size = self.config.max_chunk_size - self.config.overlap_size
        return 1 + (text_len - self.config.max_chunk_size) // effective_size + 1


class TokenAwareChunker(TextChunker):
    """
    Token感知的分块器
    
    与Tokenizer集成，按token数量而非字符数量分块。
    确保每个块的token数不超过模型窗口限制。
    """
    
    def __init__(
        self,
        tokenizer,  # BaseTokenizer instance
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        **kwargs
    ):
        """
        Args:
            tokenizer: Tokenizer实例
            max_tokens: 最大token数
            overlap_tokens: 重叠token数
        """
        # 估算字符到token的比率（英文约4字符/token，中文约2字符/token）
        char_per_token = 4
        
        # 从kwargs中移除可能冲突的参数
        kwargs.pop('max_chunk_size', None)
        kwargs.pop('overlap_size', None)
        
        super().__init__(
            max_chunk_size=max_tokens * char_per_token,
            overlap_size=overlap_tokens * char_per_token,
            **kwargs
        )
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
    
    def chunk(self, text: str) -> List[TextChunk]:
        """按token数量分块"""
        # 先用父类方法粗分
        rough_chunks = super().chunk(text)
        
        # 然后验证并调整每个块的token数
        refined_chunks = []
        for chunk in rough_chunks:
            # 检查token数
            result = self.tokenizer.encode(chunk.content, add_special_tokens=False)
            token_count = len(result.token_ids)
            
            if token_count <= self.max_tokens:
                refined_chunks.append(chunk)
            else:
                # 需要进一步分割
                sub_chunks = self._split_by_tokens(chunk)
                refined_chunks.extend(sub_chunks)
        
        # 重新编号
        total = len(refined_chunks)
        for i, chunk in enumerate(refined_chunks):
            chunk.chunk_index = i
            chunk.total_chunks = total
        
        return refined_chunks
    
    def _split_by_tokens(self, chunk: TextChunk) -> List[TextChunk]:
        """按token数分割单个块"""
        text = chunk.content
        result = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(result.token_ids) <= self.max_tokens:
            return [chunk]
        
        # 二分查找合适的分割点
        sub_chunks = []
        start = 0
        
        while start < len(text):
            # 尝试找到合适的结束位置
            end = min(start + len(text) // 2, len(text))
            
            while end > start:
                sub_text = text[start:end]
                sub_result = self.tokenizer.encode(sub_text, add_special_tokens=False)
                
                if len(sub_result.token_ids) <= self.max_tokens:
                    break
                end = (start + end) // 2
            
            if end <= start:
                end = start + 1  # 至少前进一个字符
            
            sub_chunks.append(TextChunk(
                content=text[start:end],
                start_pos=chunk.start_pos + start,
                end_pos=chunk.start_pos + end,
                chunk_index=0,  # 稍后重新编号
                total_chunks=0,
            ))
            
            start = end
        
        return sub_chunks
