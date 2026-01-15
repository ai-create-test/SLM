"""
Semantic Segmenter - 语义段落分割

将长文本智能分割为语义完整的段落。

设计原则:
1. 保持语义完整性 (不在句子中间切分)
2. 平衡段落长度
3. 支持多种分割策略
"""

from typing import List, Optional, Iterator
from dataclasses import dataclass
import re


@dataclass
class Paragraph:
    """段落数据结构"""
    content: str                    # 段落内容
    start_pos: int                  # 在原文中的起始位置
    end_pos: int                    # 在原文中的结束位置
    paragraph_index: int            # 段落索引
    total_paragraphs: int           # 总段落数
    
    def __len__(self) -> int:
        return len(self.content)
    
    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Paragraph({self.paragraph_index}/{self.total_paragraphs}, len={len(self)}, '{preview}')"


class SemanticSegmenter:
    """
    语义段落分割器
    
    将长文本分割为语义完整的段落，用于 VQ-VAE 编码。
    
    使用示例:
        segmenter = SemanticSegmenter(
            max_paragraph_len=500,
            min_paragraph_len=100,
        )
        
        paragraphs = segmenter.segment("很长的文本...")
        for p in paragraphs:
            latent = encoder.encode(p.content)
    """
    
    # 段落边界正则
    PARAGRAPH_PATTERN = re.compile(r'\n\s*\n|\r\n\s*\r\n')
    
    # 句子边界正则
    SENTENCE_PATTERN = re.compile(
        r'(?<=[.!?。！？])\s+|'     # 标点后空格
        r'(?<=[.!?。！？])(?=[A-Z\u4e00-\u9fff])|'  # 标点后大写/中文
        r'\n(?=[^\s])'              # 换行后非空白
    )
    
    def __init__(
        self,
        max_paragraph_len: int = 500,
        min_paragraph_len: int = 50,
        overlap_sentences: int = 1,
        respect_hard_breaks: bool = True,
    ):
        """
        Args:
            max_paragraph_len: 最大段落长度 (字符)
            min_paragraph_len: 最小段落长度
            overlap_sentences: 段落间重叠的句子数
            respect_hard_breaks: 是否尊重硬换行 (\\n\\n)
        """
        self.max_paragraph_len = max_paragraph_len
        self.min_paragraph_len = min_paragraph_len
        self.overlap_sentences = overlap_sentences
        self.respect_hard_breaks = respect_hard_breaks
    
    def segment(self, text: str) -> List[Paragraph]:
        """
        分割文本为段落
        
        Args:
            text: 输入文本
            
        Returns:
            Paragraph 列表
        """
        return list(self.segment_iter(text))
    
    def segment_iter(self, text: str) -> Iterator[Paragraph]:
        """
        迭代式分割 (节省内存)
        """
        if not text or not text.strip():
            return
        
        text = text.strip()
        
        if self.respect_hard_breaks:
            # 先按硬换行分割
            hard_paragraphs = self.PARAGRAPH_PATTERN.split(text)
            hard_paragraphs = [p.strip() for p in hard_paragraphs if p.strip()]
        else:
            hard_paragraphs = [text]
        
        all_paragraphs = []
        current_pos = 0
        
        for hard_para in hard_paragraphs:
            # 如果段落太长，进一步分割
            if len(hard_para) > self.max_paragraph_len:
                sub_paragraphs = self._split_long_paragraph(hard_para)
            else:
                sub_paragraphs = [hard_para]
            
            for sub_para in sub_paragraphs:
                if len(sub_para) >= self.min_paragraph_len or not all_paragraphs:
                    all_paragraphs.append((current_pos, sub_para))
                elif all_paragraphs:
                    # 合并到上一个段落
                    prev_pos, prev_content = all_paragraphs[-1]
                    all_paragraphs[-1] = (prev_pos, prev_content + " " + sub_para)
                
                current_pos += len(sub_para) + 1
        
        # 生成 Paragraph 对象
        total = len(all_paragraphs)
        for idx, (start_pos, content) in enumerate(all_paragraphs):
            yield Paragraph(
                content=content,
                start_pos=start_pos,
                end_pos=start_pos + len(content),
                paragraph_index=idx,
                total_paragraphs=total,
            )
    
    def _split_long_paragraph(self, text: str) -> List[str]:
        """
        分割过长的段落
        
        尽量在句子边界处分割。
        """
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            # 强制按长度分割
            return self._force_split(text)
        
        paragraphs = []
        current_para = []
        current_len = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # 单个句子超长，强制分割
            if sentence_len > self.max_paragraph_len:
                if current_para:
                    paragraphs.append(' '.join(current_para))
                    current_para = []
                    current_len = 0
                
                paragraphs.extend(self._force_split(sentence))
                continue
            
            # 检查是否需要开始新段落
            if current_len + sentence_len > self.max_paragraph_len and current_para:
                paragraphs.append(' '.join(current_para))
                
                # 重叠处理
                if self.overlap_sentences > 0 and len(current_para) >= self.overlap_sentences:
                    current_para = current_para[-self.overlap_sentences:]
                    current_len = sum(len(s) for s in current_para)
                else:
                    current_para = []
                    current_len = 0
            
            current_para.append(sentence)
            current_len += sentence_len
        
        # 最后一个段落
        if current_para:
            paragraphs.append(' '.join(current_para))
        
        return paragraphs
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割为句子"""
        parts = self.SENTENCE_PATTERN.split(text)
        sentences = [s.strip() for s in parts if s.strip()]
        return sentences
    
    def _force_split(self, text: str) -> List[str]:
        """强制按长度分割"""
        parts = []
        for i in range(0, len(text), self.max_paragraph_len):
            parts.append(text[i:i + self.max_paragraph_len])
        return parts
    
    def estimate_paragraphs(self, text: str) -> int:
        """估算文本会产生多少段落"""
        text_len = len(text)
        if text_len <= self.max_paragraph_len:
            return 1
        
        avg_para_len = (self.max_paragraph_len + self.min_paragraph_len) // 2
        return max(1, text_len // avg_para_len)
