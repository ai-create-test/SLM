"""
Data Loader - 训练数据加载器

提供段落级数据集用于 VQ-VAE 和 Dynamics 训练。
"""

from typing import List, Optional, Iterator, Tuple, Union
from dataclasses import dataclass
import random
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class ParagraphBatch:
    """段落批次"""
    texts: List[str]                    # 原始文本
    indices: torch.Tensor = None        # 批次内索引
    
    def __len__(self):
        return len(self.texts)


@dataclass
class SequenceBatch:
    """序列批次 (用于 Dynamics 训练)"""
    input_sequences: List[List[str]]    # 输入序列 [[p1, p2, p3], ...]
    target_paragraphs: List[str]        # 目标段落 [p4, ...]
    
    def __len__(self):
        return len(self.input_sequences)


class ParagraphDataset(Dataset):
    """
    段落数据集
    
    支持从文本文件或字符串列表加载段落数据。
    
    使用示例:
        # 从文件加载
        dataset = ParagraphDataset.from_file("data.txt")
        
        # 从列表加载
        dataset = ParagraphDataset(paragraphs=["段落1", "段落2", ...])
        
        # 获取单个段落
        text = dataset[0]
    """
    
    def __init__(
        self,
        paragraphs: List[str],
        min_length: int = 10,
        max_length: int = 1000,
        filter_empty: bool = True,
    ):
        """
        Args:
            paragraphs: 段落列表
            min_length: 最小段落长度 (字符)
            max_length: 最大段落长度 (字符)
            filter_empty: 是否过滤空段落
        """
        self.paragraphs = []
        
        for p in paragraphs:
            p = p.strip()
            if filter_empty and not p:
                continue
            if len(p) < min_length:
                continue
            if len(p) > max_length:
                p = p[:max_length]  # 截断
            self.paragraphs.append(p)
    
    def __len__(self) -> int:
        return len(self.paragraphs)
    
    def __getitem__(self, idx: int) -> str:
        return self.paragraphs[idx]
    
    @classmethod
    def from_file(
        cls,
        file_path: str,
        delimiter: str = "\n\n",
        encoding: str = "utf-8",
        **kwargs,
    ) -> "ParagraphDataset":
        """
        从文件加载数据集
        
        Args:
            file_path: 文件路径
            delimiter: 段落分隔符 (默认双换行)
            encoding: 文件编码
        """
        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()
        
        paragraphs = content.split(delimiter)
        return cls(paragraphs=paragraphs, **kwargs)
    
    @classmethod
    def from_lines(
        cls,
        file_path: str,
        encoding: str = "utf-8",
        **kwargs,
    ) -> "ParagraphDataset":
        """从文件按行加载 (每行一个段落)"""
        with open(file_path, "r", encoding=encoding) as f:
            paragraphs = f.readlines()
        return cls(paragraphs=paragraphs, **kwargs)
    
    @classmethod
    def synthetic(
        cls,
        num_paragraphs: int = 1000,
        min_words: int = 20,
        max_words: int = 100,
    ) -> "ParagraphDataset":
        """
        生成合成数据集 (用于测试)
        
        Args:
            num_paragraphs: 段落数量
            min_words: 最小单词数
            max_words: 最大单词数
        """
        import random
        
        # 简单的词汇表
        vocab = [
            "the", "a", "an", "this", "that", "these", "those",
            "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might",
            "I", "you", "he", "she", "it", "we", "they",
            "my", "your", "his", "her", "its", "our", "their",
            "and", "but", "or", "so", "if", "when", "while",
            "with", "in", "on", "at", "to", "for", "of", "by",
            "good", "bad", "new", "old", "big", "small",
            "first", "last", "long", "short", "high", "low",
            "make", "take", "give", "get", "go", "come", "see",
            "think", "know", "want", "need", "feel", "try",
            "system", "model", "data", "process", "method",
            "result", "analysis", "study", "research", "work",
        ]
        
        paragraphs = []
        for _ in range(num_paragraphs):
            num_words = random.randint(min_words, max_words)
            words = [random.choice(vocab) for _ in range(num_words)]
            # 首字母大写，添加句号
            words[0] = words[0].capitalize()
            paragraph = " ".join(words) + "."
            paragraphs.append(paragraph)
        
        return cls(paragraphs=paragraphs, min_length=5)


class SequenceDataset(Dataset):
    """
    序列数据集 (用于 Dynamics 训练)
    
    从段落数据集构建连续序列用于下一段落预测。
    
    使用示例:
        base = ParagraphDataset.synthetic(100)
        seq_dataset = SequenceDataset(base, seq_len=5)
        
        inputs, target = seq_dataset[0]
        # inputs: ["p1", "p2", "p3", "p4"]
        # target: "p5"
    """
    
    def __init__(
        self,
        paragraph_dataset: ParagraphDataset,
        seq_len: int = 5,
        stride: int = 1,
    ):
        """
        Args:
            paragraph_dataset: 基础段落数据集
            seq_len: 输入序列长度
            stride: 滑动窗口步长
        """
        self.paragraphs = paragraph_dataset.paragraphs
        self.seq_len = seq_len
        self.stride = stride
        
        # 计算有效样本数
        self.num_samples = max(0, (len(self.paragraphs) - seq_len) // stride)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[List[str], str]:
        start = idx * self.stride
        inputs = self.paragraphs[start:start + self.seq_len]
        target = self.paragraphs[start + self.seq_len]
        return inputs, target


def collate_paragraphs(batch: List[str]) -> ParagraphBatch:
    """段落批次整理函数"""
    return ParagraphBatch(
        texts=batch,
        indices=torch.arange(len(batch)),
    )


def collate_sequences(
    batch: List[Tuple[List[str], str]]
) -> SequenceBatch:
    """序列批次整理函数"""
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return SequenceBatch(
        input_sequences=inputs,
        target_paragraphs=targets,
    )


class ParagraphDataLoader:
    """
    段落数据加载器
    
    封装 PyTorch DataLoader，提供便捷的批次迭代。
    """
    
    def __init__(
        self,
        dataset: Union[ParagraphDataset, SequenceDataset],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ):
        if isinstance(dataset, SequenceDataset):
            collate_fn = collate_sequences
        else:
            collate_fn = collate_paragraphs
        
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
    
    def __iter__(self):
        return iter(self.loader)
    
    def __len__(self):
        return len(self.loader)
