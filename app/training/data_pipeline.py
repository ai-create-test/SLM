"""
Data Pipeline - 标准化数据处理管道

支持:
- JSONL 格式输入
- Parquet 格式输入 (大数据)
- 情感标注数据
- 数据验证和清洗
"""

from typing import List, Optional, Dict, Any, Union, Iterator
from dataclasses import dataclass
from pathlib import Path
import json
import random

import torch
from torch.utils.data import Dataset

from .data_loader import ParagraphDataset, SequenceDataset


@dataclass
class DataSample:
    """单条数据样本"""
    text: str
    emotion: Optional[str] = None
    emotion_vad: Optional[tuple] = None
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class NeuralFlowDataset(Dataset):
    """
    NeuralFlow 标准数据集
    
    支持多种输入格式，提供统一接口。
    
    数据格式 (JSONL):
        {"text": "段落内容", "emotion": "happy", "emotion_vad": [0.8, 0.5, 0.6]}
        {"text": "另一个段落"}
    
    使用示例:
        # 从 JSONL 加载
        dataset = NeuralFlowDataset.from_jsonl("data/train.jsonl")
        
        # 从 Parquet 加载
        dataset = NeuralFlowDataset.from_parquet("data/train.parquet")
        
        # 转换为训练格式
        para_dataset = dataset.to_paragraph_dataset()
        seq_dataset = dataset.to_sequence_dataset(seq_len=5)
    """
    
    def __init__(
        self,
        samples: List[DataSample],
        min_length: int = 10,
        max_length: int = 1000,
    ):
        """
        Args:
            samples: DataSample 列表
            min_length: 最小段落长度
            max_length: 最大段落长度
        """
        self.samples = []
        
        for sample in samples:
            text = sample.text.strip()
            if len(text) < min_length:
                continue
            if len(text) > max_length:
                text = text[:max_length]
                sample = DataSample(
                    text=text,
                    emotion=sample.emotion,
                    emotion_vad=sample.emotion_vad,
                    context=sample.context,
                    metadata=sample.metadata,
                )
            self.samples.append(sample)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> DataSample:
        return self.samples[idx]
    
    @property
    def texts(self) -> List[str]:
        """获取所有文本"""
        return [s.text for s in self.samples]
    
    @property
    def emotions(self) -> List[Optional[str]]:
        """获取所有情感标签"""
        return [s.emotion for s in self.samples]
    
    @property
    def has_emotion_labels(self) -> bool:
        """是否有情感标签"""
        return any(s.emotion is not None for s in self.samples)
    
    # =========================================================================
    # 加载方法
    # =========================================================================
    
    @classmethod
    def from_jsonl(
        cls,
        path: str,
        max_samples: Optional[int] = None,
        **kwargs,
    ) -> "NeuralFlowDataset":
        """
        从 JSONL 文件加载
        
        Args:
            path: JSONL 文件路径
            max_samples: 最大样本数
            **kwargs: 传递给构造函数
            
        Returns:
            NeuralFlowDataset
        """
        samples = []
        
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    sample = cls._parse_sample(data)
                    if sample:
                        samples.append(sample)
                except json.JSONDecodeError:
                    continue
        
        print(f"Loaded {len(samples)} samples from {path}")
        return cls(samples, **kwargs)
    
    @classmethod
    def from_parquet(
        cls,
        path: str,
        text_column: str = "text",
        emotion_column: Optional[str] = "emotion",
        max_samples: Optional[int] = None,
        **kwargs,
    ) -> "NeuralFlowDataset":
        """
        从 Parquet 文件加载
        
        Args:
            path: Parquet 文件路径
            text_column: 文本列名
            emotion_column: 情感列名
            max_samples: 最大样本数
            **kwargs: 传递给构造函数
            
        Returns:
            NeuralFlowDataset
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for Parquet support")
        
        df = pd.read_parquet(path)
        
        if max_samples:
            df = df.head(max_samples)
        
        samples = []
        for _, row in df.iterrows():
            text = row.get(text_column, "")
            emotion = row.get(emotion_column) if emotion_column else None
            
            if text:
                samples.append(DataSample(
                    text=str(text),
                    emotion=str(emotion) if emotion and pd.notna(emotion) else None,
                ))
        
        print(f"Loaded {len(samples)} samples from {path}")
        return cls(samples, **kwargs)
    
    @classmethod
    def from_text_file(
        cls,
        path: str,
        delimiter: str = "\n\n",
        **kwargs,
    ) -> "NeuralFlowDataset":
        """
        从纯文本文件加载 (每段落用空行分隔)
        
        Args:
            path: 文件路径
            delimiter: 段落分隔符
            **kwargs: 传递给构造函数
            
        Returns:
            NeuralFlowDataset
        """
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        paragraphs = content.split(delimiter)
        samples = [DataSample(text=p.strip()) for p in paragraphs if p.strip()]
        
        print(f"Loaded {len(samples)} paragraphs from {path}")
        return cls(samples, **kwargs)
    
    @classmethod
    def from_list(
        cls,
        texts: List[str],
        emotions: Optional[List[str]] = None,
        **kwargs,
    ) -> "NeuralFlowDataset":
        """
        从列表创建
        
        Args:
            texts: 文本列表
            emotions: 情感标签列表 (可选)
            **kwargs: 传递给构造函数
            
        Returns:
            NeuralFlowDataset
        """
        samples = []
        for i, text in enumerate(texts):
            emotion = emotions[i] if emotions and i < len(emotions) else None
            samples.append(DataSample(text=text, emotion=emotion))
        
        return cls(samples, **kwargs)
    
    @staticmethod
    def _parse_sample(data: Dict[str, Any]) -> Optional[DataSample]:
        """解析 JSON 数据为 DataSample"""
        text = data.get("text", "")
        if not text:
            return None
        
        emotion = data.get("emotion")
        emotion_vad = data.get("emotion_vad")
        if emotion_vad and isinstance(emotion_vad, list):
            emotion_vad = tuple(emotion_vad)
        
        return DataSample(
            text=text,
            emotion=emotion,
            emotion_vad=emotion_vad,
            context=data.get("context"),
            metadata=data.get("metadata"),
        )
    
    # =========================================================================
    # 转换方法
    # =========================================================================
    
    def to_paragraph_dataset(self) -> ParagraphDataset:
        """转换为段落数据集 (用于 VQ-VAE 训练)"""
        return ParagraphDataset(paragraphs=self.texts)
    
    def to_sequence_dataset(self, seq_len: int = 5) -> SequenceDataset:
        """
        转换为序列数据集 (用于 Dynamics 训练)
        
        Args:
            seq_len: 输入序列长度
            
        Returns:
            SequenceDataset
        """
        para_dataset = self.to_paragraph_dataset()
        return SequenceDataset(para_dataset, seq_len=seq_len)
    
    def to_emotion_dataset(self) -> "EmotionDataset":
        """转换为情感数据集 (用于 Emotion 训练)"""
        # 只保留有情感标签的样本
        samples_with_emotion = [s for s in self.samples if s.emotion]
        return EmotionDataset(samples_with_emotion)
    
    # =========================================================================
    # 数据分割
    # =========================================================================
    
    def split(
        self,
        train_ratio: float = 0.9,
        shuffle: bool = True,
        seed: int = 42,
    ) -> tuple:
        """
        分割数据集
        
        Args:
            train_ratio: 训练集比例
            shuffle: 是否打乱
            seed: 随机种子
            
        Returns:
            (train_dataset, eval_dataset)
        """
        samples = list(self.samples)
        
        if shuffle:
            random.seed(seed)
            random.shuffle(samples)
        
        split_idx = int(len(samples) * train_ratio)
        train_samples = samples[:split_idx]
        eval_samples = samples[split_idx:]
        
        return (
            NeuralFlowDataset(train_samples),
            NeuralFlowDataset(eval_samples),
        )
    
    # =========================================================================
    # 保存方法
    # =========================================================================
    
    def to_jsonl(self, path: str) -> None:
        """保存为 JSONL 格式"""
        with open(path, "w", encoding="utf-8") as f:
            for sample in self.samples:
                data = {"text": sample.text}
                if sample.emotion:
                    data["emotion"] = sample.emotion
                if sample.emotion_vad:
                    data["emotion_vad"] = list(sample.emotion_vad)
                if sample.context:
                    data["context"] = sample.context
                if sample.metadata:
                    data["metadata"] = sample.metadata
                
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        
        print(f"Saved {len(self.samples)} samples to {path}")


class EmotionDataset(Dataset):
    """
    情感标注数据集
    
    用于 Stage 3 Emotion 训练。
    """
    
    def __init__(self, samples: List[DataSample]):
        self.samples = [s for s in samples if s.emotion]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        return {
            "text": sample.text,
            "emotion": sample.emotion,
            "emotion_vad": sample.emotion_vad,
        }


# ============================================================================
# 数据验证
# ============================================================================

def validate_dataset(
    dataset: NeuralFlowDataset,
    min_samples: int = 100,
    check_emotion: bool = False,
) -> Dict[str, Any]:
    """
    验证数据集
    
    Args:
        dataset: 数据集
        min_samples: 最小样本数
        check_emotion: 是否检查情感标签
        
    Returns:
        验证结果字典
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }
    
    # 检查样本数
    if len(dataset) < min_samples:
        results["errors"].append(
            f"Not enough samples: {len(dataset)} < {min_samples}"
        )
        results["valid"] = False
    
    # 统计
    text_lengths = [len(s.text) for s in dataset.samples]
    results["stats"]["num_samples"] = len(dataset)
    results["stats"]["avg_length"] = sum(text_lengths) / len(text_lengths) if text_lengths else 0
    results["stats"]["min_length"] = min(text_lengths) if text_lengths else 0
    results["stats"]["max_length"] = max(text_lengths) if text_lengths else 0
    
    # 检查情感标签
    if check_emotion:
        num_with_emotion = sum(1 for s in dataset.samples if s.emotion)
        results["stats"]["num_with_emotion"] = num_with_emotion
        
        if num_with_emotion == 0:
            results["warnings"].append("No emotion labels found")
    
    return results
