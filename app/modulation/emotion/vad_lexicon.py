"""
VAD Lexicon - 情感词典管理器

加载和管理中英文情感词典，支持精确匹配查询。
"""

import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field

import torch


@dataclass
class VADCoordinate:
    """VAD 坐标点"""
    valence: float      # [-1.0, 1.0] 效价
    arousal: float      # [-1.0, 1.0] 唤醒度
    dominance: float    # [-1.0, 1.0] 支配度
    source: str = "lexicon"  # 来源标记
    
    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """转换为 PyTorch 张量 [3]"""
        t = torch.tensor([self.valence, self.arousal, self.dominance], dtype=torch.float32)
        if device is not None:
            t = t.to(device)
        return t
    
    def blend(self, other: "VADCoordinate", weight: float = 0.5) -> "VADCoordinate":
        """加权混合两个 VAD 坐标"""
        w = max(0.0, min(1.0, weight))
        return VADCoordinate(
            valence=self.valence * (1 - w) + other.valence * w,
            arousal=self.arousal * (1 - w) + other.arousal * w,
            dominance=self.dominance * (1 - w) + other.dominance * w,
            source="blend",
        )
    
    def scale(self, factor: float) -> "VADCoordinate":
        """缩放 VAD 值 (用于强度调节)"""
        return VADCoordinate(
            valence=max(-1.0, min(1.0, self.valence * factor)),
            arousal=max(-1.0, min(1.0, self.arousal * factor)),
            dominance=max(-1.0, min(1.0, self.dominance * factor)),
            source=self.source,
        )


class VADLexicon:
    """
    VAD 情感词典管理器
    
    支持多语言词典加载，精确匹配查询。
    
    用法:
        lexicon = VADLexicon()
        lexicon.load("data/vad/nrc_vad_en.json")
        lexicon.load("data/vad/vad_zh.json")
        
        vad = lexicon.get("happy")     # 英文
        vad = lexicon.get("开心")      # 中文
        vad = lexicon.get("unknown")   # None
    """
    
    def __init__(self, lexicon_paths: Optional[List[str]] = None):
        """
        Args:
            lexicon_paths: 词典文件路径列表 (可选)
        """
        self._entries: Dict[str, VADCoordinate] = {}
        self._sources: Dict[str, str] = {}  # word -> source file
        
        if lexicon_paths:
            for path in lexicon_paths:
                self.load(path)
    
    def load(self, path: str) -> int:
        """
        加载词典文件
        
        Args:
            path: JSON 文件路径
            
        Returns:
            加载的词条数量
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Lexicon file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        count = 0
        source_name = path.stem
        
        for word, values in data.items():
            if word.startswith("_"):  # Skip comments
                continue
            
            vad = VADCoordinate(
                valence=values.get("v", values.get("valence", 0.0)),
                arousal=values.get("a", values.get("arousal", 0.0)),
                dominance=values.get("d", values.get("dominance", 0.0)),
                source=values.get("source", source_name),
            )
            
            key = word.lower().strip()
            self._entries[key] = vad
            self._sources[key] = str(path)
            count += 1
        
        return count
    
    def get(self, word: str) -> Optional[VADCoordinate]:
        """
        获取单词的 VAD 坐标 (精确匹配)
        
        Args:
            word: 情感词
            
        Returns:
            VADCoordinate 或 None
        """
        key = word.lower().strip()
        return self._entries.get(key)
    
    def add(self, word: str, vad: VADCoordinate) -> None:
        """
        添加自定义词条
        
        Args:
            word: 情感词
            vad: VAD 坐标
        """
        key = word.lower().strip()
        self._entries[key] = vad
        self._sources[key] = "runtime"
    
    def contains(self, word: str) -> bool:
        """检查词典是否包含该词"""
        return word.lower().strip() in self._entries
    
    def sample(self, n: int = 1) -> List[Tuple[str, VADCoordinate]]:
        """随机采样词条 (用于数据增强)"""
        import random
        items = list(self._entries.items())
        return random.sample(items, min(n, len(items)))
    
    def all_words(self) -> List[str]:
        """返回所有词条"""
        return list(self._entries.keys())
    
    def __len__(self) -> int:
        return len(self._entries)
    
    def __repr__(self) -> str:
        return f"VADLexicon(entries={len(self._entries)})"


# ============================================================================
# 全局默认词典
# ============================================================================

_DEFAULT_LEXICON: Optional[VADLexicon] = None


def get_default_lexicon() -> VADLexicon:
    """获取默认全局词典 (懒加载)"""
    global _DEFAULT_LEXICON
    
    if _DEFAULT_LEXICON is None:
        _DEFAULT_LEXICON = VADLexicon()
        
        # 尝试加载默认词典文件
        base_path = Path(__file__).parent.parent.parent.parent / "data" / "vad"
        
        for filename in ["nrc_vad_en.json", "vad_zh.json", "custom_emotions.json"]:
            filepath = base_path / filename
            if filepath.exists():
                try:
                    _DEFAULT_LEXICON.load(str(filepath))
                except Exception as e:
                    print(f"Warning: Failed to load {filepath}: {e}")
    
    return _DEFAULT_LEXICON


def get_vad(word: str) -> Optional[VADCoordinate]:
    """便捷函数: 从默认词典获取 VAD"""
    return get_default_lexicon().get(word)
