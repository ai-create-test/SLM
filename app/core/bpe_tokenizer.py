"""
BPE Tokenizer Module - 基于tiktoken的工业级BPE分词器实现

设计原则：
1. 优先使用工业级实现 - 基于OpenAI的tiktoken库
2. 保持接口统一 - 继承BaseTokenizer，实现标准接口
3. 问题导向扩展 - 混入TokenAttentionMixin支持注意力权重

tiktoken优势：
- OpenAI生产环境使用的分词器
- 比HuggingFace tokenizers更快（Rust实现）
- 支持GPT-3.5/GPT-4的官方编码
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from collections import Counter

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    import regex as re
except ImportError:
    import re

from .tokenizer_base import (
    BaseTokenizer, 
    TokenizerResult, 
    PaddingStrategy,
    TruncationStrategy,
)
from .tokenizer_attention import TokenAttentionMixin, AttentionConfig


class BPETokenizer(BaseTokenizer, TokenAttentionMixin):
    """
    BPE分词器实现
    
    支持两种模式：
    1. 使用tiktoken预训练编码（推荐，工业级）
    2. 自定义BPE训练（实验性）
    
    使用示例：
        # 使用GPT-4的编码
        tokenizer = BPETokenizer(encoding_name="cl100k_base")
        result = tokenizer.encode("Hello, world!", query="greeting")
        
        # 自定义词表
        tokenizer = BPETokenizer(vocab_file="path/to/vocab.json")
    """
    
    # 预定义的tiktoken编码
    SUPPORTED_ENCODINGS = {
        "cl100k_base": "GPT-4, GPT-3.5-turbo, text-embedding-ada-002",
        "p50k_base": "Codex, text-davinci-002/003",
        "r50k_base": "GPT-3 (davinci)",
        "o200k_base": "GPT-4o",
    }
    
    def __init__(
        self,
        encoding_name: Optional[str] = "cl100k_base",
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        attention_config: Optional[AttentionConfig] = None,
        **kwargs
    ):
        """
        初始化BPE分词器
        
        Args:
            encoding_name: tiktoken编码名称（优先使用）
            vocab_file: 自定义词表文件路径
            merges_file: 自定义合并规则文件路径
            attention_config: 注意力权重配置
        """
        # 初始化Mixin
        TokenAttentionMixin.__init__(self, attention_config=attention_config)
        BaseTokenizer.__init__(self, **kwargs)
        
        self._vocab: Dict[str, int] = {}
        self._inverse_vocab: Dict[int, str] = {}
        self._merges: List[Tuple[str, str]] = []
        self._tiktoken_encoder = None
        
        if TIKTOKEN_AVAILABLE and encoding_name and not vocab_file:
            # 使用tiktoken预训练编码
            self._init_tiktoken(encoding_name)
        elif vocab_file:
            # 使用自定义词表
            self._load_vocab(vocab_file)
            if merges_file:
                self._load_merges(merges_file)
        else:
            # 空初始化，需要后续训练或加载
            self._init_default_vocab()
    
    def _init_tiktoken(self, encoding_name: str) -> None:
        """初始化tiktoken编码器"""
        try:
            self._tiktoken_encoder = tiktoken.get_encoding(encoding_name)
            # tiktoken不直接暴露词表，我们创建一个代理
            self._encoding_name = encoding_name
        except Exception as e:
            raise ValueError(f"无法加载tiktoken编码 '{encoding_name}': {e}")
    
    def _init_default_vocab(self) -> None:
        """初始化默认词表（包含特殊token）"""
        special_tokens = [
            self.pad_token, self.unk_token, 
            self.bos_token, self.eos_token,
            self.SEP_TOKEN, self.CLS_TOKEN, self.MASK_TOKEN
        ]
        self._vocab = {token: i for i, token in enumerate(special_tokens)}
        self._inverse_vocab = {i: token for token, i in self._vocab.items()}
    
    def _load_vocab(self, vocab_file: str) -> None:
        """加载自定义词表"""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self._vocab = json.load(f)
        self._inverse_vocab = {v: k for k, v in self._vocab.items()}
    
    def _load_merges(self, merges_file: str) -> None:
        """加载BPE合并规则"""
        with open(merges_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) == 2:
                        self._merges.append((parts[0], parts[1]))
    
    def encode(
        self,
        text: str,
        query: Optional[str] = None,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: Union[bool, str, PaddingStrategy] = False,
        return_offsets: bool = False,
        **kwargs
    ) -> TokenizerResult:
        """
        将文本编码为token序列
        
        Args:
            text: 输入文本
            query: 问题文本（用于计算问题导向注意力权重）
            add_special_tokens: 是否添加特殊token
            max_length: 最大长度
            truncation: 是否截断
            padding: 填充策略
            return_offsets: 是否返回位置映射
        """
        if self._tiktoken_encoder is not None:
            # 使用tiktoken编码
            token_ids = self._tiktoken_encoder.encode(text)
            tokens = [self._tiktoken_encoder.decode([tid]) for tid in token_ids]
        else:
            # 使用自定义BPE编码
            tokens, token_ids = self._custom_encode(text)
        
        # 添加特殊token
        if add_special_tokens:
            tokens = [self.CLS_TOKEN] + tokens + [self.EOS_TOKEN]
            cls_id = self._vocab.get(self.CLS_TOKEN, 0)
            eos_id = self._vocab.get(self.EOS_TOKEN, 0)
            if self._tiktoken_encoder is not None:
                # tiktoken模式下使用负数ID标记特殊token，避免与正常token冲突
                cls_id = -1  # 特殊标记
                eos_id = -2  # 特殊标记
            token_ids = [cls_id] + token_ids + [eos_id]
        
        # 截断
        if truncation and max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            tokens = tokens[:max_length]
        
        # 创建attention_mask
        attention_mask = [1] * len(token_ids)
        
        # 填充
        if padding and max_length:
            pad_length = max_length - len(token_ids)
            if pad_length > 0:
                pad_id = self._vocab.get(self.pad_token, 0)
                token_ids.extend([pad_id] * pad_length)
                tokens.extend([self.pad_token] * pad_length)
                attention_mask.extend([0] * pad_length)
        
        # 计算问题导向注意力权重
        attention_weights = None
        if query:
            attention_weights = self.compute_attention_weights(tokens, query)
        
        # 位置映射（简化实现）
        offsets = None
        if return_offsets:
            offsets = self._compute_offsets(text, tokens)
        
        return TokenizerResult(
            token_ids=token_ids,
            tokens=tokens,
            attention_weights=attention_weights,
            attention_mask=attention_mask,
            offsets=offsets,
            metadata={"encoding": getattr(self, '_encoding_name', 'custom')},
        )
    
    def _custom_encode(self, text: str) -> Tuple[List[str], List[int]]:
        """
        自定义BPE编码（当不使用tiktoken时）
        
        实现标准BPE算法：
        1. 将文本拆分为字符
        2. 迭代应用合并规则
        3. 将token映射为ID
        """
        # 预分词：按空格和标点分割
        pre_tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
        
        all_tokens = []
        all_ids = []
        
        for word in pre_tokens:
            # 将词拆分为字符
            chars = list(word)
            if not chars:
                continue
            
            # 应用BPE合并
            tokens = self._apply_bpe(chars)
            
            for token in tokens:
                all_tokens.append(token)
                if token in self._vocab:
                    all_ids.append(self._vocab[token])
                else:
                    all_ids.append(self._vocab.get(self.unk_token, 0))
        
        return all_tokens, all_ids
    
    def _apply_bpe(self, tokens: List[str]) -> List[str]:
        """应用BPE合并规则"""
        if not self._merges:
            return tokens
        
        while len(tokens) > 1:
            # 找到优先级最高的合并对
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
            
            best_pair = None
            best_rank = float('inf')
            
            for pair in pairs:
                try:
                    rank = self._merges.index(pair)
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair
                except ValueError:
                    continue
            
            if best_pair is None:
                break
            
            # 执行合并
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return tokens
    
    def _compute_offsets(
        self, 
        text: str, 
        tokens: List[str]
    ) -> List[Tuple[int, int]]:
        """计算每个token在原文中的位置"""
        offsets = []
        current_pos = 0
        
        for token in tokens:
            # 跳过特殊token
            if token in self._special_tokens:
                offsets.append((-1, -1))
                continue
            
            # 查找token在剩余文本中的位置
            idx = text.find(token, current_pos)
            if idx != -1:
                offsets.append((idx, idx + len(token)))
                current_pos = idx + len(token)
            else:
                offsets.append((-1, -1))
        
        return offsets
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """将token序列解码为文本"""
        if self._tiktoken_encoder is not None:
            # 使用tiktoken解码
            if skip_special_tokens:
                # 过滤我们手动添加的特殊token（使用负数ID标记）
                token_ids = [tid for tid in token_ids if tid >= 0]
            return self._tiktoken_encoder.decode(token_ids)
        else:
            # 使用自定义解码
            tokens = []
            for tid in token_ids:
                token = self._inverse_vocab.get(tid, self.unk_token)
                if skip_special_tokens and token in self._special_tokens:
                    continue
                tokens.append(token)
            
            text = ''.join(tokens)
            
            if clean_up_tokenization_spaces:
                # 清理多余空格
                text = re.sub(r'\s+', ' ', text).strip()
            
            return text
    
    @property
    def vocab_size(self) -> int:
        """词表大小"""
        if self._tiktoken_encoder is not None:
            return self._tiktoken_encoder.n_vocab
        return len(self._vocab)
    
    @property
    def vocab(self) -> Dict[str, int]:
        """词表字典"""
        if self._tiktoken_encoder is not None:
            # tiktoken不直接暴露词表，返回空字典
            return {}
        return self._vocab.copy()
    
    def train(
        self,
        corpus: List[str],
        vocab_size: int,
        min_frequency: int = 2,
        show_progress: bool = True,
    ) -> None:
        """
        在语料上训练BPE分词器
        
        Args:
            corpus: 训练语料列表
            vocab_size: 目标词表大小
            min_frequency: 最小词频
            show_progress: 是否显示进度
        """
        # 初始化词表为所有字符
        char_freq: Counter = Counter()
        for text in corpus:
            char_freq.update(text)
        
        # 过滤低频字符
        self._vocab = {self.pad_token: 0, self.unk_token: 1, self.bos_token: 2, self.eos_token: 3}
        next_id = len(self._vocab)
        
        for char, freq in char_freq.items():
            if freq >= min_frequency:
                self._vocab[char] = next_id
                next_id += 1
        
        # BPE合并循环
        self._merges = []
        word_freqs = self._get_word_frequencies(corpus)
        
        while len(self._vocab) < vocab_size:
            # 统计相邻对频率
            pair_freqs = self._get_pair_frequencies(word_freqs)
            if not pair_freqs:
                break
            
            # 选择最频繁的对
            best_pair = max(pair_freqs, key=pair_freqs.get)
            
            # 合并
            self._merges.append(best_pair)
            new_token = best_pair[0] + best_pair[1]
            self._vocab[new_token] = next_id
            next_id += 1
            
            # 更新词频
            word_freqs = self._merge_pair(word_freqs, best_pair)
            
            if show_progress and len(self._merges) % 100 == 0:
                print(f"BPE merges: {len(self._merges)}, vocab size: {len(self._vocab)}")
        
        self._inverse_vocab = {v: k for k, v in self._vocab.items()}
    
    def _get_word_frequencies(self, corpus: List[str]) -> Dict[Tuple[str, ...], int]:
        """获取词频统计"""
        word_freqs: Dict[Tuple[str, ...], int] = {}
        for text in corpus:
            words = re.findall(r'\w+', text, re.UNICODE)
            for word in words:
                chars = tuple(word)
                word_freqs[chars] = word_freqs.get(chars, 0) + 1
        return word_freqs
    
    def _get_pair_frequencies(
        self, 
        word_freqs: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, str], int]:
        """统计相邻对频率"""
        pair_freqs: Dict[Tuple[str, str], int] = {}
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
        return pair_freqs
    
    def _merge_pair(
        self,
        word_freqs: Dict[Tuple[str, ...], int],
        pair: Tuple[str, str],
    ) -> Dict[Tuple[str, ...], int]:
        """合并一个对"""
        new_word_freqs: Dict[Tuple[str, ...], int] = {}
        merged = pair[0] + pair[1]
        
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        
        return new_word_freqs
    
    def save(self, directory: str) -> None:
        """保存分词器到目录"""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # 保存词表
        with open(path / 'vocab.json', 'w', encoding='utf-8') as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)
        
        # 保存合并规则
        with open(path / 'merges.txt', 'w', encoding='utf-8') as f:
            for pair in self._merges:
                f.write(f"{pair[0]} {pair[1]}\n")
        
        # 保存配置
        config = {
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
            'encoding_name': getattr(self, '_encoding_name', None),
        }
        with open(path / 'config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, directory: str) -> "BPETokenizer":
        """从目录加载分词器"""
        path = Path(directory)
        
        # 加载配置
        config_file = path / 'config.json'
        config = {}
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        
        # 如果有tiktoken编码名称，使用tiktoken
        if config.get('encoding_name'):
            return cls(encoding_name=config['encoding_name'])
        
        # 否则加载自定义词表
        vocab_file = path / 'vocab.json'
        merges_file = path / 'merges.txt'
        
        return cls(
            vocab_file=str(vocab_file) if vocab_file.exists() else None,
            merges_file=str(merges_file) if merges_file.exists() else None,
            **config
        )
    
    def __repr__(self) -> str:
        if self._tiktoken_encoder is not None:
            return f"BPETokenizer(encoding='{self._encoding_name}', vocab_size={self.vocab_size})"
        return f"BPETokenizer(custom, vocab_size={self.vocab_size})"
