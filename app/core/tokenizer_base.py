"""
Tokenizer Base Module - 分词器抽象基类与核心数据结构

设计原则：
1. 高度模块化 - 通过抽象基类定义统一接口
2. 可扩展性 - TokenizerResult支持携带注意力权重和元数据
3. 工业标准 - 接口设计参考HuggingFace Tokenizers
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum


class PaddingStrategy(Enum):
    """填充策略枚举"""
    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class TruncationStrategy(Enum):
    """截断策略枚举"""
    LONGEST_FIRST = "longest_first"
    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    DO_NOT_TRUNCATE = "do_not_truncate"


@dataclass
class TokenizerResult:
    """
    分词结果容器 - 支持扩展的结构化输出
    
    设计考量：
    - token_ids: 标准输出，用于模型输入
    - tokens: 原始token字符串，用于调试和可视化
    - attention_weights: 问题导向检索的关键 - 每个token的重要性权重
    - attention_mask: 标准注意力掩码（区分padding）
    - metadata: 可扩展字段，用于情感标注、位置信息等
    """
    token_ids: List[int]
    tokens: List[str]
    attention_weights: Optional[List[float]] = None  # 问题导向权重
    attention_mask: Optional[List[int]] = None       # 标准attention mask
    token_type_ids: Optional[List[int]] = None       # 用于区分句子对
    offsets: Optional[List[Tuple[int, int]]] = None  # 原文位置映射
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.token_ids)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，便于序列化"""
        result = {
            "token_ids": self.token_ids,
            "tokens": self.tokens,
        }
        if self.attention_weights is not None:
            result["attention_weights"] = self.attention_weights
        if self.attention_mask is not None:
            result["attention_mask"] = self.attention_mask
        if self.token_type_ids is not None:
            result["token_type_ids"] = self.token_type_ids
        if self.offsets is not None:
            result["offsets"] = self.offsets
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class BatchTokenizerResult:
    """
    批量分词结果容器
    
    用于处理多个文本的批量编码结果
    """
    token_ids: List[List[int]]
    tokens: List[List[str]]
    attention_weights: Optional[List[List[float]]] = None
    attention_mask: Optional[List[List[int]]] = None
    token_type_ids: Optional[List[List[int]]] = None
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.token_ids)
    
    def __getitem__(self, index: int) -> TokenizerResult:
        """支持索引访问单个结果"""
        return TokenizerResult(
            token_ids=self.token_ids[index],
            tokens=self.tokens[index],
            attention_weights=self.attention_weights[index] if self.attention_weights else None,
            attention_mask=self.attention_mask[index] if self.attention_mask else None,
            token_type_ids=self.token_type_ids[index] if self.token_type_ids else None,
            metadata=self.metadata[index] if self.metadata else {},
        )


class BaseTokenizer(ABC):
    """
    分词器抽象基类
    
    定义统一接口，所有具体分词器实现必须继承此类。
    接口设计参考HuggingFace Tokenizers工业标准。
    
    扩展说明：
    - 实现新分词器只需继承此类并实现抽象方法
    - encode方法支持传入query参数，用于问题导向权重计算
    """
    
    # 特殊token定义
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    SEP_TOKEN = "[SEP]"
    CLS_TOKEN = "[CLS]"
    MASK_TOKEN = "[MASK]"
    
    def __init__(
        self,
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        bos_token: str = "[BOS]",
        eos_token: str = "[EOS]",
        **kwargs
    ):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self._special_tokens = {pad_token, unk_token, bos_token, eos_token}
    
    @abstractmethod
    def encode(
        self, 
        text: str,
        query: Optional[str] = None,  # 问题导向：传入问题以计算注意力权重
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
            query: 可选的问题文本，用于计算问题导向的注意力权重
            add_special_tokens: 是否添加特殊token（如[CLS], [SEP]）
            max_length: 最大长度限制
            truncation: 是否截断
            padding: 填充策略
            return_offsets: 是否返回原文位置映射
            
        Returns:
            TokenizerResult: 包含token_ids、tokens和可选注意力权重的结果
        """
        pass
    
    @abstractmethod
    def decode(
        self, 
        token_ids: List[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """
        将token序列解码为文本
        
        Args:
            token_ids: token ID序列
            skip_special_tokens: 是否跳过特殊token
            clean_up_tokenization_spaces: 是否清理分词产生的多余空格
            
        Returns:
            解码后的文本字符串
        """
        pass
    
    def encode_batch(
        self,
        texts: List[str],
        queries: Optional[List[str]] = None,
        **kwargs
    ) -> BatchTokenizerResult:
        """
        批量编码多个文本
        
        Args:
            texts: 输入文本列表
            queries: 可选的问题列表，与texts一一对应
            **kwargs: 传递给encode的其他参数
            
        Returns:
            BatchTokenizerResult: 批量编码结果
        """
        results = []
        for i, text in enumerate(texts):
            query = queries[i] if queries and i < len(queries) else None
            results.append(self.encode(text, query=query, **kwargs))
        
        return BatchTokenizerResult(
            token_ids=[r.token_ids for r in results],
            tokens=[r.tokens for r in results],
            attention_weights=[r.attention_weights for r in results] if results[0].attention_weights else None,
            attention_mask=[r.attention_mask for r in results] if results[0].attention_mask else None,
            token_type_ids=[r.token_type_ids for r in results] if results[0].token_type_ids else None,
            metadata=[r.metadata for r in results],
        )
    
    def decode_batch(
        self, 
        token_ids_list: List[List[int]],
        **kwargs
    ) -> List[str]:
        """批量解码多个token序列"""
        return [self.decode(token_ids, **kwargs) for token_ids in token_ids_list]
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """词表大小"""
        pass
    
    @property
    @abstractmethod
    def vocab(self) -> Dict[str, int]:
        """词表字典：token -> id"""
        pass
    
    def get_vocab(self) -> Dict[str, int]:
        """获取词表（兼容HuggingFace接口）"""
        return self.vocab
    
    def token_to_id(self, token: str) -> Optional[int]:
        """将token转换为id"""
        return self.vocab.get(token)
    
    def id_to_token(self, id: int) -> Optional[str]:
        """将id转换为token"""
        inverse_vocab = {v: k for k, v in self.vocab.items()}
        return inverse_vocab.get(id)
    
    def add_special_tokens(self, tokens: List[str]) -> int:
        """
        添加特殊token到词表
        
        Returns:
            新增的token数量
        """
        raise NotImplementedError("子类需要实现此方法")
    
    def save(self, path: str) -> None:
        """保存分词器到文件"""
        raise NotImplementedError("子类需要实现此方法")
    
    @classmethod
    def load(cls, path: str) -> "BaseTokenizer":
        """从文件加载分词器"""
        raise NotImplementedError("子类需要实现此方法")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(vocab_size={self.vocab_size})"
