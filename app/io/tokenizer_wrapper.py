"""
Tokenizer Wrapper - HuggingFace Tokenizer 包装器

提供统一的 Tokenizer 接口，支持多种预训练模型。

支持的 Tokenizer:
- BERT / RoBERTa / DistilBERT
- GPT2 / GPT-Neo
- T5 / FLAN-T5
- 自定义 tiktoken (已有 BPETokenizer)

设计原则:
1. 与 HuggingFace transformers 库兼容
2. 提供简单的 encode/decode 接口
3. 自动处理 padding 和 attention mask
4. 支持批量处理
"""

from typing import List, Dict, Any, Optional, Union, Literal
from dataclasses import dataclass
import torch

# 尝试导入 transformers，如果不可用则使用备用方案
try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    PreTrainedTokenizerBase = object


@dataclass
class TokenizerOutput:
    """
    Tokenizer 输出
    
    与 HuggingFace 的 BatchEncoding 兼容但更简单。
    """
    input_ids: torch.Tensor           # Token IDs [batch, seq_len]
    attention_mask: torch.Tensor      # Attention mask [batch, seq_len]
    token_type_ids: Optional[torch.Tensor] = None  # Segment IDs (BERT)
    
    def to(self, device: torch.device) -> "TokenizerOutput":
        """移动到指定设备"""
        return TokenizerOutput(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            token_type_ids=self.token_type_ids.to(device) if self.token_type_ids is not None else None,
        )
    
    def __len__(self) -> int:
        return self.input_ids.shape[0]
    
    @property
    def seq_len(self) -> int:
        return self.input_ids.shape[1]


class TokenizerWrapper:
    """
    HuggingFace Tokenizer 包装器
    
    提供统一的接口，简化 tokenization 流程。
    
    使用示例:
        # 创建
        tokenizer = TokenizerWrapper.from_pretrained("bert-base-uncased")
        
        # 单文本
        output = tokenizer.encode("Hello world")
        print(output.input_ids)  # tensor([[101, 7592, 2088, 102]])
        
        # 批量
        output = tokenizer.encode_batch(["Hello", "World"])
        print(output.input_ids.shape)  # [2, max_len]
        
        # 解码
        text = tokenizer.decode(output.input_ids[0])
    """
    
    # 预设模型映射
    PRESET_MODELS: Dict[str, str] = {
        "bert": "bert-base-uncased",
        "bert-chinese": "bert-base-chinese",
        "roberta": "roberta-base",
        "distilbert": "distilbert-base-uncased",
        "gpt2": "gpt2",
        "t5": "t5-base",
    }
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
        padding: Union[bool, str] = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt",
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer 实例
            max_length: 最大序列长度
            padding: 填充策略 ('max_length', 'longest', True, False)
            truncation: 是否截断
            return_tensors: 返回张量类型 ('pt' for PyTorch)
        """
        self._tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        max_length: int = 256,
        **kwargs,
    ) -> "TokenizerWrapper":
        """
        从预训练模型加载 Tokenizer
        
        Args:
            model_name_or_path: 模型名称或路径
                - HuggingFace 模型名称: "bert-base-uncased"
                - 本地路径: "./my_tokenizer"
                - 预设名称: "bert", "roberta", "gpt2"
            max_length: 最大序列长度
            **kwargs: 传递给 AutoTokenizer 的额外参数
            
        Returns:
            TokenizerWrapper 实例
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers 库未安装。请运行: pip install transformers"
            )
        
        # 检查是否是预设
        if model_name_or_path in cls.PRESET_MODELS:
            model_name_or_path = cls.PRESET_MODELS[model_name_or_path]
        
        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        
        # 确保有 pad_token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        return cls(tokenizer, max_length=max_length)
    
    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
    ) -> TokenizerOutput:
        """
        编码单个文本
        
        Args:
            text: 输入文本
            max_length: 覆盖默认最大长度
            
        Returns:
            TokenizerOutput
        """
        return self.encode_batch([text], max_length=max_length)
    
    def encode_batch(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
    ) -> TokenizerOutput:
        """
        批量编码文本
        
        Args:
            texts: 文本列表
            max_length: 覆盖默认最大长度
            
        Returns:
            TokenizerOutput
        """
        max_len = max_length or self.max_length
        
        # 调用 HuggingFace tokenizer
        encoded = self._tokenizer(
            texts,
            max_length=max_len,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors,
        )
        
        return TokenizerOutput(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            token_type_ids=encoded.get("token_type_ids"),
        )
    
    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> str:
        """
        解码单个序列
        
        Args:
            token_ids: Token IDs [seq_len] 或 [1, seq_len]
            skip_special_tokens: 跳过特殊 token
            
        Returns:
            解码后的文本
        """
        if token_ids.dim() == 2:
            token_ids = token_ids[0]
        
        return self._tokenizer.decode(
            token_ids.tolist(),
            skip_special_tokens=skip_special_tokens,
        )
    
    def decode_batch(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        批量解码
        
        Args:
            token_ids: Token IDs [batch, seq_len]
            skip_special_tokens: 跳过特殊 token
            
        Returns:
            解码后的文本列表
        """
        return self._tokenizer.batch_decode(
            token_ids.tolist(),
            skip_special_tokens=skip_special_tokens,
        )
    
    @property
    def vocab_size(self) -> int:
        """词表大小"""
        return len(self._tokenizer)
    
    @property
    def pad_token_id(self) -> int:
        """Padding token ID"""
        return self._tokenizer.pad_token_id
    
    @property
    def bos_token_id(self) -> Optional[int]:
        """Beginning of sequence token ID"""
        return self._tokenizer.bos_token_id
    
    @property
    def eos_token_id(self) -> Optional[int]:
        """End of sequence token ID"""
        return self._tokenizer.eos_token_id
    
    @property
    def cls_token_id(self) -> Optional[int]:
        """CLS token ID (BERT-style)"""
        return self._tokenizer.cls_token_id
    
    @property
    def sep_token_id(self) -> Optional[int]:
        """SEP token ID (BERT-style)"""
        return self._tokenizer.sep_token_id
    
    @property
    def model_max_length(self) -> int:
        """模型支持的最大长度"""
        return self._tokenizer.model_max_length
    
    def get_special_tokens(self) -> Dict[str, Any]:
        """获取所有特殊 token"""
        return {
            "pad_token": self._tokenizer.pad_token,
            "bos_token": self._tokenizer.bos_token,
            "eos_token": self._tokenizer.eos_token,
            "cls_token": getattr(self._tokenizer, "cls_token", None),
            "sep_token": getattr(self._tokenizer, "sep_token", None),
            "unk_token": self._tokenizer.unk_token,
            "mask_token": getattr(self._tokenizer, "mask_token", None),
        }
    
    def __repr__(self) -> str:
        return f"TokenizerWrapper(vocab_size={self.vocab_size}, max_length={self.max_length})"


class FallbackTokenizer:
    """
    备用 Tokenizer (当 transformers 不可用时)
    
    使用简单的字符级或基于 tiktoken 的分词。
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        max_length: int = 256,
        use_tiktoken: bool = True,
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.use_tiktoken = use_tiktoken
        
        self._pad_token_id = 0
        self._unk_token_id = 1
        self._bos_token_id = 2
        self._eos_token_id = 3
        
        if use_tiktoken:
            try:
                import tiktoken
                self._enc = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                self._enc = None
                self.use_tiktoken = False
        else:
            self._enc = None
    
    def encode(self, text: str, max_length: Optional[int] = None) -> TokenizerOutput:
        return self.encode_batch([text], max_length)
    
    def encode_batch(self, texts: List[str], max_length: Optional[int] = None) -> TokenizerOutput:
        max_len = max_length or self.max_length
        
        all_ids = []
        for text in texts:
            if self.use_tiktoken and self._enc is not None:
                ids = self._enc.encode(text)
                # 将 tiktoken ID 映射到 vocab_size 范围内
                ids = [i % (self.vocab_size - 4) + 4 for i in ids]  # 保留 0-3 给特殊 token
            else:
                # 简单的 UTF-8 字节编码
                ids = [ord(c) % (self.vocab_size - 4) + 4 for c in text]
            
            # 截断
            if len(ids) > max_len - 2:  # 留位置给 BOS/EOS
                ids = ids[:max_len - 2]
            
            # 添加特殊 token
            ids = [self._bos_token_id] + ids + [self._eos_token_id]
            all_ids.append(ids)
        
        # Padding
        max_actual_len = max(len(ids) for ids in all_ids)
        padded_ids = []
        attention_masks = []
        
        for ids in all_ids:
            pad_len = max_actual_len - len(ids)
            padded_ids.append(ids + [self._pad_token_id] * pad_len)
            attention_masks.append([1] * len(ids) + [0] * pad_len)
        
        return TokenizerOutput(
            input_ids=torch.tensor(padded_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_masks, dtype=torch.long),
        )
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        if token_ids.dim() == 2:
            token_ids = token_ids[0]
        
        ids = token_ids.tolist()
        
        if skip_special_tokens:
            special = {self._pad_token_id, self._bos_token_id, self._eos_token_id, self._unk_token_id}
            ids = [i for i in ids if i not in special]
        
        if self.use_tiktoken and self._enc is not None:
            try:
                return self._enc.decode(ids)
            except Exception:
                pass
        
        # 简单字符解码
        return ''.join(chr(i) for i in ids if 32 <= i < 127)
    
    def decode_batch(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        return [self.decode(ids, skip_special_tokens) for ids in token_ids]
    
    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id
    
    @property
    def bos_token_id(self) -> int:
        return self._bos_token_id
    
    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id


def get_tokenizer(
    model_name: str = "bert-base-uncased",
    max_length: int = 256,
    fallback: bool = True,
) -> Union[TokenizerWrapper, FallbackTokenizer]:
    """
    获取 Tokenizer 的统一入口
    
    Args:
        model_name: 模型名称
        max_length: 最大长度
        fallback: 是否在加载失败时使用备用方案
        
    Returns:
        Tokenizer 实例
    """
    if HAS_TRANSFORMERS:
        try:
            return TokenizerWrapper.from_pretrained(model_name, max_length=max_length)
        except Exception as e:
            if fallback:
                print(f"Warning: Failed to load {model_name}, using fallback tokenizer. Error: {e}")
                return FallbackTokenizer(max_length=max_length)
            raise
    else:
        if fallback:
            print("Warning: transformers not installed, using fallback tokenizer")
            return FallbackTokenizer(max_length=max_length)
        raise ImportError("transformers 库未安装")
