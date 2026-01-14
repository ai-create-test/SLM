"""
Tokenizer Factory Module - 分词器工厂

设计模式：
1. 工厂模式 - 统一创建入口
2. 注册机制 - 支持动态扩展
3. 配置驱动 - 支持从配置文件创建

使用示例：
    # 注册自定义分词器
    TokenizerFactory.register("my_tokenizer", MyTokenizer)
    
    # 创建分词器
    tokenizer = TokenizerFactory.create("bpe", encoding_name="cl100k_base")
    
    # 从配置创建
    tokenizer = TokenizerFactory.from_config("path/to/config.json")
"""

import json
from typing import Dict, Type, Optional, Any
from pathlib import Path

from .tokenizer_base import BaseTokenizer
from .bpe_tokenizer import BPETokenizer


class TokenizerFactory:
    """
    分词器工厂类
    
    功能：
    1. 管理已注册的分词器类型
    2. 提供统一的创建接口
    3. 支持从配置文件创建
    4. 支持运行时动态注册新类型
    """
    
    # 已注册的分词器类型
    _registry: Dict[str, Type[BaseTokenizer]] = {}
    
    # 预注册的分词器
    _builtin_tokenizers = {
        "bpe": BPETokenizer,
        "BPE": BPETokenizer,
        "tiktoken": BPETokenizer,
    }
    
    @classmethod
    def register(cls, name: str, tokenizer_class: Type[BaseTokenizer]) -> None:
        """
        注册新的分词器类型
        
        Args:
            name: 分词器名称
            tokenizer_class: 分词器类（必须继承BaseTokenizer）
            
        Raises:
            TypeError: 如果tokenizer_class不是BaseTokenizer的子类
        """
        if not issubclass(tokenizer_class, BaseTokenizer):
            raise TypeError(
                f"tokenizer_class必须是BaseTokenizer的子类，"
                f"但得到了 {tokenizer_class}"
            )
        cls._registry[name] = tokenizer_class
    
    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        取消注册分词器类型
        
        Args:
            name: 分词器名称
            
        Returns:
            是否成功取消注册
        """
        if name in cls._registry:
            del cls._registry[name]
            return True
        return False
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseTokenizer:
        """
        创建分词器实例
        
        Args:
            name: 分词器名称
            **kwargs: 传递给分词器构造函数的参数
            
        Returns:
            分词器实例
            
        Raises:
            ValueError: 如果分词器类型未注册
        """
        # 先查找已注册的
        if name in cls._registry:
            return cls._registry[name](**kwargs)
        
        # 再查找内置的
        if name in cls._builtin_tokenizers:
            return cls._builtin_tokenizers[name](**kwargs)
        
        # 支持快捷方式
        shortcuts = {
            "gpt4": ("bpe", {"encoding_name": "cl100k_base"}),
            "gpt4o": ("bpe", {"encoding_name": "o200k_base"}),
            "gpt3": ("bpe", {"encoding_name": "r50k_base"}),
            "codex": ("bpe", {"encoding_name": "p50k_base"}),
        }
        
        if name.lower() in shortcuts:
            base_name, default_kwargs = shortcuts[name.lower()]
            merged_kwargs = {**default_kwargs, **kwargs}
            return cls._builtin_tokenizers[base_name](**merged_kwargs)
        
        raise ValueError(
            f"未知的分词器类型: '{name}'。\n"
            f"已注册的类型: {list(cls._registry.keys())}\n"
            f"内置类型: {list(cls._builtin_tokenizers.keys())}\n"
            f"快捷方式: gpt4, gpt4o, gpt3, codex"
        )
    
    @classmethod
    def from_config(cls, config_path: str) -> BaseTokenizer:
        """
        从配置文件创建分词器
        
        配置文件格式（JSON）：
        {
            "type": "bpe",
            "params": {
                "encoding_name": "cl100k_base"
            }
        }
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            分词器实例
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        tokenizer_type = config.get("type", "bpe")
        params = config.get("params", {})
        
        return cls.create(tokenizer_type, **params)
    
    @classmethod
    def from_pretrained(cls, path: str) -> BaseTokenizer:
        """
        加载预训练/保存的分词器
        
        Args:
            path: 分词器保存路径
            
        Returns:
            分词器实例
        """
        path = Path(path)
        
        # 检查是否有配置文件
        config_file = path / "config.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 如果是tiktoken编码
            if config.get("encoding_name"):
                return BPETokenizer(encoding_name=config["encoding_name"])
        
        # 否则尝试加载自定义分词器
        return BPETokenizer.load(str(path))
    
    @classmethod
    def list_available(cls) -> Dict[str, str]:
        """
        列出所有可用的分词器类型
        
        Returns:
            分词器名称到描述的映射
        """
        available = {}
        
        # 内置类型
        for name in cls._builtin_tokenizers:
            available[name] = f"内置 - {cls._builtin_tokenizers[name].__name__}"
        
        # 已注册类型
        for name in cls._registry:
            available[name] = f"已注册 - {cls._registry[name].__name__}"
        
        # 快捷方式
        shortcuts = {
            "gpt4": "快捷方式 - GPT-4 (cl100k_base)",
            "gpt4o": "快捷方式 - GPT-4o (o200k_base)",
            "gpt3": "快捷方式 - GPT-3 (r50k_base)",
            "codex": "快捷方式 - Codex (p50k_base)",
        }
        available.update(shortcuts)
        
        return available
    
    @classmethod
    def get_tokenizer_class(cls, name: str) -> Optional[Type[BaseTokenizer]]:
        """
        获取分词器类（不创建实例）
        
        Args:
            name: 分词器名称
            
        Returns:
            分词器类，如果不存在返回None
        """
        if name in cls._registry:
            return cls._registry[name]
        if name in cls._builtin_tokenizers:
            return cls._builtin_tokenizers[name]
        return None


# 便捷函数
def get_tokenizer(name: str = "gpt4", **kwargs) -> BaseTokenizer:
    """
    获取分词器的便捷函数
    
    Args:
        name: 分词器名称或快捷方式
        **kwargs: 传递给分词器的参数
        
    Returns:
        分词器实例
        
    示例：
        tokenizer = get_tokenizer("gpt4")
        tokenizer = get_tokenizer("bpe", encoding_name="cl100k_base")
    """
    return TokenizerFactory.create(name, **kwargs)
