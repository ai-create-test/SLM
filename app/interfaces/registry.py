"""
Registry - 模块注册表

设计原则:
1. 动态注册/发现模块
2. 按类型分类管理
3. 支持自定义模块注入
4. 工厂模式创建实例
"""

from typing import Dict, Type, Optional, Any, Callable, List
from .base_module import BaseModule


class Registry:
    """
    模块注册表
    
    管理所有可用模块的注册和创建。
    
    使用示例:
        # 注册模块
        @Registry.register("encoder", "vqvae")
        class VQVAEEncoder(BaseModule):
            ...
        
        # 或者手动注册
        Registry.register_module("encoder", "custom", MyEncoder)
        
        # 创建实例
        encoder = Registry.create("encoder", "vqvae", config=my_config)
        
        # 列出可用模块
        available = Registry.list_modules("encoder")
    """
    
    # 按模块类型组织的注册表
    _registry: Dict[str, Dict[str, Type[BaseModule]]] = {}
    
    # 模块类型描述
    _type_descriptions: Dict[str, str] = {
        "encoder": "输入编码器 (Text/Paragraph → Latent)",
        "decoder": "输出解码器 (Latent → Text)",
        "memory": "记忆系统",
        "brain": "核心推理模型",
        "modulation": "控制调制层",
        "reflection": "自我回溯模块",
        "search": "外部搜索模块",
    }
    
    @classmethod
    def register(
        cls, 
        module_type: str, 
        name: str
    ) -> Callable[[Type[BaseModule]], Type[BaseModule]]:
        """
        装饰器：注册模块
        
        Args:
            module_type: 模块类型 (encoder, decoder, memory, brain, etc.)
            name: 模块名称 (vqvae, mamba, etc.)
        
        Example:
            @Registry.register("encoder", "vqvae")
            class VQVAEEncoder(BaseModule):
                ...
        """
        def decorator(module_cls: Type[BaseModule]) -> Type[BaseModule]:
            cls.register_module(module_type, name, module_cls)
            return module_cls
        return decorator
    
    @classmethod
    def register_module(
        cls,
        module_type: str,
        name: str,
        module_cls: Type[BaseModule],
    ) -> None:
        """
        手动注册模块
        
        Args:
            module_type: 模块类型
            name: 模块名称
            module_cls: 模块类
        """
        if module_type not in cls._registry:
            cls._registry[module_type] = {}
        
        if name in cls._registry[module_type]:
            raise ValueError(
                f"Module '{name}' already registered for type '{module_type}'. "
                f"Use a different name or unregister first."
            )
        
        cls._registry[module_type][name] = module_cls
    
    @classmethod
    def unregister(cls, module_type: str, name: str) -> None:
        """取消注册模块"""
        if module_type in cls._registry and name in cls._registry[module_type]:
            del cls._registry[module_type][name]
    
    @classmethod
    def get_module_class(
        cls,
        module_type: str,
        name: str,
    ) -> Type[BaseModule]:
        """
        获取注册的模块类
        
        Args:
            module_type: 模块类型
            name: 模块名称
            
        Returns:
            模块类
            
        Raises:
            KeyError: 如果模块未注册
        """
        if module_type not in cls._registry:
            raise KeyError(f"Unknown module type: {module_type}")
        
        if name not in cls._registry[module_type]:
            available = list(cls._registry[module_type].keys())
            raise KeyError(
                f"Module '{name}' not found for type '{module_type}'. "
                f"Available: {available}"
            )
        
        return cls._registry[module_type][name]
    
    @classmethod
    def create(
        cls,
        module_type: str,
        name: str,
        config: Optional[Any] = None,
        **kwargs,
    ) -> BaseModule:
        """
        创建模块实例
        
        Args:
            module_type: 模块类型
            name: 模块名称
            config: 配置对象
            **kwargs: 额外参数
            
        Returns:
            模块实例
        """
        module_cls = cls.get_module_class(module_type, name)
        
        if config is not None:
            return module_cls.from_config(config, **kwargs)
        else:
            return module_cls(**kwargs)
    
    @classmethod
    def list_modules(cls, module_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        列出已注册的模块
        
        Args:
            module_type: 可选，限定模块类型
            
        Returns:
            {module_type: [module_names]} 字典
        """
        if module_type is not None:
            if module_type not in cls._registry:
                return {module_type: []}
            return {module_type: list(cls._registry[module_type].keys())}
        
        return {
            mtype: list(modules.keys())
            for mtype, modules in cls._registry.items()
        }
    
    @classmethod
    def list_types(cls) -> List[str]:
        """列出所有模块类型"""
        return list(cls._registry.keys())
    
    @classmethod
    def get_type_description(cls, module_type: str) -> str:
        """获取模块类型描述"""
        return cls._type_descriptions.get(module_type, "Unknown module type")
    
    @classmethod
    def clear(cls) -> None:
        """清空注册表（主要用于测试）"""
        cls._registry.clear()
    
    @classmethod
    def info(cls) -> str:
        """返回注册表的可读信息"""
        lines = ["=== Module Registry ==="]
        
        for mtype, modules in cls._registry.items():
            desc = cls.get_type_description(mtype)
            lines.append(f"\n📦 {mtype} - {desc}")
            for name, module_cls in modules.items():
                lines.append(f"   └── {name}: {module_cls.__name__}")
        
        if not cls._registry:
            lines.append("\n(No modules registered)")
        
        return "\n".join(lines)
