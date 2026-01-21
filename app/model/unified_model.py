"""
Unified Model - AMHVQ+ 统一模型封装

Phase 12: 模型集成

整合三通道编解码器，提供标准化接口:
- UnifiedEncoder + UnifiedDecoder
- 工厂方法创建
- Checkpoint 兼容
"""

from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import torch
import torch.nn as nn

from ..interfaces.base_module import HierarchicalLatent
from ..interfaces.unified_latent import UnifiedLatent
from ..interfaces.config import ModelConfig, Config
from ..interfaces.registry import Registry

from ..io.unified_encoder import UnifiedEncoder, create_unified_encoder
from ..io.unified_decoder import UnifiedDecoder, create_unified_decoder
from ..brain.hierarchical_dynamics import HierarchicalDynamics, UnifiedDynamics
from ..memory.graph_memory import GraphMemory

from .model_utils import (
    save_safetensors,
    load_safetensors,
    save_config,
    load_config,
    safe_load_state_dict,
)


@dataclass
class UnifiedModelMetadata:
    """统一模型元数据"""
    model_type: str = "UnifiedNeuralFlow"
    version: str = "3.0.0"
    architecture: Dict[str, Any] = field(default_factory=dict)
    three_channel: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "version": self.version,
            "architecture": self.architecture,
            "three_channel": self.three_channel,
            "training": self.training,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UnifiedModelMetadata":
        return cls(**d)


@Registry.register("model", "unified")
class UnifiedNeuralFlowModel(nn.Module):
    """
    AMHVQ+ 统一模型
    
    整合三通道编解码器:
    - UnifiedEncoder: 语义 + 结构 + 符号编码
    - UnifiedDecoder: 结构引导解码
    - UnifiedDynamics: 动力学预测
    - GraphMemory: 结构存储
    
    使用示例:
        # 从配置创建
        model = UnifiedNeuralFlowModel.from_config(config)
        
        # 编码
        unified_latent = model.encode("def hello(): pass", scene="coding")
        
        # 解码
        text = model.decode(unified_latent)
        
        # 端到端生成
        output = model.generate("def hello():", max_length=100)
        
        # 保存/加载
        model.save_pretrained("./model_dir")
        model = UnifiedNeuralFlowModel.from_pretrained("./model_dir")
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        use_three_channel: bool = True,
        **kwargs,
    ):
        """
        Args:
            config: 配置对象
            use_three_channel: 是否启用三通道
            **kwargs: 额外参数
        """
        super().__init__()
        
        # 配置
        if config is None:
            config = Config()
        self.config = config
        self.model_config = config.model
        self.use_three_channel = use_three_channel
        
        # 元数据
        self.metadata = UnifiedModelMetadata(
            architecture={
                "d_model": self.model_config.d_model,
                "d_latent": self.model_config.d_latent,
                "max_chunks": getattr(self.model_config, 'max_chunks', 8),
            },
            three_channel={
                "enabled": use_three_channel,
                "semantic": True,
                "structure": use_three_channel,
                "symbols": use_three_channel,
            },
        )
        
        # GraphMemory (共享)
        self.graph_memory = GraphMemory(
            d_node=getattr(self.model_config, 'd_latent', 512)
        )
        
        # 编码器
        self.encoder = create_unified_encoder(
            self.model_config,
            use_three_channel=use_three_channel,
            **kwargs,
        )
        
        # 解码器
        self.decoder = create_unified_decoder(
            self.model_config,
            use_three_channel=use_three_channel,
            graph_memory=self.graph_memory,
            **kwargs,
        )
        
        # 动力学模型 (可选)
        if hasattr(self.model_config, 'use_dynamics') and self.model_config.use_dynamics:
            self.dynamics = UnifiedDynamics(
                d_latent=self.model_config.d_latent,
                d_model=self.model_config.d_model,
                num_chunks=getattr(self.model_config, 'max_chunks', 8),
                use_three_channel=use_three_channel,
            )
        else:
            self.dynamics = None
    
    def encode(
        self,
        text: Union[str, List[str]],
        scene: Optional[str] = None,
    ) -> UnifiedLatent:
        """
        编码文本到 UnifiedLatent
        
        Args:
            text: 输入文本
            scene: 场景 (None 则自动检测)
            
        Returns:
            UnifiedLatent
        """
        output = self.encoder(text, scene=scene)
        return output.unified_latent
    
    def decode(
        self,
        latent: Union[UnifiedLatent, HierarchicalLatent, torch.Tensor],
        **kwargs,
    ) -> List[str]:
        """
        解码潜向量到文本
        
        Args:
            latent: 潜向量
            **kwargs: 额外参数
            
        Returns:
            文本列表
        """
        output = self.decoder(latent, **kwargs)
        return output.text
    
    def generate(
        self,
        text: str,
        scene: Optional[str] = None,
        max_length: int = 256,
        temperature: float = 1.0,
        **kwargs,
    ) -> str:
        """
        端到端生成
        
        Args:
            text: 输入文本
            scene: 场景
            max_length: 最大长度
            temperature: 采样温度
            
        Returns:
            生成的文本
        """
        # 编码
        latent = self.encode(text, scene=scene)
        
        # 解码生成
        output = self.decoder.generate(
            latent,
            max_length=max_length,
            temperature=temperature,
            **kwargs,
        )
        
        return output.text[0] if output.text else ""
    
    def predict_next(
        self,
        history: List[UnifiedLatent],
    ) -> UnifiedLatent:
        """
        预测下一个潜向量
        
        Args:
            history: 历史潜向量列表
            
        Returns:
            预测的 UnifiedLatent
        """
        if self.dynamics is None:
            raise RuntimeError("Dynamics model not initialized")
        return self.dynamics(history)
    
    def forward(
        self,
        text: Union[str, List[str]],
        scene: Optional[str] = None,
        target_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        前向传播 (训练模式)
        
        Args:
            text: 输入文本
            scene: 场景
            target_ids: 目标 token IDs
            
        Returns:
            包含输出和损失的字典
        """
        # 编码
        encoder_output = self.encoder(text, scene=scene)
        
        # 解码
        decoder_output = self.decoder(
            encoder_output.unified_latent,
            target_ids=target_ids,
        )
        
        # 计算损失
        encoder_loss = self.encoder.get_loss(encoder_output)
        decoder_loss = decoder_output.loss if decoder_output.loss is not None else torch.tensor(0.0)
        
        total_loss = encoder_loss + decoder_loss
        
        return {
            "loss": total_loss,
            "encoder_loss": encoder_loss,
            "decoder_loss": decoder_loss,
            "unified_latent": encoder_output.unified_latent,
            "decoder_output": decoder_output,
            "encoding_info": encoder_output.encoding_info,
        }
    
    def get_trainable_params(self, stage: str = "all") -> List[nn.Parameter]:
        """
        获取指定阶段的可训练参数
        
        Args:
            stage: "encoder", "decoder", "dynamics", "all"
            
        Returns:
            参数列表
        """
        params = []
        
        if stage in ["encoder", "all"]:
            params.extend(self.encoder.parameters())
        
        if stage in ["decoder", "all"]:
            params.extend(self.decoder.parameters())
        
        if stage in ["dynamics", "all"] and self.dynamics is not None:
            params.extend(self.dynamics.parameters())
        
        return params
    
    def save_pretrained(
        self,
        path: str,
        save_graph_memory: bool = True,
    ) -> None:
        """
        保存模型到目录
        
        Args:
            path: 目标目录
            save_graph_memory: 是否保存 GraphMemory
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        config_dict = {
            "model": self.config.model.__dict__ if hasattr(self.config.model, '__dict__') else {},
            "metadata": self.metadata.to_dict(),
            "use_three_channel": self.use_three_channel,
        }
        save_config(config_dict, path / "config.json")
        
        # 保存模型权重
        save_safetensors(self.state_dict(), path / "model.safetensors")
        
        # 保存 GraphMemory (如果需要)
        if save_graph_memory:
            gm_path = path / "graph_memory"
            gm_path.mkdir(exist_ok=True)
            # GraphMemory 序列化
            torch.save({
                "nodes": {k: {"name": v.name, "vector": v.vector, "type": v.node_type} 
                         for k, v in self.graph_memory._nodes.items()},
                "num_nodes": self.graph_memory.num_nodes,
                "num_edges": self.graph_memory.num_edges,
            }, gm_path / "memory.pt")
    
    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: str = "cpu",
        load_graph_memory: bool = True,
    ) -> "UnifiedNeuralFlowModel":
        """
        从目录加载模型
        
        Args:
            path: 模型目录
            device: 目标设备
            load_graph_memory: 是否加载 GraphMemory
            
        Returns:
            加载的模型
        """
        path = Path(path)
        
        # 加载配置
        config_dict = load_config(path / "config.json")
        
        # 创建配置对象
        config = Config()
        if "model" in config_dict:
            for k, v in config_dict["model"].items():
                if hasattr(config.model, k):
                    setattr(config.model, k, v)
        
        use_three_channel = config_dict.get("use_three_channel", True)
        
        # 创建模型
        model = cls(config=config, use_three_channel=use_three_channel)
        
        # 加载权重
        weights_path = path / "model.safetensors"
        if weights_path.exists():
            state_dict = load_safetensors(weights_path)
            safe_load_state_dict(model, state_dict)
        else:
            # 尝试加载 .pt 格式
            pt_path = path / "model.pt"
            if pt_path.exists():
                state_dict = torch.load(pt_path, map_location=device)
                safe_load_state_dict(model, state_dict)
        
        # 加载 GraphMemory
        if load_graph_memory:
            gm_path = path / "graph_memory" / "memory.pt"
            if gm_path.exists():
                gm_data = torch.load(gm_path, map_location=device)
                # 恢复节点 (简化版)
                model.graph_memory.clear()
        
        # 加载元数据
        if "metadata" in config_dict:
            model.metadata = UnifiedModelMetadata.from_dict(config_dict["metadata"])
        
        return model.to(device)
    
    @classmethod
    def from_config(cls, config: Config, **kwargs) -> "UnifiedNeuralFlowModel":
        """从配置创建模型"""
        # 从 kwargs 中移除以避免重复传参
        kwargs.pop('use_three_channel', None)
        use_three_channel = getattr(config.model, 'use_three_channel', True)
        return cls(config=config, use_three_channel=use_three_channel, **kwargs)
    
    @classmethod
    def from_preset(cls, preset: str = "base") -> "UnifiedNeuralFlowModel":
        """
        从预设创建模型
        
        Args:
            preset: "base", "small", "large", "coding"
        """
        config = Config()
        
        if preset == "small":
            config.model.d_model = 512
            config.model.d_latent = 256
        elif preset == "large":
            config.model.d_model = 1024
            config.model.d_latent = 768
        elif preset == "coding":
            config.model.use_three_channel = True
            # 更强的结构编码
        
        return cls.from_config(config)
    
    def __repr__(self) -> str:
        return (
            f"UnifiedNeuralFlowModel(\n"
            f"  three_channel={self.use_three_channel},\n"
            f"  encoder={type(self.encoder).__name__},\n"
            f"  decoder={type(self.decoder).__name__},\n"
            f"  dynamics={'enabled' if self.dynamics else 'disabled'},\n"
            f")"
        )


# ============================================================================
# 工厂方法
# ============================================================================

def create_unified_model(
    config: Optional[Config] = None,
    model_type: str = "unified",
    use_three_channel: bool = True,
    **kwargs,
) -> nn.Module:
    """
    工厂方法：创建统一模型
    
    Args:
        config: 配置
        model_type: "unified" 或 "legacy"
        use_three_channel: 是否启用三通道
        **kwargs: 额外参数
        
    Returns:
        模型实例
    """
    if config is None:
        config = Config()
    
    if model_type == "unified":
        return UnifiedNeuralFlowModel(
            config=config,
            use_three_channel=use_three_channel,
            **kwargs,
        )
    elif model_type == "legacy":
        # 导入旧版模型
        from .neuralflow_model import NeuralFlowModel
        return NeuralFlowModel(config=config, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_model(path: str, device: str = "cpu") -> nn.Module:
    """
    智能加载模型 (自动检测类型)
    
    Args:
        path: 模型目录
        device: 目标设备
        
    Returns:
        模型实例
    """
    path = Path(path)
    config_path = path / "config.json"
    
    if config_path.exists():
        config_dict = load_config(config_path)
        model_type = config_dict.get("metadata", {}).get("model_type", "NeuralFlow")
        
        if model_type == "UnifiedNeuralFlow":
            return UnifiedNeuralFlowModel.from_pretrained(path, device=device)
        else:
            from .neuralflow_model import NeuralFlowModel
            return NeuralFlowModel.from_pretrained(path, device=device)
    else:
        raise FileNotFoundError(f"Config not found at {config_path}")
