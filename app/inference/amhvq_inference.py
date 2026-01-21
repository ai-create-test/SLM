"""
AMHVQ+ Inference Interface - 推理接口

Phase 16: 精度自适应推理

提供简洁的推理 API:
- 场景自适应
- 精度控制
- 批量处理
"""

from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import torch

from ..interfaces.unified_latent import UnifiedLatent, PrecisionConfig
from ..model.unified_model import UnifiedNeuralFlowModel, create_unified_model, load_model


@dataclass
class InferenceConfig:
    """推理配置"""
    scene: str = "auto"  # "auto", "chat", "coding", "technical", "creative"
    precision: str = "adaptive"  # "low", "medium", "high", "adaptive"
    max_length: int = 256
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    use_structure: Optional[bool] = None  # None = 自动
    use_symbols: Optional[bool] = None    # None = 自动


@dataclass
class InferenceOutput:
    """推理输出"""
    text: str
    latent: UnifiedLatent
    scene: str
    precision_config: PrecisionConfig
    metadata: Dict[str, Any]


class AMHVQInference:
    """
    AMHVQ+ 推理接口
    
    提供简洁的推理 API，自动处理场景检测和精度调整。
    
    使用示例:
        inference = AMHVQInference.from_pretrained("./model_dir")
        
        # 自动场景检测
        output = inference.generate("def hello(): pass")
        
        # 指定场景
        output = inference.generate("Hello!", scene="chat")
        
        # 精度控制
        output = inference.generate("code here", precision="high")
    """
    
    def __init__(
        self,
        model: UnifiedNeuralFlowModel,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def encode(
        self,
        text: Union[str, List[str]],
        scene: Optional[str] = None,
    ) -> UnifiedLatent:
        """
        编码文本
        
        Args:
            text: 输入文本
            scene: 场景 (None = 自动检测)
            
        Returns:
            UnifiedLatent
        """
        with torch.no_grad():
            return self.model.encode(text, scene=scene)
    
    def decode(
        self,
        latent: UnifiedLatent,
        config: Optional[InferenceConfig] = None,
    ) -> str:
        """
        解码潜向量
        
        Args:
            latent: UnifiedLatent
            config: 推理配置
            
        Returns:
            生成的文本
        """
        config = config or InferenceConfig()
        
        with torch.no_grad():
            output = self.model.decoder.generate(
                latent,
                max_length=config.max_length,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
            )
            return output.text[0] if output.text else ""
    
    def generate(
        self,
        text: str,
        scene: Optional[str] = None,
        precision: str = "adaptive",
        max_length: int = 256,
        temperature: float = 1.0,
        **kwargs,
    ) -> InferenceOutput:
        """
        端到端生成
        
        Args:
            text: 输入文本
            scene: 场景 (None = 自动检测)
            precision: "low", "medium", "high", "adaptive"
            max_length: 最大长度
            temperature: 采样温度
            
        Returns:
            InferenceOutput
        """
        # 编码
        latent = self.encode(text, scene=scene)
        
        # 根据精度调整配置
        precision_config = self._get_precision_config(precision, latent.scene)
        
        # 解码
        config = InferenceConfig(
            scene=latent.scene,
            precision=precision,
            max_length=max_length,
            temperature=temperature,
        )
        
        generated_text = self.decode(latent, config)
        
        return InferenceOutput(
            text=generated_text,
            latent=latent,
            scene=latent.scene,
            precision_config=precision_config,
            metadata={
                "input_length": len(text),
                "output_length": len(generated_text),
                "channels_active": {
                    "semantic": True,
                    "structure": latent.has_structure,
                    "symbols": latent.has_symbols,
                },
            },
        )
    
    def _get_precision_config(self, precision: str, scene: str) -> PrecisionConfig:
        """获取精度配置"""
        if precision == "low":
            return PrecisionConfig(
                semantic=128,
                structure=False,
                symbols=False,
            )
        elif precision == "medium":
            return PrecisionConfig(
                semantic=256,
                structure=scene == "coding",
                symbols=False,
            )
        elif precision == "high":
            return PrecisionConfig(
                semantic=512,
                structure=True,
                symbols=True,
            )
        else:  # adaptive
            if scene == "coding":
                return PrecisionConfig(semantic=512, structure=True, symbols=True)
            elif scene == "technical":
                return PrecisionConfig(semantic=512, structure=True, symbols=False)
            else:
                return PrecisionConfig(semantic=256, structure=False, symbols=False)
    
    def batch_generate(
        self,
        texts: List[str],
        scene: Optional[str] = None,
        **kwargs,
    ) -> List[InferenceOutput]:
        """批量生成"""
        return [self.generate(text, scene=scene, **kwargs) for text in texts]
    
    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: str = "cpu",
    ) -> "AMHVQInference":
        """从预训练模型创建"""
        model = load_model(path, device=device)
        return cls(model, device=device)
    
    @classmethod
    def from_config(
        cls,
        config_path: str,
        device: str = "cpu",
    ) -> "AMHVQInference":
        """从配置创建"""
        from ..interfaces.config import Config
        config = Config.from_yaml(config_path)
        model = create_unified_model(config=config)
        return cls(model, device=device)
