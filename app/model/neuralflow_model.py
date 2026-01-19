"""
NeuralFlowModel - 统一模型封装

整合所有可训练组件，提供标准化接口:
- save_pretrained() / from_pretrained()
- 模块化组件访问
- 多阶段训练支持
"""

from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, asdict
import os
import json
from pathlib import Path

import torch
import torch.nn as nn

from ..interfaces.config import Config, ModelConfig, PipelineConfig
from ..io.paragraph_encoder import ParagraphEncoder
from ..io.paragraph_decoder import ParagraphDecoder
from ..brain.dynamics_model import DynamicsModel
from ..brain.modulated_dynamics import ModulatedDynamicsModel
from ..memory.latent_memory_bank import LatentMemoryBank
from ..memory.cross_attention_fuser import CrossAttentionFuser
from ..modulation import SemanticEmotionEncoder, SceneEncoder

from .model_utils import (
    ModelDirectory,
    save_safetensors,
    load_safetensors,
    save_config,
    load_config,
    save_training_state,
    load_training_state,
    safe_load_state_dict,
    is_safetensors_available,
)


@dataclass
class ModelMetadata:
    """模型元数据"""
    model_type: str = "NeuralFlow"
    version: str = "2.0.0"
    architecture: Dict[str, Any] = None
    emotion_system: Dict[str, Any] = None
    training: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelMetadata":
        return cls(**d)


class NeuralFlowModel(nn.Module):
    """
    NeuralFlow 统一模型封装
    
    整合所有可训练组件:
    - IO层: ParagraphEncoder, ParagraphDecoder
    - 核心大脑: DynamicsModel, ModulatedDynamicsModel
    - 情感系统: SemanticEmotionEncoder (VAD-based)
    - 记忆系统: LatentMemoryBank, CrossAttentionFuser
    
    使用示例:
        # 从配置创建
        model = NeuralFlowModel(config)
        
        # 加载预训练
        model = NeuralFlowModel.from_pretrained("models/neuralflow-base/")
        
        # 保存
        model.save_pretrained("models/my-model/")
        
        # 推理
        output = model.generate("输入文本", emotion="happy")
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        **kwargs,
    ):
        """
        Args:
            config: 完整配置对象
            **kwargs: 覆盖配置的参数
        """
        super().__init__()
        
        # 使用默认配置
        if config is None:
            config = Config()
        
        self.config = config
        model_cfg = config.model
        pipeline_cfg = config.pipeline
        
        # =====================================================================
        # IO Layer (编码/解码)
        # =====================================================================
        self.encoder = ParagraphEncoder.from_config(model_cfg)
        self.decoder = ParagraphDecoder.from_config(model_cfg)
        
        # =====================================================================
        # Core Brain (动力学模型)
        # =====================================================================
        self.dynamics = DynamicsModel.from_config(model_cfg)
        self.modulated_dynamics = ModulatedDynamicsModel.from_config(model_cfg)
        
        # =====================================================================
        # Emotion System (新版 VAD-based)
        # =====================================================================
        self.emotion_encoder = SemanticEmotionEncoder(
            d_emotion=model_cfg.d_emotion,
        )
        
        # =====================================================================
        # Scene Encoder
        # =====================================================================
        if pipeline_cfg.enable_scene:
            self.scene_encoder = SceneEncoder(d_scene=model_cfg.d_scene)
        else:
            self.scene_encoder = None
        
        # =====================================================================
        # Memory System (记忆)
        # =====================================================================
        if pipeline_cfg.enable_memory:
            self.memory_bank = LatentMemoryBank(
                d_latent=model_cfg.d_latent,
                max_size=model_cfg.memory_size,
            )
            self.memory_fuser = CrossAttentionFuser(
                d_latent=model_cfg.d_latent,
            )
        else:
            self.memory_bank = None
            self.memory_fuser = None
        
        # =====================================================================
        # 元数据
        # =====================================================================
        self._metadata = ModelMetadata(
            architecture={
                "d_latent": model_cfg.d_latent,
                "d_model": model_cfg.d_model,
                "d_emotion": model_cfg.d_emotion,
                "num_layers": model_cfg.num_layers,
                "brain_type": model_cfg.brain_type,
                "codebook_size": model_cfg.codebook_size,
                "num_codebooks": model_cfg.num_codebooks,
            },
            emotion_system={
                "type": "semantic",
                "tiers": ["lexicon", "retriever"],
                "lexicon_size": self.emotion_encoder.emotion_count,
            },
            training={
                "completed_stages": [],
                "total_steps": 0,
                "final_losses": {},
            },
        )
    
    # =========================================================================
    # 核心接口
    # =========================================================================
    
    def encode(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        编码文本到潜向量
        
        Args:
            text: 输入文本
            
        Returns:
            潜向量 [batch, d_latent]
        """
        output = self.encoder(text)
        return output.latent.vector
    
    def decode(
        self,
        latent: torch.Tensor,
        emotion: Optional[Union[str, torch.Tensor]] = None,
        scene: Optional[Union[str, torch.Tensor]] = None,
        **kwargs,
    ) -> List[str]:
        """
        解码潜向量到文本
        
        Args:
            latent: 潜向量 [batch, d_latent]
            emotion: 情感 (字符串或向量)
            scene: 场景 (字符串或向量)
            
        Returns:
            解码文本列表
        """
        emotion_vec = None
        scene_vec = None
        
        if emotion is not None:
            if isinstance(emotion, str):
                emotion_vec = self.emotion_encoder(emotion)
            else:
                emotion_vec = emotion
        
        if scene is not None and self.scene_encoder is not None:
            if isinstance(scene, str):
                scene_vec = self.scene_encoder.encode_name(scene)
            else:
                scene_vec = scene
        
        output = self.decoder.generate(
            latent,
            emotion=emotion_vec,
            scene=scene_vec,
            **kwargs,
        )
        return output.text
    
    def predict_next(
        self,
        z_history: torch.Tensor,
        emotion: Optional[Union[str, torch.Tensor]] = None,
        use_modulated: bool = True,
    ) -> torch.Tensor:
        """
        预测下一个潜向量
        
        Args:
            z_history: 历史潜向量序列 [batch, seq_len, d_latent]
            emotion: 情感控制
            use_modulated: 是否使用调制版本
            
        Returns:
            预测潜向量 [batch, d_latent]
        """
        if emotion is not None and use_modulated:
            if isinstance(emotion, str):
                emotion_vec = self.emotion_encoder(emotion)
            else:
                emotion_vec = emotion
            
            output = self.modulated_dynamics(z_history, emotion=emotion_vec)
        else:
            output = self.dynamics(z_history)
        
        return output.predicted_latent
    
    def generate(
        self,
        text: str,
        emotion: Optional[str] = None,
        scene: Optional[str] = None,
        max_length: int = 256,
        **kwargs,
    ) -> str:
        """
        端到端生成
        
        Args:
            text: 输入文本
            emotion: 情感控制
            scene: 场景控制
            max_length: 最大长度
            
        Returns:
            生成的文本
        """
        # 编码
        z = self.encode(text)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        
        # 预测下一个潜向量
        z_seq = z.unsqueeze(1)  # [batch, 1, d_latent]
        z_next = self.predict_next(z_seq, emotion=emotion)
        
        # 解码
        texts = self.decode(z_next, emotion=emotion, scene=scene, max_length=max_length)
        
        return texts[0] if texts else ""
    
    # =========================================================================
    # 组件访问
    # =========================================================================
    
    def get_trainable_params(self, stage: str) -> List[nn.Parameter]:
        """
        获取指定阶段的可训练参数
        
        Args:
            stage: 训练阶段 ("vqvae", "dynamics", "emotion", "finetune")
            
        Returns:
            参数列表
        """
        if stage == "vqvae":
            params = list(self.encoder.parameters())
            params += list(self.decoder.parameters())
            return params
        
        elif stage == "dynamics":
            return list(self.dynamics.parameters())
        
        elif stage == "emotion":
            params = list(self.emotion_encoder.vad_encoder.parameters())
            params += list(self.modulated_dynamics.parameters())
            return params
        
        elif stage == "finetune":
            return list(self.parameters())
        
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    def freeze_encoder(self) -> None:
        """冻结编码器"""
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
    
    def unfreeze_encoder(self) -> None:
        """解冻编码器"""
        self.encoder.train()
        for p in self.encoder.parameters():
            p.requires_grad = True
    
    # =========================================================================
    # 保存/加载
    # =========================================================================
    
    def save_pretrained(
        self,
        path: str,
        save_memory: bool = False,
    ) -> None:
        """
        保存模型到目录
        
        Args:
            path: 目标目录
            save_memory: 是否保存记忆库
        """
        model_dir = ModelDirectory(path)
        model_dir.ensure_exists()
        
        # 保存配置
        config_data = self.config.to_dict()
        config_data["_metadata"] = self._metadata.to_dict()
        save_config(config_data, str(model_dir.config_path))
        
        # 保存权重
        state_dict = self.state_dict()
        
        if is_safetensors_available():
            weights_path = model_dir.path / ModelDirectory.WEIGHTS_FILE
            save_safetensors(state_dict, str(weights_path))
        else:
            weights_path = model_dir.path / ModelDirectory.WEIGHTS_FILE_PT
            torch.save(state_dict, str(weights_path))
        
        # 保存情感词典
        emotion_dir = model_dir.path / ModelDirectory.EMOTION_DIR
        if hasattr(self.emotion_encoder, 'lexicon'):
            # 复制词典文件
            pass  # TODO: 实现词典导出
        
        # 保存记忆库
        if save_memory and self.memory_bank is not None:
            memory_path = model_dir.path / ModelDirectory.MEMORY_DIR / "memory.faiss"
            self.memory_bank.save(str(memory_path))
        
        print(f"Model saved to {path}")
    
    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: str = "cpu",
        load_memory: bool = False,
    ) -> "NeuralFlowModel":
        """
        从目录加载模型
        
        Args:
            path: 模型目录
            device: 目标设备
            load_memory: 是否加载记忆库
            
        Returns:
            加载的模型
        """
        model_dir = ModelDirectory(path)
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found at {path}")
        
        # 加载配置
        config_data = load_config(str(model_dir.config_path))
        metadata = config_data.pop("_metadata", {})
        
        config = Config.load(config_data) if isinstance(config_data, dict) else config_data
        
        # 创建模型
        model = cls(config)
        
        # 加载权重
        state_dict = load_safetensors(str(model_dir.weights_path), device=device)
        missing, unexpected = safe_load_state_dict(model, state_dict, strict=False)
        
        if missing:
            print(f"Warning: Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"Warning: Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        
        # 加载元数据
        if metadata:
            model._metadata = ModelMetadata.from_dict(metadata)
        
        # 加载记忆库
        if load_memory and model.memory_bank is not None:
            memory_path = model_dir.path / ModelDirectory.MEMORY_DIR / "memory.faiss"
            if memory_path.exists():
                model.memory_bank.load(str(memory_path))
        
        model.to(device)
        print(f"Model loaded from {path}")
        
        return model
    
    @classmethod
    def from_config(cls, config: Config) -> "NeuralFlowModel":
        """从配置创建模型"""
        return cls(config)
    
    @classmethod
    def from_preset(cls, preset: str = "base") -> "NeuralFlowModel":
        """从预设创建模型"""
        config = Config.from_preset(preset)
        return cls(config)
    
    # =========================================================================
    # 训练状态管理
    # =========================================================================
    
    def update_training_metadata(
        self,
        stage: str,
        steps: int,
        loss: float,
    ) -> None:
        """更新训练元数据"""
        if stage not in self._metadata.training["completed_stages"]:
            self._metadata.training["completed_stages"].append(stage)
        
        self._metadata.training["total_steps"] += steps
        self._metadata.training["final_losses"][stage] = loss
    
    @property
    def completed_stages(self) -> List[str]:
        """已完成的训练阶段"""
        return self._metadata.training.get("completed_stages", [])
    
    def __repr__(self) -> str:
        params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"NeuralFlowModel(\n"
            f"  d_latent={self.config.model.d_latent},\n"
            f"  d_model={self.config.model.d_model},\n"
            f"  brain_type='{self.config.model.brain_type}',\n"
            f"  params={params:,} (trainable={trainable:,}),\n"
            f"  stages={self.completed_stages}\n"
            f")"
        )
