"""
Modulated Dynamics Model - 带情感调制的动力学模型

核心功能：让情感/场景信号深度调制推理过程

与原始 DynamicsModel 的区别：
- 原始：情感信号完全独立，不参与推理
- 新版：情感信号通过 AdaLN 在每一层调制输出

设计理念：
- 相同输入 + 不同情感 = 不同输出
- neutral vs angry 应该产生明显不同的预测
"""

from dataclasses import dataclass
from typing import Optional, List, Union
import torch
import torch.nn as nn

from ..interfaces.base_module import BaseModule, ModuleOutput, LatentVector
from ..interfaces.config import ModelConfig
from ..interfaces.registry import Registry
from .modulated_mamba import ModulatedMambaStack, MultiConditionModulatedMamba
from .dynamics_model import GRUDynamics


@dataclass
class ModulatedDynamicsOutput(ModuleOutput):
    """调制动力学输出"""
    predicted_latent: torch.Tensor = None  # 预测的下一个潜向量
    hidden_states: torch.Tensor = None     # 中间隐藏状态
    final_state: torch.Tensor = None       # 最终状态
    emotion_used: bool = False             # 是否使用了情感调制


@Registry.register("brain", "modulated_dynamics")
class ModulatedDynamicsModel(BaseModule):
    """
    带情感/场景调制的动力学模型
    
    通过 AdaLN 让情感信号在每一层深度调制推理过程。
    
    使用示例:
        model = ModulatedDynamicsModel(
            d_latent=512,
            d_model=768,
            d_condition=128,  # 情感/场景条件维度
        )
        
        # 无情感 (退化为普通 Mamba)
        output = model(z_history)
        
        # 有情感调制
        emotion = emotion_encoder("happy")  # [batch, d_condition]
        output = model(z_history, emotion=emotion)
        
        # 情感 + 场景
        scene = scene_encoder("battle")
        output = model(z_history, emotion=emotion, scene=scene)
    """
    
    MODULE_TYPE = "brain"
    
    def __init__(
        self,
        d_latent: int = 512,
        d_model: int = 768,
        d_condition: int = 128,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 6,
        dropout: float = 0.1,
        use_multi_condition: bool = False,
        d_emotion: int = None,
        d_scene: int = None,
    ):
        """
        Args:
            d_latent: 潜空间维度
            d_model: 内部隐藏维度
            d_condition: 条件向量维度 (单条件模式)
            d_state: SSM 状态维度
            d_conv: 卷积核大小
            expand: 扩展因子
            num_layers: 层数
            dropout: Dropout率
            use_multi_condition: 是否使用多条件 (emotion+scene 分开)
            d_emotion: 情感维度 (多条件模式)
            d_scene: 场景维度 (多条件模式)
        """
        super().__init__()
        
        self.d_latent = d_latent
        self.d_model = d_model
        self.d_condition = d_condition
        self.num_layers = num_layers
        self.use_multi_condition = use_multi_condition
        
        # ========== 输入投影 ==========
        self.input_proj = nn.Sequential(
            nn.Linear(d_latent, d_model),
            nn.LayerNorm(d_model),
        )
        
        # ========== 记忆融合投影 ==========
        self.memory_proj = nn.Linear(d_latent, d_model)
        
        # ========== 核心：带调制的 Mamba 堆叠 ==========
        if use_multi_condition:
            # 多条件模式：emotion 和 scene 分开
            d_emotion = d_emotion or d_condition
            d_scene = d_scene or d_condition
            
            self.blocks = nn.ModuleList([
                MultiConditionModulatedMamba(
                    d_model=d_model,
                    d_emotion=d_emotion,
                    d_scene=d_scene,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ])
        else:
            # 单条件模式：统一的 condition 向量
            self.mamba_stack = ModulatedMambaStack(
                d_model=d_model,
                d_condition=d_condition,
                num_layers=num_layers,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            self.blocks = None
        
        # ========== 输出投影 ==========
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_latent),
        )
    
    def forward(
        self,
        z_sequence: torch.Tensor,
        emotion: Optional[torch.Tensor] = None,
        scene: Optional[torch.Tensor] = None,
        memory_context: Optional[torch.Tensor] = None,
        states: Optional[List[torch.Tensor]] = None,
    ) -> ModulatedDynamicsOutput:
        """
        前向传播
        
        Args:
            z_sequence: 历史潜向量序列 [batch, seq_len, d_latent]
            emotion: 情感向量 [batch, d_condition] 或 [batch, d_emotion]
            scene: 场景向量 [batch, d_scene] (多条件模式)
            memory_context: 记忆上下文 [batch, d_latent]
            states: 各层状态
            
        Returns:
            ModulatedDynamicsOutput
        """
        batch_size, seq_len, _ = z_sequence.shape
        device = z_sequence.device
        
        # 输入投影
        h = self.input_proj(z_sequence)  # [batch, seq_len, d_model]
        
        # 融合记忆上下文
        if memory_context is not None:
            mem = self.memory_proj(memory_context).unsqueeze(1)
            h = h + mem
        
        # 判断是否有条件
        emotion_used = emotion is not None
        
        # 通过调制 Mamba 层
        if self.use_multi_condition and self.blocks is not None:
            # 多条件模式
            new_states = []
            for i, block in enumerate(self.blocks):
                state = states[i] if states is not None else None
                h, new_state = block(h, emotion, scene, state)
                new_states.append(new_state)
        else:
            # 单条件模式
            condition = emotion  # 使用 emotion 作为统一条件
            h, new_states = self.mamba_stack(h, condition, states)
        
        # 取最后时间步
        h_last = h[:, -1, :]  # [batch, d_model]
        
        # 输出投影
        predicted = self.output_proj(h_last)  # [batch, d_latent]
        
        return ModulatedDynamicsOutput(
            data=predicted,
            predicted_latent=predicted,
            hidden_states=h,
            final_state=new_states[-1] if new_states else None,
            emotion_used=emotion_used,
        )
    
    def predict_with_emotion(
        self,
        z_history: List[LatentVector],
        emotion: torch.Tensor,
        memory_context: Optional[LatentVector] = None,
    ) -> LatentVector:
        """
        便捷方法：带情感的预测
        
        Args:
            z_history: 历史潜向量列表
            emotion: 情感向量 [d_condition] 或 [1, d_condition]
            memory_context: 记忆上下文
        """
        # 堆叠历史
        z_sequence = torch.stack([z.vector for z in z_history], dim=1)  # [1, seq, d_latent]
        if z_sequence.dim() == 2:
            z_sequence = z_sequence.unsqueeze(0)
        
        # 确保 emotion 是 2D
        if emotion.dim() == 1:
            emotion = emotion.unsqueeze(0)
        
        # 记忆上下文
        mem = memory_context.vector.unsqueeze(0) if memory_context else None
        
        # 预测
        output = self.forward(z_sequence, emotion=emotion, memory_context=mem)
        
        return LatentVector(vector=output.predicted_latent.squeeze(0))
    
    @classmethod
    def from_config(cls, config: ModelConfig, **kwargs) -> "ModulatedDynamicsModel":
        """从配置创建"""
        return cls(
            d_latent=config.d_latent,
            d_model=config.d_model,
            d_condition=getattr(config, 'd_condition', 128),
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            num_layers=config.num_layers,
            dropout=config.dropout,
            **kwargs,
        )
