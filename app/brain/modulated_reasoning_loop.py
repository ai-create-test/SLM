"""
Modulated Reasoning Loop - 带情感调制的推理循环

核心功能：在整个推理流程中深度集成情感/场景调制

与原始 ReasoningLoop 的区别：
- 原始：情感在 TODO 注释中，未实现
- 新版：使用 ModulatedDynamicsModel，情感在每层 Mamba 中调制
"""

from typing import Optional, List
from dataclasses import dataclass
import torch
import torch.nn as nn

from ..interfaces.base_module import BaseModule, ModuleOutput, LatentVector
from ..interfaces.config import ModelConfig
from ..interfaces.registry import Registry
from .modulated_dynamics import ModulatedDynamicsModel, ModulatedDynamicsOutput
from .act_controller import ACTController, ACTOutput, SimpleThinkingStep


@dataclass
class ModulatedReasoningOutput(ModuleOutput):
    """调制推理输出"""
    predicted_latent: LatentVector = None    # 预测的下一个潜向量
    reasoning_steps: int = 0                 # 实际推理步数
    act_output: Optional[ACTOutput] = None   # ACT 详细输出
    dynamics_output: ModulatedDynamicsOutput = None  # 动力学模型输出
    emotion_used: bool = False               # 是否使用了情感
    memory_used: Optional[List[int]] = None  # 使用的记忆索引


@Registry.register("brain", "modulated_reasoning_loop")
class ModulatedReasoningLoop(BaseModule):
    """
    带情感调制的推理循环
    
    完整的推理流程：
    1. 接收历史潜向量序列
    2. 接收情感/场景条件
    3. 通过 ModulatedDynamicsModel 进行调制预测
    4. 可选：通过 ACT 进行自适应深度思考
    5. 输出调制后的预测
    
    使用示例:
        loop = ModulatedReasoningLoop(d_latent=512, d_condition=128)
        
        # 不同情感产生不同输出
        out_neutral = loop(z_seq, emotion=neutral_vec)
        out_angry = loop(z_seq, emotion=angry_vec)
        
        assert (out_neutral.predicted_latent - out_angry.predicted_latent).abs().mean() > 0
    """
    
    MODULE_TYPE = "brain"
    
    def __init__(
        self,
        d_latent: int = 512,
        d_model: int = 768,
        d_condition: int = 128,
        d_state: int = 64,
        num_layers: int = 6,
        max_think_steps: int = 10,
        use_act: bool = True,
        use_multi_condition: bool = False,
        d_emotion: int = None,
        d_scene: int = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_latent: 潜向量维度
            d_model: 内部隐藏维度
            d_condition: 条件向量维度
            d_state: SSM 状态维度
            num_layers: 层数
            max_think_steps: ACT 最大步数
            use_act: 是否使用 ACT
            use_multi_condition: 是否使用多条件模式
            d_emotion: 情感维度 (多条件模式)
            d_scene: 场景维度 (多条件模式)
            dropout: Dropout 率
        """
        super().__init__()
        
        self.d_latent = d_latent
        self.d_model = d_model
        self.d_condition = d_condition
        self.use_act = use_act
        
        # ========== 带调制的动力学模型 ==========
        self.dynamics = ModulatedDynamicsModel(
            d_latent=d_latent,
            d_model=d_model,
            d_condition=d_condition,
            d_state=d_state,
            num_layers=num_layers,
            dropout=dropout,
            use_multi_condition=use_multi_condition,
            d_emotion=d_emotion,
            d_scene=d_scene,
        )
        
        # ========== ACT 控制器 ==========
        if use_act:
            self.act = ACTController(
                d_model=d_model,
                max_steps=max_think_steps,
            )
            self.think_step = SimpleThinkingStep(d_model)
            
            # 情感融合到 ACT context
            self.condition_to_model = nn.Linear(d_condition, d_model)
        else:
            self.act = None
            self.think_step = None
            self.condition_to_model = None
        
        # ========== 投影层 ==========
        self.latent_to_model = nn.Linear(d_latent, d_model)
        self.model_to_latent = nn.Linear(d_model, d_latent)
        
        # ========== 最终处理 ==========
        self.final_norm = nn.LayerNorm(d_latent)
    
    def forward(
        self,
        z_history: torch.Tensor,
        emotion: Optional[torch.Tensor] = None,
        scene: Optional[torch.Tensor] = None,
        memory_context: Optional[torch.Tensor] = None,
    ) -> ModulatedReasoningOutput:
        """
        执行调制推理循环
        
        Args:
            z_history: 历史潜向量序列 [batch, seq_len, d_latent]
            emotion: 情感向量 [batch, d_condition]
            scene: 场景向量 [batch, d_scene] (多条件模式)
            memory_context: 记忆上下文 [batch, d_latent]
            
        Returns:
            ModulatedReasoningOutput
        """
        batch_size = z_history.shape[0]
        
        # 1. 调制动力学预测
        dynamics_output = self.dynamics(
            z_history,
            emotion=emotion,
            scene=scene,
            memory_context=memory_context,
        )
        initial_prediction = dynamics_output.predicted_latent  # [batch, d_latent]
        
        # 2. 转换到模型空间
        state = self.latent_to_model(initial_prediction)  # [batch, d_model]
        
        # 3. ACT 自适应思考 (带情感 context)
        if self.use_act and self.act is not None:
            # 将情感也融入 ACT 的 context
            context = None
            if emotion is not None and self.condition_to_model is not None:
                context = self.condition_to_model(emotion)  # [batch, d_model]
            
            act_output = self.act(
                initial_state=state,
                step_fn=self.think_step,
                context=context,
            )
            final_state = act_output.final_state
            num_steps = int(act_output.num_steps.mean().item())
        else:
            act_output = None
            final_state = state
            num_steps = 1
        
        # 4. 转换回潜向量空间
        predicted_latent = self.model_to_latent(final_state)
        predicted_latent = self.final_norm(predicted_latent)
        
        # 包装为 LatentVector
        latent = LatentVector(
            vector=predicted_latent,
            metadata={
                "reasoning_steps": num_steps,
                "emotion_used": emotion is not None,
            },
        )
        
        return ModulatedReasoningOutput(
            data=predicted_latent,
            predicted_latent=latent,
            reasoning_steps=num_steps,
            act_output=act_output,
            dynamics_output=dynamics_output,
            emotion_used=emotion is not None,
        )
    
    @classmethod
    def from_config(cls, config: ModelConfig, **kwargs) -> "ModulatedReasoningLoop":
        """从配置创建"""
        return cls(
            d_latent=config.d_latent,
            d_model=config.d_model,
            d_condition=getattr(config, 'd_condition', 128),
            d_state=config.d_state,
            num_layers=config.num_layers,
            max_think_steps=config.max_think_steps,
            use_act=config.use_act,
            dropout=config.dropout,
            **kwargs,
        )
