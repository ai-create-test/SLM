"""
Reasoning Loop - 思考循环协调器

核心功能：协调整个推理流程

整合:
- Dynamics Model (预测)
- ACT Controller (自适应深度)
- Memory Retrieval (记忆)
- Modulation (情感/场景)
"""

from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import torch
import torch.nn as nn

from ..interfaces.base_module import BaseModule, ModuleOutput, LatentVector
from ..interfaces.config import ModelConfig
from ..interfaces.registry import Registry
from .dynamics_model import DynamicsModel, DynamicsOutput
from .act_controller import ACTController, ACTOutput, SimpleThinkingStep


@dataclass
class ReasoningOutput(ModuleOutput):
    """推理输出"""
    predicted_latent: LatentVector    # 预测的下一个潜向量
    reasoning_steps: int              # 实际推理步数
    act_output: Optional[ACTOutput]   # ACT 详细输出
    dynamics_output: DynamicsOutput   # 动力学模型输出
    memory_used: Optional[List[int]]  # 使用的记忆索引


@Registry.register("brain", "reasoning_loop")
class ReasoningLoop(BaseModule):
    """
    思考循环协调器
    
    完整的推理流程：
    1. 接收历史潜向量序列
    2. 从记忆库检索相关记忆 (外部提供)
    3. 通过 Dynamics Model 进行初步预测
    4. 通过 ACT 进行自适应深度思考
    5. 应用情感/场景调制 (外部提供)
    6. 输出最终预测的潜向量
    
    使用示例:
        loop = ReasoningLoop.from_config(config)
        
        # 推理
        output = loop(
            z_history=latent_sequence,
            memory_context=memory_vec,
            emotion=emotion_vec,
        )
        z_next = output.predicted_latent
    """
    
    MODULE_TYPE = "brain"
    
    def __init__(
        self,
        d_latent: int = 512,
        d_model: int = 768,
        d_state: int = 64,
        num_layers: int = 6,
        max_think_steps: int = 10,
        brain_type: str = "mamba",
        use_act: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_latent: 潜向量维度
            d_model: 内部隐藏维度
            d_state: SSM 状态维度
            num_layers: 动力学模型层数
            max_think_steps: 最大思考步数
            brain_type: 模型类型 ('mamba', 'gru')
            use_act: 是否使用 ACT
            dropout: Dropout 率
        """
        super().__init__()
        
        self.d_latent = d_latent
        self.d_model = d_model
        self.use_act = use_act
        
        # ========== 动力学模型 ==========
        self.dynamics = DynamicsModel(
            d_latent=d_latent,
            d_model=d_model,
            d_state=d_state,
            num_layers=num_layers,
            brain_type=brain_type,
            dropout=dropout,
        )
        
        # ========== ACT 控制器 ==========
        if use_act:
            self.act = ACTController(
                d_model=d_model,
                max_steps=max_think_steps,
            )
            # 思考步骤 (用于 ACT 循环)
            self.think_step = SimpleThinkingStep(d_model)
        else:
            self.act = None
            self.think_step = None
        
        # ========== 投影层 ==========
        self.latent_to_model = nn.Linear(d_latent, d_model)
        self.model_to_latent = nn.Linear(d_model, d_latent)
        
        # ========== 最终处理 ==========
        self.final_norm = nn.LayerNorm(d_latent)
    
    def forward(
        self,
        z_history: torch.Tensor,
        memory_context: Optional[torch.Tensor] = None,
        emotion: Optional[torch.Tensor] = None,
        scene: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
    ) -> ReasoningOutput:
        """
        执行推理循环
        
        Args:
            z_history: 历史潜向量序列 [batch, seq_len, d_latent]
            memory_context: 记忆上下文 [batch, d_latent]
            emotion: 情感向量 [batch, d_emotion]
            scene: 场景向量 [batch, d_scene]
            return_intermediate: 是否返回中间结果
            
        Returns:
            ReasoningOutput
        """
        batch_size = z_history.shape[0]
        
        # 1. 动力学模型初步预测
        dynamics_output = self.dynamics(z_history, memory_context)
        initial_prediction = dynamics_output.predicted_latent  # [batch, d_latent]
        
        # 2. 转换到模型空间
        state = self.latent_to_model(initial_prediction)  # [batch, d_model]
        
        # 3. 准备上下文 (情感 + 场景)
        context = None
        if emotion is not None or scene is not None:
            # TODO: 这里应该使用 modulation 模块的 AdaLN
            # 当前简化为加法
            context_parts = []
            if emotion is not None:
                context_parts.append(emotion)
            if scene is not None:
                context_parts.append(scene)
            
            # 假设情感/场景已经是 d_model 维度
            # 实际应该有投影层
            
        # 4. ACT 自适应思考
        if self.use_act and self.act is not None:
            act_output = self.act(
                initial_state=state,
                step_fn=self.think_step,
                context=context,
            )
            final_state = act_output.final_state
            num_steps = act_output.num_steps.mean().item()
        else:
            # 不使用 ACT，直接使用初始预测
            act_output = None
            final_state = state
            num_steps = 1
        
        # 5. 转换回潜向量空间
        predicted_latent = self.model_to_latent(final_state)  # [batch, d_latent]
        predicted_latent = self.final_norm(predicted_latent)
        
        # 包装为 LatentVector
        latent = LatentVector(
            vector=predicted_latent,
            metadata={
                "reasoning_steps": num_steps,
                "used_act": self.use_act,
            },
        )
        
        return ReasoningOutput(
            data=predicted_latent,
            predicted_latent=latent,
            reasoning_steps=int(num_steps),
            act_output=act_output,
            dynamics_output=dynamics_output,
            memory_used=None,  # 由外部记忆模块填充
        )
    
    def get_loss(
        self,
        output: ReasoningOutput,
        target: torch.Tensor,
        loss_type: str = "mse",
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        Args:
            output: 推理输出
            target: 目标潜向量 [batch, d_latent]
            loss_type: 损失类型 ('mse', 'cosine', 'contrastive')
            
        Returns:
            损失字典
        """
        predicted = output.predicted_latent.vector
        
        losses = {}
        
        # 预测损失
        if loss_type == "mse":
            losses["prediction"] = F.mse_loss(predicted, target)
        elif loss_type == "cosine":
            losses["prediction"] = 1 - F.cosine_similarity(predicted, target).mean()
        else:
            losses["prediction"] = F.mse_loss(predicted, target)
        
        # ACT 正则化损失
        if output.act_output is not None:
            losses["ponder"] = self.act.get_ponder_loss(output.act_output)
        
        # 总损失
        losses["total"] = sum(losses.values())
        
        return losses
    
    @classmethod
    def from_config(cls, config: ModelConfig, **kwargs) -> "ReasoningLoop":
        """从配置创建实例"""
        return cls(
            d_latent=config.d_latent,
            d_model=config.d_model,
            d_state=config.d_state,
            num_layers=config.num_layers,
            max_think_steps=config.max_think_steps,
            brain_type=config.brain_type,
            dropout=config.dropout,
            **kwargs,
        )


# 需要导入 F
import torch.nn.functional as F
