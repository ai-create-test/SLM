"""
Dynamics Model - 动态动力学预测模型

核心功能：预测下一个段落的潜向量

架构选择:
- Mamba (State Space Model) - 推荐，线性复杂度
- GRU - 简单稳定，作为备选
- Transformer - 传统方案，二次复杂度

预测任务:
    z_{t+1}^ = f(z_1, z_2, ..., z_t, memory_context, emotion)
"""

from typing import Optional, Tuple, List
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces.base_module import BaseModule, ModuleOutput, LatentVector
from ..interfaces.config import ModelConfig
from ..interfaces.registry import Registry


@dataclass
class DynamicsOutput(ModuleOutput):
    """动力学模型输出"""
    predicted_latent: torch.Tensor = None     # 预测的下一个潜向量 [batch, d_latent]
    hidden_states: torch.Tensor = None        # 隐藏状态序列 [batch, seq_len, d_model]
    final_state: Optional[torch.Tensor] = None  # 最终状态 (用于续写)


class MambaBlock(nn.Module):
    """
    Mamba Block - 选择性状态空间模型
    
    核心公式 (简化版):
        h_t = A * h_{t-1} + B * x_t
        y_t = C * h_t + D * x_t
    
    其中 A, B, C 是输入依赖的 (选择性机制)
    
    参考: Mamba: Linear-Time Sequence Modeling with Selective State Spaces
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dropout: float = 0.0,
    ):
        """
        Args:
            d_model: 输入维度
            d_state: SSM 状态维度
            d_conv: 卷积核大小
            expand: 内部扩展因子
            dt_min/dt_max: 时间步长范围
            dt_init: 时间步长初始化方式
            dropout: Dropout 率
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        
        # ========== 输入投影 ==========
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # ========== 1D 卷积 ==========
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )
        
        # ========== SSM 参数投影 ==========
        # 输入依赖的 B, C, dt
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        
        # 可学习的 A 参数 (对数空间)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(self.d_inner, -1))
        
        # D 参数 (跳跃连接)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # dt 参数
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # 初始化 dt
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # ========== 输出投影 ==========
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # ========== Dropout ==========
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入序列 [batch, seq_len, d_model]
            state: 可选的初始状态 [batch, d_inner, d_state]
            
        Returns:
            output: 输出序列 [batch, seq_len, d_model]
            new_state: 新状态 [batch, d_inner, d_state]
        """
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        xz = self.in_proj(x)  # [batch, seq_len, d_inner * 2]
        x, z = xz.chunk(2, dim=-1)  # 各 [batch, seq_len, d_inner]
        
        # 1D 卷积
        x = x.transpose(1, 2)  # [batch, d_inner, seq_len]
        x = self.conv1d(x)[:, :, :seq_len]  # 截断到原长度
        x = x.transpose(1, 2)  # [batch, seq_len, d_inner]
        
        x = F.silu(x)
        
        # SSM 计算
        y, new_state = self._ssm_forward(x, state)
        
        # 门控
        y = y * F.silu(z)
        
        # 输出投影
        output = self.out_proj(y)
        output = self.dropout(output)
        
        return output, new_state
    
    def _ssm_forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        选择性 SSM 前向传播
        
        简化版实现，完整版需要 CUDA 优化。
        """
        batch_size, seq_len, d_inner = x.shape
        
        # 计算输入依赖的参数
        x_proj = self.x_proj(x)  # [batch, seq_len, d_state * 2 + 1]
        
        # 分解为 B, C, dt
        B = x_proj[:, :, :self.d_state]          # [batch, seq_len, d_state]
        C = x_proj[:, :, self.d_state:2*self.d_state]  # [batch, seq_len, d_state]
        dt_raw = x_proj[:, :, -1:]               # [batch, seq_len, 1]
        
        # 计算 dt
        dt = F.softplus(self.dt_proj(dt_raw))    # [batch, seq_len, d_inner]
        
        # 计算 A (离散化)
        A = -torch.exp(self.A_log)  # [d_inner, d_state]
        
        # 初始化状态
        if state is None:
            state = torch.zeros(batch_size, d_inner, self.d_state, device=x.device)
        
        # 序列处理 (简化版，实际应用需要 scan 优化)
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch, d_inner]
            B_t = B[:, t, :]  # [batch, d_state]
            C_t = C[:, t, :]  # [batch, d_state]
            dt_t = dt[:, t, :]  # [batch, d_inner]
            
            # 离散化 A 和 B
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))  # [batch, d_inner, d_state]
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # [batch, d_inner, d_state]
            
            # 状态更新
            state = dA * state + dB * x_t.unsqueeze(-1)
            
            # 输出计算
            y_t = torch.sum(state * C_t.unsqueeze(1), dim=-1)  # [batch, d_inner]
            y_t = y_t + self.D * x_t  # 跳跃连接
            
            outputs.append(y_t)
        
        output = torch.stack(outputs, dim=1)  # [batch, seq_len, d_inner]
        
        return output, state


class GRUDynamics(nn.Module):
    """
    GRU 动力学模型 (备选方案)
    
    简单稳定，适合快速原型验证。
    """
    
    def __init__(
        self,
        d_latent: int,
        d_model: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(d_latent, d_model)
        
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.output_proj = nn.Linear(d_model, d_latent)
    
    def forward(
        self,
        z_sequence: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_sequence: 潜向量序列 [batch, seq_len, d_latent]
            hidden: 隐藏状态
            
        Returns:
            predicted: 预测的下一个潜向量 [batch, d_latent]
            new_hidden: 新隐藏状态
        """
        x = self.input_proj(z_sequence)
        output, new_hidden = self.gru(x, hidden)
        
        # 取最后一个时间步
        predicted = self.output_proj(output[:, -1, :])
        
        return predicted, new_hidden


@Registry.register("brain", "dynamics")
class DynamicsModel(BaseModule):
    """
    动态动力学预测模型
    
    预测下一个段落的潜向量。
    
    使用示例:
        model = DynamicsModel.from_config(config)
        
        # 预测下一个潜向量
        z_history = torch.randn(batch, seq_len, d_latent)
        output = model(z_history, memory_context=memory_vec)
        z_next = output.predicted_latent
    """
    
    MODULE_TYPE = "brain"
    
    def __init__(
        self,
        d_latent: int = 512,
        d_model: int = 768,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 6,
        brain_type: str = "mamba",
        dropout: float = 0.1,
    ):
        """
        Args:
            d_latent: 潜空间维度
            d_model: 内部隐藏维度
            d_state: SSM 状态维度
            d_conv: 卷积核大小
            expand: 扩展因子
            num_layers: 层数
            brain_type: 模型类型 ('mamba', 'gru')
            dropout: Dropout率
        """
        super().__init__()
        
        self.d_latent = d_latent
        self.d_model = d_model
        self.brain_type = brain_type
        self.num_layers = num_layers
        
        # ========== 输入投影 ==========
        self.input_proj = nn.Sequential(
            nn.Linear(d_latent, d_model),
            nn.LayerNorm(d_model),
        )
        
        # ========== 记忆融合投影 ==========
        self.memory_proj = nn.Linear(d_latent, d_model)
        
        # ========== 核心动力学层 ==========
        if brain_type == "mamba":
            self.blocks = nn.ModuleList([
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ])
        elif brain_type == "gru":
            self.gru = GRUDynamics(d_latent, d_model, num_layers, dropout)
            self.blocks = None
        else:
            raise ValueError(f"Unknown brain_type: {brain_type}")
        
        # ========== 层归一化 ==========
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        
        # ========== 输出投影 ==========
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_latent),
        )
        
        # ========== Dropout ==========
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        z_sequence: torch.Tensor,
        memory_context: Optional[torch.Tensor] = None,
        states: Optional[List[torch.Tensor]] = None,
    ) -> DynamicsOutput:
        """
        前向传播
        
        Args:
            z_sequence: 历史潜向量序列 [batch, seq_len, d_latent]
            memory_context: 记忆上下文 [batch, d_latent] (来自记忆检索)
            states: 各层的状态 (用于流式推理)
            
        Returns:
            DynamicsOutput
        """
        batch_size, seq_len, _ = z_sequence.shape
        
        # 输入投影
        h = self.input_proj(z_sequence)  # [batch, seq_len, d_model]
        
        # 融合记忆上下文 (如果提供)
        if memory_context is not None:
            mem = self.memory_proj(memory_context).unsqueeze(1)  # [batch, 1, d_model]
            h = h + mem  # 广播加法
        
        # 通过动力学层
        new_states = []
        
        if self.brain_type == "mamba":
            for i, (block, norm) in enumerate(zip(self.blocks, self.norms)):
                state = states[i] if states is not None else None
                residual = h
                h, new_state = block(h, state)
                h = norm(residual + h)
                new_states.append(new_state)
        else:
            # GRU 路径
            state = states[0] if states is not None else None
            predicted, new_state = self.gru(z_sequence, state)
            new_states = [new_state]
            
            return DynamicsOutput(
                data=predicted,
                predicted_latent=predicted,
                hidden_states=h,
                final_state=new_state,
            )
        
        # 最终归一化
        h = self.final_norm(h)
        
        # 取最后一个时间步进行预测
        h_last = h[:, -1, :]  # [batch, d_model]
        
        # 输出投影得到预测的潜向量
        predicted = self.output_proj(h_last)  # [batch, d_latent]
        
        return DynamicsOutput(
            data=predicted,
            predicted_latent=predicted,
            hidden_states=h,
            final_state=new_states[-1] if new_states else None,
        )
    
    def predict_next(
        self,
        z_history: List[LatentVector],
        memory_context: Optional[LatentVector] = None,
    ) -> LatentVector:
        """
        便捷方法：从 LatentVector 列表预测下一个
        """
        # 堆叠历史向量
        z_sequence = torch.stack([z.vector for z in z_history], dim=1)
        
        # 准备记忆上下文
        mem = memory_context.vector if memory_context is not None else None
        
        # 预测
        output = self.forward(z_sequence.unsqueeze(0), mem.unsqueeze(0) if mem is not None else None)
        
        return LatentVector(vector=output.predicted_latent.squeeze(0))
    
    @classmethod
    def from_config(cls, config: ModelConfig, **kwargs) -> "DynamicsModel":
        """从配置创建实例"""
        return cls(
            d_latent=config.d_latent,
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            num_layers=config.num_layers,
            brain_type=config.brain_type,
            dropout=config.dropout,
            **kwargs,
        )
