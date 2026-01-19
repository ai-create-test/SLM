"""
Modulated Mamba - 带情感/场景调制的 Mamba Block

核心功能：将 AdaLN 深度集成到 Mamba 层中

设计原理：
- 情感/场景不是简单加法，而是通过 AdaLN 调制每一层的输出
- 这让情感信号能深度影响推理过程，而非仅在表面起作用
- 使用 Zero-Init 确保初始时不影响预训练行为
- 支持 Memory-Augmented 模式，通过 Cross-Attention 融合记忆上下文
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn

from ..modulation.adaln import AdaptiveLayerNorm, MultiConditionAdaLN
from ..memory.cross_attention_fuser import CrossAttentionFuser
from .dynamics_model import MambaBlock


class ModulatedMambaBlock(nn.Module):
    """
    带情感调制的 Mamba Block
    
    结构:
        x → MambaBlock → AdaLN(condition) → output
        
    情感/场景条件通过 AdaLN 调制 Mamba 的输出，
    实现深度干预而非简单向量加法。
    
    使用示例:
        block = ModulatedMambaBlock(d_model=768, d_condition=128)
        
        x = torch.randn(batch, seq_len, d_model)
        condition = emotion_encoder("happy")  # [batch, d_condition]
        
        output, state = block(x, condition)  # 情感深度调制
    """
    
    def __init__(
        self,
        d_model: int,
        d_condition: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: 隐藏维度
            d_condition: 条件向量维度 (情感/场景)
            d_state: SSM 状态维度
            d_conv: 卷积核大小
            expand: 扩展因子
            dropout: Dropout 率
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_condition = d_condition
        
        # ========== Mamba 核心 ==========
        self.mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        
        # ========== AdaLN 调制 ==========
        self.adaln = AdaptiveLayerNorm(
            d_model=d_model,
            d_condition=d_condition,
        )
        
        # ========== 层归一化 (Pre-LN) ==========
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入 [batch, seq_len, d_model]
            condition: 条件向量 [batch, d_condition] (情感/场景)
            state: 可选的 Mamba 状态
            
        Returns:
            output: 调制后的输出 [batch, seq_len, d_model]
            new_state: 新的 Mamba 状态
        """
        residual = x
        
        # 1. Mamba 处理
        h, new_state = self.mamba(x, state)
        
        # 2. 残差连接 + 归一化
        h = self.norm(residual + h)
        
        # 3. AdaLN 调制 (如果提供条件)
        if condition is not None:
            h = self.adaln(h, condition)
        
        return h, new_state


class ModulatedMambaStack(nn.Module):
    """
    多层 ModulatedMambaBlock 堆叠
    
    每一层都共享同一个情感条件，实现全局情感调制。
    
    使用示例:
        stack = ModulatedMambaStack(d_model=768, d_condition=128, num_layers=6)
        
        x = torch.randn(batch, seq_len, d_model)
        emotion = emotion_encoder("happy")
        
        output = stack(x, emotion)
    """
    
    def __init__(
        self,
        d_model: int,
        d_condition: int,
        num_layers: int = 6,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        
        # 创建多层
        self.blocks = nn.ModuleList([
            ModulatedMambaBlock(
                d_model=d_model,
                d_condition=d_condition,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # 最终归一化
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入 [batch, seq_len, d_model]
            condition: 条件向量 [batch, d_condition]
            states: 各层状态列表
            
        Returns:
            output: 输出 [batch, seq_len, d_model]
            new_states: 新状态列表
        """
        h = x
        new_states = []
        
        for i, block in enumerate(self.blocks):
            state = states[i] if states is not None else None
            h, new_state = block(h, condition, state)
            new_states.append(new_state)
        
        h = self.final_norm(h)
        
        return h, new_states


class MultiConditionModulatedMamba(nn.Module):
    """
    多条件 (情感+场景) 调制的 Mamba Block
    
    同时接收情感和场景信号，融合后进行调制。
    """
    
    def __init__(
        self,
        d_model: int,
        d_emotion: int,
        d_scene: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        fusion_type: str = "add",
    ):
        """
        Args:
            d_model: 隐藏维度
            d_emotion: 情感向量维度
            d_scene: 场景向量维度
            fusion_type: 融合方式 ('add', 'concat', 'gate')
        """
        super().__init__()
        
        self.mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        
        self.adaln = MultiConditionAdaLN(
            d_model=d_model,
            d_emotion=d_emotion,
            d_scene=d_scene,
            fusion_type=fusion_type,
        )
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        emotion: Optional[torch.Tensor] = None,
        scene: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入 [batch, seq_len, d_model]
            emotion: 情感向量 [batch, d_emotion]
            scene: 场景向量 [batch, d_scene]
            state: Mamba 状态
        """
        residual = x
        
        # Mamba
        h, new_state = self.mamba(x, state)
        
        # 残差 + 归一化
        h = self.norm(residual + h)
        
        # 多条件 AdaLN
        if emotion is not None and scene is not None:
            h = self.adaln(h, emotion, scene)
        
        return h, new_state


class MemoryAugmentedMambaBlock(nn.Module):
    """
    记忆增强的 Mamba Block
    
    结构:
        x → MambaBlock → CrossAttention(memory) → AdaLN(condition) → output
    
    在标准 Mamba 处理后，通过 Cross-Attention 融合记忆上下文，
    然后应用情感/场景调制。这使得记忆能够深度影响推理过程。
    
    使用示例:
        block = MemoryAugmentedMambaBlock(d_model=768, d_condition=128)
        
        x = torch.randn(batch, seq_len, d_model)
        condition = emotion_encoder("happy")     # [batch, d_condition]
        memory = retriever.retrieve(query)       # [batch, k, d_model]
        
        output, state = block(x, condition, memory_context=memory)
    """
    
    def __init__(
        self,
        d_model: int,
        d_condition: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        num_heads: int = 8,
        use_memory_fusion: bool = True,
        memory_gate_init: float = 0.0,
    ):
        """
        Args:
            d_model: 隐藏维度
            d_condition: 条件向量维度 (情感/场景)
            d_state: SSM 状态维度
            d_conv: 卷积核大小
            expand: 扩展因子
            dropout: Dropout 率
            num_heads: Cross-Attention 头数
            use_memory_fusion: 是否启用记忆融合
            memory_gate_init: 记忆门初始值 (0 = 初始不使用记忆)
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_condition = d_condition
        self.use_memory_fusion = use_memory_fusion
        
        # ========== Mamba 核心 ==========
        self.mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        
        # ========== 记忆融合 (可选) ==========
        if use_memory_fusion:
            self.memory_cross_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.memory_norm = nn.LayerNorm(d_model)
            # 可学习的记忆门控，初始化为接近 0
            self.memory_gate = nn.Parameter(
                torch.tensor(memory_gate_init, dtype=torch.float32)
            )
        else:
            self.memory_cross_attn = None
            self.memory_norm = None
            self.memory_gate = None
        
        # ========== AdaLN 调制 ==========
        self.adaln = AdaptiveLayerNorm(
            d_model=d_model,
            d_condition=d_condition,
        )
        
        # ========== 层归一化 ==========
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        memory_context: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入 [batch, seq_len, d_model]
            condition: 条件向量 [batch, d_condition] (情感/场景)
            state: 可选的 Mamba 状态
            memory_context: 记忆上下文 [batch, num_memories, d_model]
            memory_mask: 记忆掩码 [batch, num_memories] (True=padding)
            
        Returns:
            output: 调制后的输出 [batch, seq_len, d_model]
            new_state: 新的 Mamba 状态
        """
        residual = x
        batch_size, seq_len, _ = x.shape
        
        # 1. Mamba 处理
        h, new_state = self.mamba(x, state)
        
        # 2. 残差连接 + 归一化
        h = self.norm(residual + h)
        
        # 3. 记忆融合 (如果提供且启用)
        if self.use_memory_fusion and memory_context is not None:
            mem_residual = h
            
            # Cross-Attention: Q=h, K,V=memory
            attn_output, _ = self.memory_cross_attn(
                query=h,
                key=memory_context.expand(batch_size, -1, -1) if memory_context.size(0) == 1 else memory_context,
                value=memory_context.expand(batch_size, -1, -1) if memory_context.size(0) == 1 else memory_context,
                key_padding_mask=memory_mask,
            )
            
            # 门控残差
            gate = torch.sigmoid(self.memory_gate)
            h = mem_residual + gate * self.memory_norm(attn_output)
        
        # 4. AdaLN 调制 (如果提供条件)
        if condition is not None:
            h = self.adaln(h, condition)
        
        return h, new_state
    
    def get_memory_gate_value(self) -> float:
        """获取当前记忆门控值"""
        if self.memory_gate is not None:
            return torch.sigmoid(self.memory_gate).item()
        return 0.0


class MemoryAugmentedMambaStack(nn.Module):
    """
    多层 MemoryAugmentedMambaBlock 堆叠
    
    支持配置哪些层启用记忆融合 (节省计算)。
    
    使用示例:
        stack = MemoryAugmentedMambaStack(
            d_model=768,
            d_condition=128,
            num_layers=6,
            memory_fusion_layers=[0, 2, 4],  # 只在这些层融合记忆
        )
        
        x = torch.randn(batch, seq_len, d_model)
        memory = retriever.retrieve(query)
        emotion = encoder("happy")
        
        output, states = stack(x, emotion, memory_context=memory)
    """
    
    def __init__(
        self,
        d_model: int,
        d_condition: int,
        num_layers: int = 6,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        num_heads: int = 8,
        memory_fusion_layers: Optional[List[int]] = None,
    ):
        """
        Args:
            d_model: 隐藏维度
            d_condition: 条件向量维度
            num_layers: 层数
            memory_fusion_layers: 启用记忆融合的层索引 (默认所有层)
        """
        super().__init__()
        
        self.num_layers = num_layers
        
        # 确定哪些层启用记忆融合
        if memory_fusion_layers is None:
            memory_fusion_layers = list(range(num_layers))
        self.memory_fusion_layers = set(memory_fusion_layers)
        
        # 创建多层
        self.blocks = nn.ModuleList([
            MemoryAugmentedMambaBlock(
                d_model=d_model,
                d_condition=d_condition,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                num_heads=num_heads,
                use_memory_fusion=(i in self.memory_fusion_layers),
            )
            for i in range(num_layers)
        ])
        
        # 最终归一化
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        states: Optional[List[torch.Tensor]] = None,
        memory_context: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入 [batch, seq_len, d_model]
            condition: 条件向量 [batch, d_condition]
            states: 各层状态列表
            memory_context: 记忆上下文 [batch, num_memories, d_model]
            memory_mask: 记忆掩码
            
        Returns:
            output: 输出 [batch, seq_len, d_model]
            new_states: 新状态列表
        """
        h = x
        new_states = []
        
        for i, block in enumerate(self.blocks):
            state = states[i] if states is not None else None
            
            # 只在指定层传递记忆
            mem = memory_context if i in self.memory_fusion_layers else None
            mask = memory_mask if i in self.memory_fusion_layers else None
            
            h, new_state = block(h, condition, state, mem, mask)
            new_states.append(new_state)
        
        h = self.final_norm(h)
        
        return h, new_states
