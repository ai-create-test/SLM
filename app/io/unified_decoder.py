"""
Unified Decoder - 三通道统一解码器

AMHVQ+ 核心组件：融合语义、结构、符号三通道解码。

解码流程:
    1. 检查是否有结构信息
    2. 有结构 → 骨架检索 → 槽位填充 → 符号锚点强制替换
    3. 无结构 → 纯语义解码
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces.base_module import BaseModule, ModuleOutput, HierarchicalLatent
from ..interfaces.unified_latent import (
    UnifiedLatent,
    StructureRef,
    StructureSlot,
    SymbolAnchors,
    SymbolAnchor,
)
from ..interfaces.registry import Registry
from ..memory.graph_memory import GraphMemory

from .paragraph_decoder import ParagraphDecoder, DecoderOutput


@dataclass
class UnifiedDecoderOutput(ModuleOutput):
    """统一解码器输出"""
    text: Optional[List[str]] = None           # 生成的文本
    token_ids: Optional[torch.Tensor] = None   # Token IDs
    logits: Optional[torch.Tensor] = None      # Logits
    loss: Optional[torch.Tensor] = None        # 损失
    
    # 结构解码信息
    skeleton_used: bool = False                # 是否使用了骨架
    slots_filled: Dict[int, str] = field(default_factory=dict)  # 填充的槽位
    anchors_applied: int = 0                   # 应用的锚点数
    
    decoding_info: Dict[str, Any] = field(default_factory=dict)


class SlotFiller(nn.Module):
    """
    槽位填充器
    
    根据语义条件生成槽位内容。
    """
    
    def __init__(
        self,
        d_latent: int = 512,
        d_model: int = 768,
        vocab_size: int = 50000,
        max_slot_tokens: int = 16,
    ):
        super().__init__()
        
        self.d_latent = d_latent
        self.d_model = d_model
        self.max_slot_tokens = max_slot_tokens
        
        # 槽位类型嵌入
        self.slot_type_embedding = nn.Embedding(32, d_model)  # 32 种槽位类型
        
        # 上下文编码器
        self.context_encoder = nn.Sequential(
            nn.Linear(d_latent + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # 生成头
        self.generator = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, vocab_size),
        )
    
    def forward(
        self,
        semantic_latent: torch.Tensor,
        slot_type_id: int,
        context_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        生成槽位内容
        
        Args:
            semantic_latent: 语义潜向量 [batch, d_latent]
            slot_type_id: 槽位类型 ID
            context_embed: 可选的上下文嵌入 [batch, d_model]
            
        Returns:
            logits: [batch, vocab_size]
        """
        batch_size = semantic_latent.shape[0]
        device = semantic_latent.device
        
        # 获取槽位类型嵌入
        slot_embed = self.slot_type_embedding(
            torch.tensor([slot_type_id], device=device)
        ).expand(batch_size, -1)
        
        # 编码上下文
        if context_embed is not None:
            combined = torch.cat([semantic_latent, context_embed], dim=-1)
        else:
            combined = torch.cat([semantic_latent, slot_embed], dim=-1)
        
        context = self.context_encoder(combined)
        
        # 生成 logits
        logits = self.generator(context + slot_embed)
        
        return logits


class SkeletonAssembler:
    """
    骨架组装器
    
    将填充后的槽位组装成最终输出。
    """
    
    SLOT_TYPE_TO_ID = {
        "identifier": 0,
        "variable": 1,
        "function": 2,
        "class": 3,
        "argument": 4,
        "value": 5,
        "operator": 6,
        "keyword": 7,
        "string": 8,
        "number": 9,
        "Name": 10,
        "Constant": 11,
        "Call": 12,
        "token": 13,
    }
    
    @staticmethod
    def get_slot_type_id(slot_type: str) -> int:
        return SkeletonAssembler.SLOT_TYPE_TO_ID.get(slot_type, 13)
    
    @staticmethod
    def fill_skeleton(
        skeleton_str: str,
        filled_slots: Dict[int, str],
    ) -> str:
        """
        填充骨架字符串
        
        Args:
            skeleton_str: 骨架字符串 (含 □0, □1, ... 占位符)
            filled_slots: slot_id -> 内容 映射
            
        Returns:
            填充后的字符串
        """
        result = skeleton_str
        for slot_id, content in filled_slots.items():
            placeholder = f"□{slot_id}"
            result = result.replace(placeholder, content)
        return result
    
    @staticmethod
    def assemble_from_structure(
        structure_ref: StructureRef,
        filled_slots: Dict[int, str],
    ) -> str:
        """从结构引用组装输出"""
        if structure_ref.skeleton_str:
            return SkeletonAssembler.fill_skeleton(
                structure_ref.skeleton_str,
                filled_slots,
            )
        
        # 如果没有骨架字符串，按槽位顺序拼接
        parts = []
        for slot in sorted(structure_ref.slots, key=lambda s: s.slot_id):
            if slot.slot_id in filled_slots:
                parts.append(filled_slots[slot.slot_id])
        return " ".join(parts)


@Registry.register("decoder", "unified")
class UnifiedDecoder(BaseModule):
    """
    三通道统一解码器 (AMHVQ+)
    
    解码流程:
        UnifiedLatent
            ↓
        检查通道激活状态
            ↓
        ┌─────────────────────────────────────────┐
        │ 有结构通道                │  无结构通道  │
        ├─────────────────────────────────────────┤
        │ 1. 骨架检索               │  纯语义解码  │
        │ 2. 符号锚点填充           │              │
        │ 3. 语义条件生成未锚定槽位 │              │
        │ 4. 组装输出               │              │
        └─────────────────────────────────────────┘
            ↓
          文本输出
    
    使用示例:
        decoder = UnifiedDecoder.from_config(config)
        output = decoder(unified_latent)
        print(output.text)
    """
    
    MODULE_TYPE = "decoder"
    
    def __init__(
        self,
        d_latent: int = 512,
        d_model: int = 768,
        vocab_size: int = 50000,
        max_length: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        # 结构解码参数
        use_structure_decoding: bool = True,
        graph_memory: Optional[GraphMemory] = None,
        # 符号锚点参数
        use_symbol_anchoring: bool = True,
        # 通用参数
        dropout: float = 0.1,
    ):
        """
        Args:
            d_latent: 潜空间维度
            d_model: 解码器隐藏维度
            vocab_size: 词表大小
            max_length: 最大生成长度
            num_layers: Transformer 层数
            num_heads: 注意力头数
            use_structure_decoding: 是否启用结构解码
            graph_memory: GraphMemory 实例 (用于骨架检索)
            use_symbol_anchoring: 是否启用符号锚定
            dropout: Dropout 率
        """
        super().__init__()
        
        self.d_latent = d_latent
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_structure_decoding = use_structure_decoding
        self.use_symbol_anchoring = use_symbol_anchoring
        
        # ========== 语义解码器 (纯语义路径) ==========
        self.semantic_decoder = ParagraphDecoder(
            d_latent=d_latent,
            d_model=d_model,
            vocab_size=vocab_size,
            max_length=max_length,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # ========== 结构解码组件 ==========
        if use_structure_decoding:
            self.slot_filler = SlotFiller(
                d_latent=d_latent,
                d_model=d_model,
                vocab_size=vocab_size,
            )
            self.graph_memory = graph_memory
        else:
            self.slot_filler = None
            self.graph_memory = None
        
        # 潜空间投影 (从 HierarchicalLatent 到单向量)
        self.latent_proj = nn.Linear(d_latent, d_latent)
    
    def forward(
        self,
        latent: Union[UnifiedLatent, HierarchicalLatent, torch.Tensor],
        target_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        emotion: Optional[torch.Tensor] = None,
        scene: Optional[torch.Tensor] = None,
    ) -> UnifiedDecoderOutput:
        """
        三通道解码
        
        Args:
            latent: 潜空间 (UnifiedLatent / HierarchicalLatent / Tensor)
            target_ids: 目标 token IDs (训练时)
            attention_mask: 注意力掩码
            emotion: 情感向量
            scene: 场景向量
            
        Returns:
            UnifiedDecoderOutput
        """
        # 处理不同的输入类型
        if isinstance(latent, UnifiedLatent):
            unified_latent = latent
            hierarchical = latent.semantic
            structure_ref = latent.structure
            symbol_anchors = latent.symbols
        elif isinstance(latent, HierarchicalLatent):
            unified_latent = None
            hierarchical = latent
            structure_ref = None
            symbol_anchors = None
        else:
            # 假设是 Tensor
            unified_latent = None
            hierarchical = None
            structure_ref = None
            symbol_anchors = None
            semantic_vector = latent
        
        # 获取语义向量
        if hierarchical is not None:
            semantic_vector = hierarchical.to_single_vector()  # [batch, d_latent]
        
        semantic_vector = self.latent_proj(semantic_vector)
        
        # 决定解码路径
        use_structure = (
            self.use_structure_decoding
            and structure_ref is not None
            and len(structure_ref.slots) > 0
        )
        
        if use_structure:
            # ========== 结构引导解码 ==========
            return self._decode_with_structure(
                semantic_vector,
                structure_ref,
                symbol_anchors,
                target_ids,
            )
        else:
            # ========== 纯语义解码 ==========
            return self._decode_semantic_only(
                semantic_vector,
                target_ids,
                attention_mask,
                emotion,
                scene,
            )
    
    def _decode_with_structure(
        self,
        semantic_vector: torch.Tensor,
        structure_ref: StructureRef,
        symbol_anchors: Optional[SymbolAnchors],
        target_ids: Optional[torch.Tensor],
    ) -> UnifiedDecoderOutput:
        """结构引导解码"""
        batch_size = semantic_vector.shape[0]
        device = semantic_vector.device
        
        filled_slots: Dict[int, str] = {}
        anchors_applied = 0
        
        # Step 1: 从符号锚点填充
        if self.use_symbol_anchoring and symbol_anchors is not None:
            for anchor in symbol_anchors.anchors:
                if anchor.slot_id is not None and anchor.token_text:
                    filled_slots[anchor.slot_id] = anchor.token_text
                    anchors_applied += 1
        
        # Step 2: 语义条件生成未锚定槽位
        if self.slot_filler is not None:
            for slot in structure_ref.slots:
                if slot.slot_id not in filled_slots:
                    # 生成槽位内容
                    slot_type_id = SkeletonAssembler.get_slot_type_id(slot.slot_type)
                    logits = self.slot_filler(semantic_vector, slot_type_id)
                    
                    # 采样 token
                    probs = F.softmax(logits, dim=-1)
                    token_id = torch.argmax(probs, dim=-1).item()
                    
                    # 转换为文本 (简化: 使用占位符)
                    filled_slots[slot.slot_id] = f"<gen_{slot.slot_id}>"
        
        # Step 3: 组装输出
        assembled_text = SkeletonAssembler.assemble_from_structure(
            structure_ref,
            filled_slots,
        )
        
        return UnifiedDecoderOutput(
            data=semantic_vector,
            text=[assembled_text] * batch_size,
            skeleton_used=True,
            slots_filled=filled_slots,
            anchors_applied=anchors_applied,
            decoding_info={
                "mode": "structure_guided",
                "total_slots": len(structure_ref.slots),
                "anchored_slots": anchors_applied,
                "generated_slots": len(filled_slots) - anchors_applied,
            },
        )
    
    def _decode_semantic_only(
        self,
        semantic_vector: torch.Tensor,
        target_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        emotion: Optional[torch.Tensor],
        scene: Optional[torch.Tensor],
    ) -> UnifiedDecoderOutput:
        """纯语义解码"""
        # 使用原始 ParagraphDecoder
        decoder_output = self.semantic_decoder(
            latent=semantic_vector,
            target_ids=target_ids,
            attention_mask=attention_mask,
            emotion=emotion,
            scene=scene,
        )
        
        return UnifiedDecoderOutput(
            data=decoder_output.data,
            text=decoder_output.text,
            token_ids=None,
            logits=decoder_output.logits,
            loss=decoder_output.loss,
            skeleton_used=False,
            slots_filled={},
            anchors_applied=0,
            decoding_info={
                "mode": "semantic_only",
            },
        )
    
    def generate(
        self,
        latent: Union[UnifiedLatent, HierarchicalLatent, torch.Tensor],
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        emotion: Optional[torch.Tensor] = None,
        scene: Optional[torch.Tensor] = None,
    ) -> UnifiedDecoderOutput:
        """
        生成文本
        
        Args:
            latent: 潜空间
            max_length: 最大生成长度
            temperature: 采样温度
            top_p: nucleus 采样阈值
            top_k: top-k 采样
            emotion: 情感向量
            scene: 场景向量
            
        Returns:
            UnifiedDecoderOutput
        """
        # 检查是否可以使用结构解码
        if isinstance(latent, UnifiedLatent) and latent.has_structure:
            # 使用结构引导生成
            return self.forward(latent, emotion=emotion, scene=scene)
        
        # 获取语义向量
        if isinstance(latent, UnifiedLatent):
            semantic_vector = latent.semantic.to_single_vector()
        elif isinstance(latent, HierarchicalLatent):
            semantic_vector = latent.to_single_vector()
        else:
            semantic_vector = latent
        
        semantic_vector = self.latent_proj(semantic_vector)
        
        # 使用语义解码器生成
        decoder_output = self.semantic_decoder.generate(
            latent=semantic_vector,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            emotion=emotion,
            scene=scene,
        )
        
        return UnifiedDecoderOutput(
            data=decoder_output.data,
            text=decoder_output.text,
            logits=decoder_output.logits,
            skeleton_used=False,
            decoding_info={"mode": "semantic_generation"},
        )
    
    def set_graph_memory(self, graph_memory: GraphMemory) -> None:
        """设置 GraphMemory (用于骨架检索)"""
        self.graph_memory = graph_memory
    
    @classmethod
    def from_config(cls, config, graph_memory: Optional[GraphMemory] = None, **kwargs) -> "UnifiedDecoder":
        """从配置创建实例"""
        # 从 kwargs 中移除已经显式设置的参数以避免重复
        use_structure_decoding = kwargs.pop('use_structure_decoding', getattr(config, 'use_three_channel', True))
        use_symbol_anchoring = kwargs.pop('use_symbol_anchoring', getattr(config, 'use_three_channel', True))
        kwargs.pop('graph_memory', None)  # 已经作为参数传入
        
        return cls(
            d_latent=getattr(config, 'd_latent', 512),
            d_model=getattr(config, 'd_model', 768),
            vocab_size=getattr(config, 'vocab_size', 50000),
            max_length=getattr(config, 'max_length', 256),
            num_layers=getattr(config, 'decoder_layers', 6),
            num_heads=getattr(config, 'num_heads', 8),
            use_structure_decoding=use_structure_decoding,
            use_symbol_anchoring=use_symbol_anchoring,
            dropout=getattr(config, 'dropout', 0.1),
            graph_memory=graph_memory,
            **kwargs,
        )


# ============================================================
# 工厂方法
# ============================================================

def create_unified_decoder(
    config,
    use_three_channel: Optional[bool] = None,
    graph_memory: Optional[GraphMemory] = None,
    **kwargs,
) -> BaseModule:
    """
    工厂方法：创建统一解码器
    
    Args:
        config: ModelConfig
        use_three_channel: 是否启用三通道 (None 则从 config 读取)
        graph_memory: GraphMemory 实例
        **kwargs: 额外参数
        
    Returns:
        解码器实例
    """
    # 从 kwargs 中移除以避免重复传参
    kwargs.pop('use_three_channel', None)
    kwargs.pop('graph_memory', None)
    kwargs.pop('use_structure_decoding', None)
    kwargs.pop('use_symbol_anchoring', None)
    
    if use_three_channel is None:
        use_three_channel = getattr(config, 'use_three_channel', True)
    
    if use_three_channel:
        return UnifiedDecoder.from_config(
            config,
            graph_memory=graph_memory,
            **kwargs,
        )
    else:
        # 退化为纯语义解码
        return UnifiedDecoder.from_config(
            config,
            graph_memory=None,
            use_structure_decoding=False,
            use_symbol_anchoring=False,
            **kwargs,
        )
