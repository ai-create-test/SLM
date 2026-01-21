"""
Unified Latent - 三通道统一潜空间表示

AMHVQ+ 核心数据结构，融合：
- Semantic Channel: 语义通道 (HierarchicalLatent)
- Structure Channel: 结构通道 (AST/句法骨架引用)
- Symbol Channel: 符号通道 (精确 token 锚点)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
import torch

from .base_module import HierarchicalLatent, LatentVector


@dataclass
class SymbolAnchor:
    """
    符号锚点 - 精确 token 存储
    
    用于保存需要精确还原的 token 信息。
    """
    position: int              # 在原文中的 token 位置
    token_id: int              # 精确的 token ID
    token_text: str = ""       # 原始文本 (可选,用于调试)
    slot_id: Optional[int] = None  # 对应骨架中的槽位 ID
    is_critical: bool = True   # 是否为关键符号 (变量名、函数名等)
    
    def __repr__(self) -> str:
        return f"Anchor(pos={self.position}, id={self.token_id}, text='{self.token_text}')"


@dataclass
class SymbolAnchors:
    """
    符号锚点集合
    """
    anchors: List[SymbolAnchor] = field(default_factory=list)
    vector: Optional[torch.Tensor] = None  # 锚点的向量编码 [batch, d_symbol]
    
    @property
    def num_anchors(self) -> int:
        return len(self.anchors)
    
    def get_token_ids(self) -> List[int]:
        """返回所有锚点的 token ID"""
        return [a.token_id for a in self.anchors]
    
    def get_by_slot(self, slot_id: int) -> Optional[SymbolAnchor]:
        """根据槽位 ID 获取锚点"""
        for anchor in self.anchors:
            if anchor.slot_id == slot_id:
                return anchor
        return None
    
    def to(self, device: torch.device) -> "SymbolAnchors":
        return SymbolAnchors(
            anchors=self.anchors,
            vector=self.vector.to(device) if self.vector is not None else None,
        )


@dataclass
class StructureSlot:
    """
    结构槽位 - 骨架中待填充的位置
    """
    slot_id: int               # 槽位 ID
    slot_type: str             # 类型 (identifier, literal, expression, etc.)
    parent_node: str           # 父节点类型
    context: str = ""          # 上下文信息
    
    def __repr__(self) -> str:
        return f"Slot({self.slot_id}, type={self.slot_type})"


@dataclass
class StructureRef:
    """
    结构引用 - 指向 GraphMemory 中存储的结构
    
    不直接存储完整结构，而是存储引用 + 摘要向量。
    """
    graph_node_ids: List[str]  # GraphMemory 中的节点 ID 列表
    structure_type: str        # 结构类型 (ast, syntax_tree, etc.)
    summary_vector: Optional[torch.Tensor] = None  # 结构摘要向量 [batch, d_structure]
    slots: List[StructureSlot] = field(default_factory=list)  # 槽位列表
    skeleton_str: str = ""     # 骨架字符串表示 (用于调试)
    
    @property
    def num_slots(self) -> int:
        return len(self.slots)
    
    def get_slot_ids(self) -> List[int]:
        return [s.slot_id for s in self.slots]
    
    def to(self, device: torch.device) -> "StructureRef":
        return StructureRef(
            graph_node_ids=self.graph_node_ids,
            structure_type=self.structure_type,
            summary_vector=self.summary_vector.to(device) if self.summary_vector is not None else None,
            slots=self.slots,
            skeleton_str=self.skeleton_str,
        )


@dataclass
class PrecisionConfig:
    """
    精度配置 - 控制各通道激活
    """
    semantic: bool = True      # 语义通道 (始终激活)
    structure: bool = False    # 结构通道
    symbols: bool = False      # 符号通道
    
    @classmethod
    def for_scene(cls, scene: str) -> "PrecisionConfig":
        """根据场景返回配置"""
        configs = {
            "chat": cls(semantic=True, structure=False, symbols=False),
            "coding": cls(semantic=True, structure=True, symbols=True),
            "creative": cls(semantic=True, structure=False, symbols=False),
            "debate": cls(semantic=True, structure=True, symbols=False),
            "analysis": cls(semantic=True, structure=True, symbols=False),
            "teaching": cls(semantic=True, structure=False, symbols=False),
            "roleplay": cls(semantic=True, structure=False, symbols=False),
            "formal": cls(semantic=True, structure=True, symbols=True),
        }
        return configs.get(scene, cls())
    
    def all_channels(self) -> bool:
        """是否全部通道激活"""
        return self.semantic and self.structure and self.symbols
    
    def semantic_only(self) -> bool:
        """是否仅语义通道"""
        return self.semantic and not self.structure and not self.symbols


@dataclass
class UnifiedLatent:
    """
    统一潜空间表示 (AMHVQ+)
    
    三通道融合：
    - semantic: 语义通道 (HierarchicalLatent) - 捕捉意图和含义
    - structure: 结构通道 (StructureRef) - 捕捉骨架和逻辑
    - symbols: 符号通道 (SymbolAnchors) - 捕捉精确 token
    
    使用示例:
        # 编码
        latent = unified_encoder(text, scene="coding")
        
        # 检查通道
        if latent.has_structure:
            skeleton = latent.structure.skeleton_str
        
        # 解码
        output = unified_decoder(latent)
    """
    # 核心三通道
    semantic: HierarchicalLatent                    # 语义通道 (必需)
    structure: Optional[StructureRef] = None        # 结构通道 (可选)
    symbols: Optional[SymbolAnchors] = None         # 符号通道 (可选)
    
    # 元数据
    scene: str = "chat"                             # 场景标识
    precision_config: PrecisionConfig = field(default_factory=PrecisionConfig)
    original_text: str = ""                         # 原始文本 (用于验证)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_structure(self) -> bool:
        return self.structure is not None
    
    @property
    def has_symbols(self) -> bool:
        return self.symbols is not None and self.symbols.num_anchors > 0
    
    @property
    def semantic_only(self) -> bool:
        """是否仅语义通道激活"""
        return not self.has_structure and not self.has_symbols
    
    @property
    def num_semantic_tokens(self) -> int:
        return self.semantic.num_tokens
    
    @property
    def device(self) -> torch.device:
        return self.semantic.device
    
    def to(self, device: torch.device) -> "UnifiedLatent":
        """移动到指定设备"""
        return UnifiedLatent(
            semantic=self.semantic.to(device),
            structure=self.structure.to(device) if self.structure is not None else None,
            symbols=self.symbols.to(device) if self.symbols is not None else None,
            scene=self.scene,
            precision_config=self.precision_config,
            original_text=self.original_text,
            metadata=self.metadata,
        )
    
    def to_hierarchical(self) -> HierarchicalLatent:
        """提取语义通道 (兼容接口)"""
        return self.semantic
    
    def to_legacy(self) -> LatentVector:
        """转换为旧版 LatentVector (兼容接口)"""
        return self.semantic.to_legacy()
    
    @classmethod
    def from_hierarchical(
        cls,
        hierarchical: HierarchicalLatent,
        scene: str = "chat",
    ) -> "UnifiedLatent":
        """从 HierarchicalLatent 创建 (纯语义模式)"""
        return cls(
            semantic=hierarchical,
            scene=scene,
            precision_config=PrecisionConfig.for_scene(scene),
        )
    
    @classmethod
    def from_legacy(
        cls,
        legacy: LatentVector,
        scene: str = "chat",
    ) -> "UnifiedLatent":
        """从旧版 LatentVector 创建 (兼容模式)"""
        hierarchical = HierarchicalLatent.from_single_vector(legacy.vector)
        return cls.from_hierarchical(hierarchical, scene)
    
    def get_reconstruction_info(self) -> Dict[str, Any]:
        """获取重建所需的全部信息"""
        info = {
            "has_structure": self.has_structure,
            "has_symbols": self.has_symbols,
            "num_semantic_tokens": self.num_semantic_tokens,
            "scene": self.scene,
        }
        if self.has_structure:
            info["num_slots"] = self.structure.num_slots
        if self.has_symbols:
            info["num_anchors"] = self.symbols.num_anchors
        return info
    
    def __repr__(self) -> str:
        channels = ["semantic"]
        if self.has_structure:
            channels.append("structure")
        if self.has_symbols:
            channels.append(f"symbols({self.symbols.num_anchors})")
        return f"UnifiedLatent(scene={self.scene}, channels={channels})"


# ============================================================
# 工具函数
# ============================================================

def detect_latent_type(latent: Any) -> str:
    """检测潜空间类型"""
    if isinstance(latent, UnifiedLatent):
        return "unified"
    elif isinstance(latent, HierarchicalLatent):
        return "hierarchical"
    elif isinstance(latent, LatentVector):
        return "legacy"
    elif isinstance(latent, torch.Tensor):
        return "tensor"
    else:
        return "unknown"


def to_unified(
    latent: Union[UnifiedLatent, HierarchicalLatent, LatentVector, torch.Tensor],
    scene: str = "chat",
) -> UnifiedLatent:
    """
    将任意潜空间类型转换为 UnifiedLatent
    
    Args:
        latent: 输入潜空间 (任意类型)
        scene: 场景标识
        
    Returns:
        UnifiedLatent
    """
    latent_type = detect_latent_type(latent)
    
    if latent_type == "unified":
        return latent
    elif latent_type == "hierarchical":
        return UnifiedLatent.from_hierarchical(latent, scene)
    elif latent_type == "legacy":
        return UnifiedLatent.from_legacy(latent, scene)
    elif latent_type == "tensor":
        # 假设是 [batch, d_latent] 的单向量
        legacy = LatentVector(vector=latent)
        return UnifiedLatent.from_legacy(legacy, scene)
    else:
        raise ValueError(f"Unknown latent type: {type(latent)}")


def to_hierarchical(
    latent: Union[UnifiedLatent, HierarchicalLatent, LatentVector, torch.Tensor],
) -> HierarchicalLatent:
    """
    将任意潜空间类型转换为 HierarchicalLatent
    """
    latent_type = detect_latent_type(latent)
    
    if latent_type == "unified":
        return latent.semantic
    elif latent_type == "hierarchical":
        return latent
    elif latent_type == "legacy":
        return HierarchicalLatent.from_single_vector(latent.vector)
    elif latent_type == "tensor":
        return HierarchicalLatent.from_single_vector(latent)
    else:
        raise ValueError(f"Unknown latent type: {type(latent)}")


def to_legacy(
    latent: Union[UnifiedLatent, HierarchicalLatent, LatentVector, torch.Tensor],
) -> LatentVector:
    """
    将任意潜空间类型转换为 LatentVector (向后兼容)
    """
    latent_type = detect_latent_type(latent)
    
    if latent_type == "unified":
        return latent.to_legacy()
    elif latent_type == "hierarchical":
        return latent.to_legacy()
    elif latent_type == "legacy":
        return latent
    elif latent_type == "tensor":
        return LatentVector(vector=latent)
    else:
        raise ValueError(f"Unknown latent type: {type(latent)}")
