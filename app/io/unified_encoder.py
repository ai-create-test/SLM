"""
Unified Encoder - 三通道统一编码器

AMHVQ+ 核心组件：融合语义、结构、符号三通道编码。

编码流程:
    1. 场景路由 → 决定激活哪些通道
    2. 语义通道 (始终激活): HierarchicalParagraphEncoder → HierarchicalLatent
    3. 结构通道 (场景触发): StructureParser → GraphMemory → StructureRef
    4. 符号通道 (场景触发): SymbolAnchorEncoder → SymbolAnchors
    5. 打包为 UnifiedLatent
"""

from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any, Tuple
import torch
import torch.nn as nn

from ..interfaces.base_module import BaseModule, ModuleOutput, HierarchicalLatent
from ..interfaces.unified_latent import (
    UnifiedLatent,
    StructureRef,
    StructureSlot,
    SymbolAnchors,
    SymbolAnchor,
    PrecisionConfig,
)
from ..interfaces.registry import Registry
from ..memory.graph_memory import GraphMemory

from .paragraph_encoder import HierarchicalParagraphEncoder, HierarchicalEncoderOutput
from .structure_parser import (
    parse_structure,
    StructureGraph,
    StructureSummaryEncoder,
    get_structure_parser,
)
from .symbol_anchor import SymbolAnchorEncoder, detect_critical_tokens
from ..modulation.scene_encoder import SceneAwarePrecisionRouter


@dataclass
class UnifiedEncoderOutput(ModuleOutput):
    """统一编码器输出"""
    unified_latent: UnifiedLatent = None       # 统一潜空间
    hierarchical_output: HierarchicalEncoderOutput = None  # 语义通道原始输出
    structure_graph: Optional[StructureGraph] = None       # 结构图 (调试用)
    encoding_info: Dict[str, Any] = None       # 编码信息


@Registry.register("encoder", "unified")
class UnifiedEncoder(BaseModule):
    """
    三通道统一编码器 (AMHVQ+)
    
    完整流程:
        Text 
          ↓
        Scene Router → 决定通道激活
          ↓
        ┌─────────────────────────────────────────┐
        │ 语义通道     │  结构通道    │  符号通道   │
        │ (始终激活)   │  (场景触发)  │  (场景触发)  │
        ├─────────────────────────────────────────┤
        │ Hierarchical │ Structure   │  Symbol     │
        │ Encoder      │ Parser +    │  Anchor     │
        │              │ GraphMemory │  Encoder    │
        └─────────────────────────────────────────┘
          ↓                ↓               ↓
                     UnifiedLatent
    
    使用示例:
        encoder = UnifiedEncoder.from_config(config)
        output = encoder("def hello(): pass", scene="coding")
        latent = output.unified_latent
        
        # 检查激活的通道
        print(latent.has_structure)  # True
        print(latent.has_symbols)    # True
    """
    
    MODULE_TYPE = "encoder"
    
    def __init__(
        self,
        d_model: int = 768,
        d_latent: int = 512,
        d_structure: int = 256,
        d_symbol: int = 128,
        # Hierarchical Encoder 参数
        max_chunks: int = 8,
        matryoshka_dims: List[int] = None,
        rvq_layers: int = 3,
        # 结构通道参数
        structure_parser_type: str = "auto",
        use_graph_memory: bool = True,
        graph_memory_d_node: int = 256,
        # 符号通道参数
        symbol_threshold: float = 0.5,
        max_symbol_anchors: int = 32,
        # 通用参数
        use_three_channel: bool = True,
        base_model_name: str = "bert-base-uncased",
        freeze_base: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: 模型维度
            d_latent: 潜空间维度
            d_structure: 结构向量维度
            d_symbol: 符号向量维度
            max_chunks: 最大语义块数
            matryoshka_dims: Matryoshka 嵌套维度
            rvq_layers: RVQ 层数
            structure_parser_type: 结构解析器类型
            use_graph_memory: 是否使用 GraphMemory 存储结构
            graph_memory_d_node: GraphMemory 节点维度
            symbol_threshold: 关键 token 检测阈值
            max_symbol_anchors: 最大锚点数
            use_three_channel: 是否启用三通道 (False 则退化为纯语义)
            base_model_name: 基础语言模型
            freeze_base: 是否冻结基础模型
            dropout: Dropout 率
        """
        super().__init__()
        
        self.use_three_channel = use_three_channel
        self.d_model = d_model
        self.d_latent = d_latent
        self.d_structure = d_structure
        self.d_symbol = d_symbol
        
        # ========== 场景精度路由 ==========
        self.precision_router = SceneAwarePrecisionRouter(
            d_model=d_model,
            use_auto_detection=True,
        )
        
        # ========== 通道 1: 语义编码器 (始终存在) ==========
        self.semantic_encoder = HierarchicalParagraphEncoder(
            d_model=d_model,
            d_latent=d_latent,
            max_chunks=max_chunks,
            matryoshka_dims=matryoshka_dims or [64, 128, 256, d_latent],
            rvq_layers=rvq_layers,
            base_model_name=base_model_name,
            freeze_base=freeze_base,
            dropout=dropout,
        )
        
        # ========== 通道 2: 结构编码器 (可选) ==========
        if use_three_channel:
            self.structure_parser_type = structure_parser_type
            self.structure_summary_encoder = StructureSummaryEncoder(
                d_model=d_structure,
                max_nodes=32,
            )
            
            if use_graph_memory:
                self.graph_memory = GraphMemory(d_node=graph_memory_d_node)
            else:
                self.graph_memory = None
        else:
            self.structure_summary_encoder = None
            self.graph_memory = None
        
        # ========== 通道 3: 符号编码器 (可选) ==========
        if use_three_channel:
            self.symbol_encoder = SymbolAnchorEncoder(
                d_model=d_model,
                d_output=d_symbol,
                max_anchors=max_symbol_anchors,
                threshold=symbol_threshold,
            )
        else:
            self.symbol_encoder = None
        
        # 结构 ID 计数器
        self._structure_counter = 0
    
    def forward(
        self,
        text: Union[str, List[str]],
        scene: Optional[str] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> UnifiedEncoderOutput:
        """
        三通道编码
        
        Args:
            text: 输入文本
            scene: 场景 (None 则自动检测)
            attention_mask: 注意力掩码
            
        Returns:
            UnifiedEncoderOutput
        """
        # 处理单文本
        if isinstance(text, str):
            text_list = [text]
            single_input = True
        else:
            text_list = text
            single_input = False
        
        # 1. 场景路由
        if scene is None and self.use_three_channel:
            scene = self.precision_router.detect_scene(text_list[0])
        scene = scene or "chat"
        
        precision_config = self.precision_router.get_config(scene)
        
        # 2. 通道 1: 语义编码 (始终激活)
        semantic_output = self.semantic_encoder(text_list, attention_mask)
        hierarchical_latent = semantic_output.hierarchical_latent
        
        # 3. 通道 2: 结构编码 (条件激活)
        structure_ref = None
        structure_graph = None
        
        if self.use_three_channel and precision_config.structure and self.structure_summary_encoder is not None:
            structure_ref, structure_graph = self._encode_structure(
                text_list[0],
                hierarchical_latent,
            )
        
        # 4. 通道 3: 符号编码 (条件激活)
        symbol_anchors = None
        
        if self.use_three_channel and precision_config.symbols and self.symbol_encoder is not None:
            symbol_anchors = self._encode_symbols(
                text_list[0],
                semantic_output,
            )
        
        # 5. 打包为 UnifiedLatent
        unified_latent = UnifiedLatent(
            semantic=hierarchical_latent,
            structure=structure_ref,
            symbols=symbol_anchors,
            scene=scene,
            precision_config=precision_config,
            original_text=text_list[0] if single_input else "",
            metadata={
                "num_texts": len(text_list),
                "channels_active": {
                    "semantic": True,
                    "structure": structure_ref is not None,
                    "symbols": symbol_anchors is not None and symbol_anchors.num_anchors > 0,
                }
            },
        )
        
        return UnifiedEncoderOutput(
            data=semantic_output.data,
            unified_latent=unified_latent,
            hierarchical_output=semantic_output,
            structure_graph=structure_graph,
            encoding_info={
                "scene": scene,
                "precision_config": {
                    "semantic": precision_config.semantic,
                    "structure": precision_config.structure,
                    "symbols": precision_config.symbols,
                },
                "structure_nodes": len(structure_graph.nodes) if structure_graph else 0,
                "symbol_anchors": symbol_anchors.num_anchors if symbol_anchors else 0,
            },
        )
    
    def _encode_structure(
        self,
        text: str,
        hierarchical_latent: HierarchicalLatent,
    ) -> Tuple[Optional[StructureRef], Optional[StructureGraph]]:
        """编码结构通道"""
        try:
            # 解析结构
            graph = parse_structure(text, parser_type=self.structure_parser_type)
            
            if not graph.nodes:
                return None, graph
            
            # 计算结构摘要向量
            summary_vector = self.structure_summary_encoder(graph)
            
            # 存储到 GraphMemory (如果启用)
            structure_id = f"struct_{self._structure_counter}"
            self._structure_counter += 1
            
            graph_node_ids = []
            if self.graph_memory is not None:
                # 准备节点数据
                nodes_data = []
                for node_id, node in graph.nodes.items():
                    nodes_data.append({
                        "id": node_id,
                        "type": node.node_type,
                        "content": node.content,
                        "slot_id": node.slot_id,
                        "span": node.span,
                    })
                
                # 准备边数据
                edges_data = [(e[0], e[1], e[2]) for e in graph.edges]
                
                # 存储
                graph_node_ids = self.graph_memory.store_structure(
                    structure_id=structure_id,
                    nodes=nodes_data,
                    edges=edges_data,
                    summary_vector=summary_vector,
                    structure_type=graph.structure_type,
                )
            
            # 提取槽位
            slots = []
            for slot_id, slot_type, parent_type in graph.get_slots():
                slots.append(StructureSlot(
                    slot_id=slot_id,
                    slot_type=slot_type,
                    parent_node=parent_type,
                ))
            
            # 创建 StructureRef
            structure_ref = StructureRef(
                graph_node_ids=graph_node_ids,
                structure_type=graph.structure_type,
                summary_vector=summary_vector.unsqueeze(0) if summary_vector.dim() == 1 else summary_vector,
                slots=slots,
                skeleton_str=graph.to_skeleton_string(),
            )
            
            return structure_ref, graph
            
        except Exception as e:
            # 结构解析失败时返回 None
            return None, None
    
    def _encode_symbols(
        self,
        text: str,
        semantic_output: HierarchicalEncoderOutput,
    ) -> Optional[SymbolAnchors]:
        """编码符号通道"""
        try:
            # 使用规则检测关键 token
            critical_tokens = detect_critical_tokens(text)
            
            if not critical_tokens:
                return SymbolAnchors()
            
            # 创建锚点
            anchors = SymbolAnchors()
            for i, (pos, token_text, token_type) in enumerate(critical_tokens):
                anchor = SymbolAnchor(
                    position=pos,
                    token_id=hash(token_text) % 100000,  # 简化: 用 hash 作为 token_id
                    token_text=token_text,
                    slot_id=i,
                    anchor_type=token_type,
                    is_critical=True,
                )
                anchors.anchors.append(anchor)
            
            # 计算符号向量 (使用语义向量的一部分)
            if semantic_output.data is not None:
                anchors.vector = semantic_output.data.mean() * torch.ones(self.d_symbol)
            
            return anchors
            
        except Exception as e:
            return SymbolAnchors()
    
    def encode_text(self, text: Union[str, List[str]], scene: str = None) -> UnifiedLatent:
        """便捷方法：直接返回 UnifiedLatent"""
        output = self.forward(text, scene=scene)
        return output.unified_latent
    
    def get_loss(self, output: UnifiedEncoderOutput) -> torch.Tensor:
        """获取编码器损失"""
        # 主要是语义通道的 RVQ commitment loss
        if output.hierarchical_output is not None:
            return self.semantic_encoder.get_loss(output.hierarchical_output)
        return torch.tensor(0.0, device=self.device)
    
    @classmethod
    def from_config(cls, config, **kwargs) -> "UnifiedEncoder":
        """从配置创建实例"""
        # 从 kwargs 中移除已经显式设置的参数以避免重复
        use_three_channel = kwargs.pop('use_three_channel', getattr(config, 'use_three_channel', True))
        
        return cls(
            d_model=getattr(config, 'd_model', 768),
            d_latent=getattr(config, 'd_latent', 512),
            max_chunks=getattr(config, 'max_chunks', 8),
            matryoshka_dims=getattr(config, 'matryoshka_dims', None),
            rvq_layers=getattr(config, 'rvq_layers', 3),
            structure_parser_type=getattr(config, 'structure_parser_type', 'auto'),
            use_graph_memory=getattr(config, 'use_graph_memory_for_structure', True),
            symbol_threshold=getattr(config, 'symbol_anchor_threshold', 0.5),
            max_symbol_anchors=getattr(config, 'max_symbol_anchors', 32),
            use_three_channel=use_three_channel,
            dropout=getattr(config, 'dropout', 0.1),
            **kwargs,
        )


# ============================================================
# 工厂方法更新
# ============================================================

def create_unified_encoder(
    config,
    use_three_channel: Optional[bool] = None,
    **kwargs,
) -> BaseModule:
    """
    工厂方法：创建统一编码器
    
    Args:
        config: ModelConfig
        use_three_channel: 是否启用三通道 (None 则从 config 读取)
        **kwargs: 额外参数
        
    Returns:
        编码器实例
    """
    # 从 kwargs 中移除以避免重复传参
    kwargs.pop('use_three_channel', None)
    
    if use_three_channel is None:
        use_three_channel = getattr(config, 'use_three_channel', True)
    
    if use_three_channel:
        return UnifiedEncoder.from_config(config, use_three_channel=True, **kwargs)
    else:
        # 退化为纯语义模式
        return UnifiedEncoder.from_config(config, use_three_channel=False, **kwargs)
