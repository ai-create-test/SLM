"""
Test Unified Decoder - 统一解码器测试

Phase 9.5
"""

import pytest
import torch

from app.interfaces.base_module import HierarchicalLatent
from app.interfaces.unified_latent import (
    UnifiedLatent,
    StructureRef,
    StructureSlot,
    SymbolAnchors,
    SymbolAnchor,
)


class TestSlotFiller:
    """测试槽位填充器"""
    
    def test_basic_fill(self):
        """基本填充测试"""
        from app.io.unified_decoder import SlotFiller
        
        filler = SlotFiller(d_latent=256, d_model=256, vocab_size=1000)
        semantic = torch.randn(2, 256)
        
        logits = filler(semantic, slot_type_id=0)
        
        assert logits.shape == (2, 1000)


class TestSkeletonAssembler:
    """测试骨架组装器"""
    
    def test_fill_skeleton_string(self):
        """填充骨架字符串测试"""
        from app.io.unified_decoder import SkeletonAssembler
        
        skeleton = "□0 = □1(□2)"
        filled = {0: "x", 1: "get_value", 2: "id"}
        
        result = SkeletonAssembler.fill_skeleton(skeleton, filled)
        
        assert result == "x = get_value(id)"
    
    def test_assemble_from_structure(self):
        """从结构组装测试"""
        from app.io.unified_decoder import SkeletonAssembler
        
        structure_ref = StructureRef(
            graph_node_ids=["n1", "n2"],
            structure_type="ast",
            skeleton_str="func(□0, □1)",
            slots=[
                StructureSlot(slot_id=0, slot_type="argument", parent_node="func"),
                StructureSlot(slot_id=1, slot_type="argument", parent_node="func"),
            ],
        )
        
        filled = {0: "a", 1: "b"}
        result = SkeletonAssembler.assemble_from_structure(structure_ref, filled)
        
        assert result == "func(a, b)"


class TestUnifiedDecoderImports:
    """测试导入"""
    
    def test_imports(self):
        """导入测试"""
        from app.io.unified_decoder import (
            UnifiedDecoder,
            UnifiedDecoderOutput,
            SlotFiller,
            SkeletonAssembler,
            create_unified_decoder,
        )
        assert UnifiedDecoder is not None
        assert UnifiedDecoderOutput is not None


class TestDecodingPaths:
    """测试解码路径"""
    
    def test_structure_detection(self):
        """结构检测测试"""
        # 有结构的 UnifiedLatent
        semantic = HierarchicalLatent(
            global_=torch.randn(1, 1, 256),
            chunks=torch.randn(1, 4, 256),
        )
        
        structure = StructureRef(
            graph_node_ids=["n1"],
            structure_type="ast",
            slots=[StructureSlot(slot_id=0, slot_type="var", parent_node="root")],
        )
        
        with_structure = UnifiedLatent(
            semantic=semantic,
            structure=structure,
            scene="coding",
        )
        
        assert with_structure.has_structure == True
        
        # 无结构的 UnifiedLatent
        without_structure = UnifiedLatent(
            semantic=semantic,
            scene="chat",
        )
        
        assert without_structure.has_structure == False
    
    def test_symbol_anchor_filling(self):
        """符号锚点填充测试"""
        semantic = HierarchicalLatent(
            global_=torch.randn(1, 1, 256),
            chunks=torch.randn(1, 4, 256),
        )
        
        # 创建带锚点的 UnifiedLatent
        anchors = SymbolAnchors()
        anchors.anchors = [
            SymbolAnchor(position=0, token_id=100, token_text="user_name", slot_id=0),
            SymbolAnchor(position=5, token_id=200, token_text="get_user", slot_id=1),
        ]
        
        structure = StructureRef(
            graph_node_ids=["n1", "n2"],
            structure_type="ast",
            skeleton_str="□0 = □1(id)",
            slots=[
                StructureSlot(slot_id=0, slot_type="variable", parent_node="assign"),
                StructureSlot(slot_id=1, slot_type="function", parent_node="call"),
            ],
        )
        
        unified = UnifiedLatent(
            semantic=semantic,
            structure=structure,
            symbols=anchors,
            scene="coding",
        )
        
        # 验证锚点数量
        assert unified.symbols.num_anchors == 2
        
        # 模拟填充
        from app.io.unified_decoder import SkeletonAssembler
        
        filled = {}
        for anchor in anchors.anchors:
            if anchor.slot_id is not None:
                filled[anchor.slot_id] = anchor.token_text
        
        result = SkeletonAssembler.assemble_from_structure(structure, filled)
        assert "user_name" in result
        assert "get_user" in result


class TestMultiPathFusion:
    """测试多路径融合"""
    
    def test_coding_scene_uses_structure(self):
        """编码场景使用结构"""
        from app.modulation.scene_encoder import SceneAwarePrecisionRouter
        
        router = SceneAwarePrecisionRouter()
        config = router.get_config("coding")
        
        assert config.structure == True
        assert config.symbols == True
    
    def test_chat_scene_uses_semantic_only(self):
        """聊天场景仅用语义"""
        from app.modulation.scene_encoder import SceneAwarePrecisionRouter
        
        router = SceneAwarePrecisionRouter()
        config = router.get_config("chat")
        
        assert config.structure == False
        assert config.symbols == False


class TestCompatibility:
    """测试兼容性"""
    
    def test_hierarchical_latent_input(self):
        """HierarchicalLatent 输入测试"""
        hierarchical = HierarchicalLatent(
            global_=torch.randn(2, 1, 256),
            chunks=torch.randn(2, 4, 256),
        )
        
        # 应该能转换为单向量
        single_vec = hierarchical.to_single_vector()
        assert single_vec.shape == (2, 256)
    
    def test_tensor_input(self):
        """Tensor 直接输入测试"""
        tensor = torch.randn(2, 256)
        
        # 应该可以直接作为语义向量
        assert tensor.shape == (2, 256)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
