"""
Test Unified Encoder - 统一编码器测试

Phase 8.4
"""

import pytest
import torch

from app.interfaces.unified_latent import UnifiedLatent, PrecisionConfig


class TestUnifiedEncoderBasic:
    """基础测试 (不依赖模型加载)"""
    
    def test_imports(self):
        """导入测试"""
        from app.io.unified_encoder import (
            UnifiedEncoder,
            UnifiedEncoderOutput,
            create_unified_encoder,
        )
        assert UnifiedEncoder is not None
    
    def test_precision_router_integration(self):
        """精度路由集成测试"""
        from app.modulation.scene_encoder import SceneAwarePrecisionRouter
        
        router = SceneAwarePrecisionRouter(use_auto_detection=False)
        
        # 测试不同场景的配置
        coding_config = router.get_config("coding")
        assert coding_config.structure == True
        assert coding_config.symbols == True
        
        chat_config = router.get_config("chat")
        assert chat_config.structure == False
        assert chat_config.symbols == False


class TestUnifiedLatentPackaging:
    """UnifiedLatent 打包测试"""
    
    def test_create_unified_latent(self):
        """创建 UnifiedLatent 测试"""
        from app.interfaces.base_module import HierarchicalLatent
        from app.interfaces.unified_latent import (
            UnifiedLatent,
            StructureRef,
            SymbolAnchors,
            SymbolAnchor,
        )
        
        # 创建语义通道
        semantic = HierarchicalLatent(
            global_=torch.randn(1, 1, 256),
            chunks=torch.randn(1, 4, 256),
        )
        
        # 创建结构通道
        structure = StructureRef(
            graph_node_ids=["node_1", "node_2"],
            structure_type="ast",
            summary_vector=torch.randn(1, 128),
        )
        
        # 创建符号通道
        symbols = SymbolAnchors()
        symbols.anchors.append(SymbolAnchor(
            position=0,
            token_id=100,
            token_text="user_name",
        ))
        
        # 打包
        unified = UnifiedLatent(
            semantic=semantic,
            structure=structure,
            symbols=symbols,
            scene="coding",
        )
        
        assert unified.has_structure == True
        assert unified.has_symbols == True
        assert unified.scene == "coding"
    
    def test_semantic_only_mode(self):
        """纯语义模式测试"""
        from app.interfaces.base_module import HierarchicalLatent
        
        semantic = HierarchicalLatent(
            global_=torch.randn(1, 1, 256),
            chunks=torch.randn(1, 4, 256),
        )
        
        unified = UnifiedLatent(
            semantic=semantic,
            structure=None,
            symbols=None,
            scene="chat",
        )
        
        assert unified.has_structure == False
        assert unified.has_symbols == False
        assert unified.semantic_only == True


class TestThreeChannelFlow:
    """三通道流程测试"""
    
    def test_structure_channel_components(self):
        """结构通道组件测试"""
        from app.io.structure_parser import parse_structure, StructureSummaryEncoder
        from app.memory.graph_memory import GraphMemory
        
        # 解析代码
        code = "x = get_value()"
        graph = parse_structure(code)
        
        assert len(graph.nodes) > 0
        
        # 编码摘要
        encoder = StructureSummaryEncoder(d_model=256)
        summary = encoder(graph)
        
        assert summary.shape == (256,)
        
        # 存储到 GraphMemory
        gm = GraphMemory(d_node=256)
        nodes_data = [{"id": "n1", "type": "x", "slot_id": 0}]
        created = gm.store_structure("test", nodes_data, [], summary)
        
        assert len(created) >= 1
    
    def test_symbol_channel_components(self):
        """符号通道组件测试"""
        from app.io.symbol_anchor import (
            detect_critical_tokens,
            SymbolAnchors,
            SymbolAnchor,
        )
        
        # 检测关键 token
        code = "user_name = get_user(id)"
        critical = detect_critical_tokens(code)
        
        assert len(critical) > 0
        
        # 创建锚点
        anchors = SymbolAnchors()
        for i, (pos, text, type_) in enumerate(critical):
            anchors.anchors.append(SymbolAnchor(
                position=pos,
                token_id=hash(text) % 10000,
                token_text=text,
                slot_id=i,
            ))
        
        assert anchors.num_anchors > 0
        
        # 验证有变量名
        texts = [a.token_text for a in anchors.anchors]
        assert "user_name" in texts or "get_user" in texts


class TestSceneRouting:
    """场景路由测试"""
    
    def test_auto_scene_detection(self):
        """自动场景检测测试"""
        from app.modulation.scene_encoder import SceneAwarePrecisionRouter
        
        router = SceneAwarePrecisionRouter()
        
        # 代码检测
        assert router.detect_scene("def hello(): pass") == "coding"
        assert router.detect_scene("class Foo:") == "coding"
        
        # 正式文档检测
        assert router.detect_scene("Dear Sir, I am writing to...") == "formal"
        
        # 默认聊天
        assert router.detect_scene("Hello! How are you?") == "chat"
    
    def test_precision_config_routing(self):
        """精度配置路由测试"""
        from app.modulation.scene_encoder import SceneAwarePrecisionRouter, SCENE_PRECISION_CONFIG
        
        router = SceneAwarePrecisionRouter()
        
        for scene, expected_config in SCENE_PRECISION_CONFIG.items():
            config = router.get_config(scene)
            assert config.semantic == expected_config.semantic
            assert config.structure == expected_config.structure
            assert config.symbols == expected_config.symbols


class TestCompatibilityMode:
    """兼容模式测试"""
    
    def test_three_channel_disabled(self):
        """三通道禁用测试"""
        from app.interfaces.unified_latent import to_hierarchical, to_legacy
        from app.interfaces.base_module import HierarchicalLatent
        
        # 创建纯语义 UnifiedLatent
        semantic = HierarchicalLatent(
            global_=torch.randn(2, 1, 256),
            chunks=torch.randn(2, 4, 256),
        )
        
        unified = UnifiedLatent(
            semantic=semantic,
            scene="chat",
        )
        
        # 转换为 hierarchical
        h = to_hierarchical(unified)
        assert isinstance(h, HierarchicalLatent)
        
        # 转换为 legacy
        legacy = to_legacy(unified)
        assert legacy.vector.shape == (2, 256)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
