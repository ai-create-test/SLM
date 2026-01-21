"""
Test Precision Router - 精度路由测试

Phase 7.3
"""

import pytest
import torch

from app.modulation.scene_encoder import (
    SceneEncoder,
    SceneAwarePrecisionRouter,
    AutoSceneDetector,
    PrecisionConfig,
    SCENE_IDS,
    SCENE_PRECISION_CONFIG,
)


class TestPrecisionConfig:
    """测试精度配置"""
    
    def test_default_config(self):
        """默认配置测试"""
        config = PrecisionConfig()
        
        assert config.semantic == True
        assert config.structure == False
        assert config.symbols == False
    
    def test_coding_config(self):
        """编程场景配置"""
        config = SCENE_PRECISION_CONFIG["coding"]
        
        assert config.semantic == True
        assert config.structure == True
        assert config.symbols == True
        assert config.all_channels() == True
    
    def test_chat_config(self):
        """聊天场景配置"""
        config = SCENE_PRECISION_CONFIG["chat"]
        
        assert config.semantic == True
        assert config.structure == False
        assert config.symbols == False
        assert config.semantic_only() == True


class TestSceneAwarePrecisionRouter:
    """测试场景感知精度路由器"""
    
    def test_basic_routing(self):
        """基本路由测试"""
        router = SceneAwarePrecisionRouter(use_auto_detection=False)
        
        config = router.get_config("coding")
        
        assert config.structure == True
        assert config.symbols == True
    
    def test_scene_detection(self):
        """场景检测测试"""
        router = SceneAwarePrecisionRouter()
        
        # 代码检测
        assert router.detect_scene("def hello(): pass") == "coding"
        assert router.detect_scene("class MyClass:") == "coding"
        assert router.detect_scene("import os") == "coding"
        
        # 正式文档检测
        assert router.detect_scene("Dear Sir,") == "formal"
        
        # 默认聊天
        assert router.detect_scene("Hello there!") == "chat"
    
    def test_route_method(self):
        """路由方法测试"""
        router = SceneAwarePrecisionRouter(use_auto_detection=False)
        
        routing = router.route("any text", scene="coding")
        
        assert routing == {
            "semantic": True,
            "structure": True,
            "symbols": True,
        }
    
    def test_forward_with_text(self):
        """文本前向传播测试"""
        router = SceneAwarePrecisionRouter(use_auto_detection=False)
        
        config = router.forward("def foo(): pass")
        
        assert isinstance(config, PrecisionConfig)
        assert config.structure == True  # 检测为 coding
    
    def test_forward_with_scene(self):
        """指定场景前向传播"""
        router = SceneAwarePrecisionRouter()
        
        config = router.forward("any text", scene="creative")
        
        assert config.structure == False
        assert config.symbols == False
    
    def test_all_scenes(self):
        """所有场景列表"""
        scenes = SceneAwarePrecisionRouter.get_all_scenes()
        
        assert "chat" in scenes
        assert "coding" in scenes
        assert "creative" in scenes
        assert len(scenes) == len(SCENE_IDS)


class TestAutoSceneDetector:
    """测试自动场景检测器"""
    
    def test_basic_classification(self):
        """基本分类测试"""
        detector = AutoSceneDetector(d_model=256)
        hidden = torch.randn(2, 16, 256)
        
        logits = detector(hidden)
        
        assert logits.shape == (2, 8)  # 8 scenes
    
    def test_predict(self):
        """预测场景名称"""
        detector = AutoSceneDetector(d_model=128)
        hidden = torch.randn(4, 8, 128)
        
        scenes = detector.predict(hidden)
        
        assert len(scenes) == 4
        assert all(s in SCENE_IDS for s in scenes)


class TestSceneEncoder:
    """测试场景编码器 (原有功能)"""
    
    def test_basic_encoding(self):
        """基本编码测试"""
        encoder = SceneEncoder(d_scene=128)
        
        scene_vec = encoder(torch.tensor([1]))  # coding
        
        assert scene_vec.shape == (1, 128)
    
    def test_encode_name(self):
        """按名称编码"""
        encoder = SceneEncoder(d_scene=64)
        
        scene_vec = encoder.encode_name("creative")
        
        assert scene_vec.shape == (1, 64)
    
    def test_all_scenes_different(self):
        """不同场景向量应不同"""
        encoder = SceneEncoder(d_scene=128)
        
        chat_vec = encoder.encode_name("chat")
        coding_vec = encoder.encode_name("coding")
        
        # 向量应该不同
        assert not torch.allclose(chat_vec, coding_vec)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
