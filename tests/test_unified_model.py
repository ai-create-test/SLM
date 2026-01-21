"""
Test Unified Model - 统一模型测试

Phase 12 测试
"""

import pytest
import torch
from pathlib import Path
import tempfile

from app.interfaces.base_module import HierarchicalLatent
from app.interfaces.unified_latent import UnifiedLatent
from app.interfaces.config import Config


class TestUnifiedModelImports:
    """测试导入"""
    
    def test_imports(self):
        """导入测试"""
        from app.model.unified_model import (
            UnifiedNeuralFlowModel,
            UnifiedModelMetadata,
            create_unified_model,
            load_model,
        )
        assert UnifiedNeuralFlowModel is not None
        assert create_unified_model is not None


class TestUnifiedModelMetadata:
    """测试模型元数据"""
    
    def test_metadata_creation(self):
        """元数据创建测试"""
        from app.model.unified_model import UnifiedModelMetadata
        
        metadata = UnifiedModelMetadata(
            model_type="UnifiedNeuralFlow",
            version="3.0.0",
        )
        
        assert metadata.model_type == "UnifiedNeuralFlow"
        assert metadata.version == "3.0.0"
    
    def test_metadata_serialization(self):
        """元数据序列化测试"""
        from app.model.unified_model import UnifiedModelMetadata
        
        metadata = UnifiedModelMetadata(
            model_type="Test",
            architecture={"d_model": 768},
        )
        
        d = metadata.to_dict()
        assert d["model_type"] == "Test"
        assert d["architecture"]["d_model"] == 768
        
        restored = UnifiedModelMetadata.from_dict(d)
        assert restored.model_type == "Test"


class TestFactoryMethods:
    """测试工厂方法"""
    
    def test_create_unified_model(self):
        """创建统一模型"""
        from app.model.unified_model import create_unified_model
        
        # 这个测试可能需要加载基础模型，暂时跳过完整初始化
        # model = create_unified_model(model_type="unified", use_three_channel=True)
        pass
    
    def test_from_preset(self):
        """从预设创建"""
        from app.model.unified_model import UnifiedNeuralFlowModel
        
        # 预设测试
        # model = UnifiedNeuralFlowModel.from_preset("base")
        pass


class TestCheckpointCompatibility:
    """测试 Checkpoint 兼容性"""
    
    def test_save_load_config(self):
        """保存/加载配置"""
        from app.model.model_utils import save_config, load_config
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            
            test_config = {
                "model_type": "UnifiedNeuralFlow",
                "d_model": 768,
            }
            
            save_config(test_config, config_path)
            loaded = load_config(config_path)
            
            assert loaded["model_type"] == "UnifiedNeuralFlow"
            assert loaded["d_model"] == 768


class TestModelMethods:
    """测试模型方法 (不需要完整初始化)"""
    
    def test_hierarchical_latent_compatible(self):
        """HierarchicalLatent 兼容性"""
        h = HierarchicalLatent(
            global_=torch.randn(2, 1, 256),
            chunks=torch.randn(2, 4, 256),
        )
        
        # 可以转换为单向量
        single = h.to_single_vector()
        assert single.shape == (2, 256)
    
    def test_unified_latent_compatible(self):
        """UnifiedLatent 兼容性"""
        h = HierarchicalLatent(
            global_=torch.randn(2, 1, 256),
            chunks=torch.randn(2, 4, 256),
        )
        
        unified = UnifiedLatent(
            semantic=h,
            scene="chat",
        )
        
        assert unified.semantic is not None
        assert unified.scene == "chat"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
