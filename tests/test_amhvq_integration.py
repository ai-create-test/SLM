"""
Test AMHVQ+ Integration - 集成测试

Phase 17: 测试验证
"""

import pytest
import torch

from app.interfaces.base_module import HierarchicalLatent
from app.interfaces.unified_latent import UnifiedLatent, PrecisionConfig


class TestPhase13ConfigPresets:
    """测试配置预设"""
    
    def test_config_files_exist(self):
        """配置文件存在"""
        from pathlib import Path
        
        configs_dir = Path("configs")
        assert (configs_dir / "amhvq_base.yaml").exists()
        assert (configs_dir / "amhvq_coding.yaml").exists()
        assert (configs_dir / "amhvq_small.yaml").exists()


class TestPhase14Modulation:
    """测试层次调制"""
    
    def test_emotion_modulator(self):
        """情感调制测试"""
        from app.modulation.hierarchical_modulation import HierarchicalEmotionModulator
        
        modulator = HierarchicalEmotionModulator(d_latent=256, d_emotion=32)
        
        h = HierarchicalLatent(
            global_=torch.randn(2, 1, 256),
            chunks=torch.randn(2, 4, 256),
        )
        
        modulated = modulator(h, emotion=0)
        
        assert modulated.global_.shape == h.global_.shape
        assert modulated.chunks.shape == h.chunks.shape
    
    def test_scene_modulator(self):
        """场景调制测试"""
        from app.modulation.hierarchical_modulation import HierarchicalSceneModulator
        
        modulator = HierarchicalSceneModulator(d_latent=256, d_scene=32)
        
        h = HierarchicalLatent(
            global_=torch.randn(2, 1, 256),
            chunks=torch.randn(2, 4, 256),
        )
        
        modulated = modulator(h, scene="coding")
        
        assert modulated.global_.shape == h.global_.shape


class TestPhase15Memory:
    """测试层次记忆"""
    
    def test_hierarchical_memory_store(self):
        """层次记忆存储测试"""
        from app.memory.hierarchical_memory import HierarchicalMemoryStore
        
        store = HierarchicalMemoryStore(d_latent=256)
        
        h1 = HierarchicalLatent(
            global_=torch.randn(1, 1, 256),
            chunks=torch.randn(1, 4, 256),
        )
        
        # 存储
        entry_id = store.store(h1, metadata={"test": True})
        assert store.num_entries == 1
        
        # 检索
        h2 = HierarchicalLatent(
            global_=h1.global_ + 0.1 * torch.randn_like(h1.global_),
            chunks=torch.randn(1, 4, 256),
        )
        
        result = store.retrieve(h2, top_k=1)
        assert len(result.entries) == 1
        assert result.similarities[0] > 0.5


class TestPhase16Inference:
    """测试推理接口"""
    
    def test_inference_config(self):
        """推理配置测试"""
        from app.inference.amhvq_inference import InferenceConfig
        
        config = InferenceConfig(
            scene="coding",
            precision="high",
            max_length=512,
        )
        
        assert config.scene == "coding"
        assert config.precision == "high"
    
    def test_precision_config(self):
        """精度配置测试"""
        config = PrecisionConfig(
            semantic=512,
            structure=True,
            symbols=True,
        )
        
        assert config.semantic == 512
        assert config.structure == True


class TestPhase17Integration:
    """集成测试"""
    
    def test_full_pipeline_imports(self):
        """完整管道导入测试"""
        from app.io.unified_encoder import UnifiedEncoder
        from app.io.unified_decoder import UnifiedDecoder
        from app.brain.hierarchical_dynamics import HierarchicalDynamics
        from app.training.unified_training_stages import UnifiedTrainingStage
        from app.model.unified_model import UnifiedNeuralFlowModel
        
        assert UnifiedEncoder is not None
        assert UnifiedDecoder is not None
        assert HierarchicalDynamics is not None
        assert UnifiedTrainingStage is not None
        assert UnifiedNeuralFlowModel is not None
    
    def test_hierarchical_latent_flow(self):
        """层次潜向量流程测试"""
        # 创建 HierarchicalLatent
        h = HierarchicalLatent(
            global_=torch.randn(2, 1, 256),
            chunks=torch.randn(2, 4, 256),
        )
        
        # 测试方法
        single = h.to_single_vector()
        assert single.shape == (2, 256)
        
        flat = h.flatten()
        assert flat.shape[1] == 5  # 1 global + 4 chunks
    
    def test_unified_latent_flow(self):
        """统一潜向量流程测试"""
        h = HierarchicalLatent(
            global_=torch.randn(2, 1, 256),
            chunks=torch.randn(2, 4, 256),
        )
        
        u = UnifiedLatent(
            semantic=h,
            scene="coding",
        )
        
        assert u.semantic is not None
        assert u.scene == "coding"
        assert u.has_structure == False
        assert u.has_symbols == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
