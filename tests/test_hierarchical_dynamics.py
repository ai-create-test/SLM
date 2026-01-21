"""
Test Hierarchical Dynamics - 层次化动力学测试

Phase 10.4
"""

import pytest
import torch

from app.interfaces.base_module import HierarchicalLatent
from app.interfaces.unified_latent import UnifiedLatent


class TestSetEncoder:
    """测试集合编码器"""
    
    def test_basic_encoding(self):
        """基本编码测试"""
        from app.brain.hierarchical_dynamics import SetEncoder
        
        encoder = SetEncoder(d_latent=256, d_output=512)
        
        hierarchical = HierarchicalLatent(
            global_=torch.randn(2, 1, 256),
            chunks=torch.randn(2, 4, 256),
        )
        
        output = encoder(hierarchical)
        
        assert output.shape == (2, 512)
    
    def test_sequence_encoding(self):
        """序列编码测试"""
        from app.brain.hierarchical_dynamics import SetEncoder
        
        encoder = SetEncoder(d_latent=128, d_output=256)
        
        sequence = [
            HierarchicalLatent(global_=torch.randn(2, 1, 128), chunks=torch.randn(2, 4, 128)),
            HierarchicalLatent(global_=torch.randn(2, 1, 128), chunks=torch.randn(2, 4, 128)),
            HierarchicalLatent(global_=torch.randn(2, 1, 128), chunks=torch.randn(2, 4, 128)),
        ]
        
        output = encoder.encode_sequence(sequence)
        
        assert output.shape == (2, 3, 256)


class TestHierarchicalDynamics:
    """测试层次化动力学"""
    
    def test_basic_prediction(self):
        """基本预测测试"""
        from app.brain.hierarchical_dynamics import HierarchicalDynamics
        
        dynamics = HierarchicalDynamics(
            d_latent=256,
            d_model=512,
            num_chunks=4,
        )
        
        sequence = [
            HierarchicalLatent(global_=torch.randn(2, 1, 256), chunks=torch.randn(2, 4, 256)),
            HierarchicalLatent(global_=torch.randn(2, 1, 256), chunks=torch.randn(2, 4, 256)),
        ]
        
        output = dynamics(sequence)
        
        assert output.predicted_global.shape == (2, 256)
        assert output.predicted_chunks.shape == (2, 4, 256)
        assert output.predicted_hierarchical is not None
    
    def test_predict_next(self):
        """便捷预测方法测试"""
        from app.brain.hierarchical_dynamics import HierarchicalDynamics
        
        dynamics = HierarchicalDynamics(d_latent=128, d_model=256, num_chunks=4)
        
        history = [
            HierarchicalLatent(global_=torch.randn(1, 1, 128), chunks=torch.randn(1, 4, 128)),
        ]
        
        predicted = dynamics.predict_next(history)
        
        assert isinstance(predicted, HierarchicalLatent)


class TestUnifiedDynamics:
    """测试统一动力学"""
    
    def test_unified_prediction(self):
        """UnifiedLatent 预测测试"""
        from app.brain.hierarchical_dynamics import UnifiedDynamics
        
        dynamics = UnifiedDynamics(
            d_latent=256,
            d_model=512,
            num_chunks=4,
        )
        
        sequence = [
            UnifiedLatent(
                semantic=HierarchicalLatent(
                    global_=torch.randn(2, 1, 256),
                    chunks=torch.randn(2, 4, 256),
                ),
                scene="chat",
            ),
        ]
        
        predicted = dynamics(sequence)
        
        assert isinstance(predicted, UnifiedLatent)
        assert predicted.semantic is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
