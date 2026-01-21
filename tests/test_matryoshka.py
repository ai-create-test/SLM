"""
Test Matryoshka Projection - 嵌套投影测试

Phase 3.3
"""

import pytest
import torch

from app.io.matryoshka import (
    MatryoshkaProjection,
    MatryoshkaOutput,
    adaptive_precision,
)


class TestMatryoshkaProjection:
    """测试 Matryoshka 投影"""
    
    def test_basic_projection(self):
        """基本投影测试"""
        proj = MatryoshkaProjection(d_input=768, d_output=512)
        x = torch.randn(4, 768)
        
        z = proj(x)
        
        assert z.shape == (4, 512)
    
    def test_sequence_projection(self):
        """序列投影测试"""
        proj = MatryoshkaProjection(d_input=512, d_output=256)
        x = torch.randn(2, 8, 512)
        
        z = proj(x)
        
        assert z.shape == (2, 8, 256)
    
    def test_nesting_dims(self):
        """嵌套维度测试"""
        nesting_dims = [64, 128, 256, 512]
        proj = MatryoshkaProjection(
            d_input=768,
            d_output=512,
            nesting_dims=nesting_dims
        )
        
        assert proj.nesting_dims == nesting_dims
        assert proj.num_levels == 4
    
    def test_get_nested(self):
        """获取嵌套向量测试"""
        proj = MatryoshkaProjection(
            d_input=768,
            d_output=256,
            nesting_dims=[64, 128, 256]
        )
        x = torch.randn(4, 768)
        z = proj(x)
        
        z64 = proj.get_nested(z, level=0)
        z128 = proj.get_nested(z, level=1)
        z256 = proj.get_nested(z, level=2)
        
        assert z64.shape == (4, 64)
        assert z128.shape == (4, 128)
        assert z256.shape == (4, 256)
    
    def test_forward_with_nested(self):
        """带嵌套输出的前向传播"""
        proj = MatryoshkaProjection(
            d_input=512,
            d_output=256,
            nesting_dims=[64, 128, 256]
        )
        x = torch.randn(2, 512)
        
        output = proj.forward_with_nested(x)
        
        assert isinstance(output, MatryoshkaOutput)
        assert output.full.shape == (2, 256)
        assert 64 in output.nested
        assert 128 in output.nested
        assert 256 in output.nested


class TestMultiLevelLoss:
    """测试多级损失"""
    
    def test_mse_loss(self):
        """MSE 多级损失测试"""
        proj = MatryoshkaProjection(
            d_input=256,
            d_output=128,
            nesting_dims=[32, 64, 128]
        )
        x = torch.randn(4, 256)
        z = proj(x)
        target = torch.randn(4, 128)
        
        loss = proj.multi_level_loss(z, target, loss_type="mse")
        
        assert loss.item() >= 0
    
    def test_cosine_loss(self):
        """Cosine 多级损失测试"""
        proj = MatryoshkaProjection(
            d_input=256,
            d_output=128,
            nesting_dims=[32, 64, 128]
        )
        x = torch.randn(4, 256)
        z = proj(x)
        target = torch.randn(4, 128)
        
        loss = proj.multi_level_loss(z, target, loss_type="cosine")
        
        assert 0 <= loss.item() <= 2  # cosine 损失范围
    
    def test_contrastive_loss(self):
        """对比损失测试"""
        proj = MatryoshkaProjection(
            d_input=256,
            d_output=128,
            nesting_dims=[32, 64, 128]
        )
        
        anchor = torch.randn(4, 128)
        positive = torch.randn(4, 128)
        negatives = torch.randn(4, 8, 128)  # 8 negatives per sample
        
        loss = proj.multi_level_contrastive_loss(anchor, positive, negatives)
        
        assert loss.item() >= 0


class TestAdaptivePrecision:
    """测试自适应精度"""
    
    def test_basic_adaptive(self):
        """基本自适应测试"""
        z = torch.randn(4, 256)
        complexity = torch.tensor([0.1, 0.5, 0.8, 1.0])
        nesting_dims = [64, 128, 192, 256]
        
        output = adaptive_precision(z, complexity, nesting_dims)
        
        assert output.shape == (4, 256)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
