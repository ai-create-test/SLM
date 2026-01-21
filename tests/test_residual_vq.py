"""
Test Residual VQ - 残差向量量化测试

Phase 1.3
"""

import pytest
import torch

from app.io.residual_vq import (
    ResidualVQ,
    ResidualVQLayer,
    RVQOutput,
    progressive_decode,
)


class TestResidualVQLayer:
    """测试单层 VQ"""
    
    def test_basic_quantization(self):
        """基本量化测试"""
        layer = ResidualVQLayer(d_latent=256, codebook_size=1024)
        z = torch.randn(4, 256)
        
        quantized, indices, loss, perplexity = layer(z)
        
        assert quantized.shape == z.shape
        assert indices.shape == (4,)
        assert loss.item() >= 0
        assert perplexity.item() > 0
    
    def test_sequence_quantization(self):
        """序列量化测试"""
        layer = ResidualVQLayer(d_latent=256, codebook_size=512)
        z = torch.randn(2, 8, 256)  # [batch, seq_len, d_latent]
        
        quantized, indices, loss, perplexity = layer(z)
        
        assert quantized.shape == z.shape
        assert indices.shape == (2, 8)
    
    def test_decode(self):
        """解码测试"""
        layer = ResidualVQLayer(d_latent=256, codebook_size=512)
        indices = torch.randint(0, 512, (4, 8))
        
        decoded = layer.decode(indices)
        
        assert decoded.shape == (4, 8, 256)


class TestResidualVQ:
    """测试多层 RVQ"""
    
    def test_multi_layer_quantization(self):
        """多层量化测试"""
        rvq = ResidualVQ(d_latent=256, codebook_size=512, num_layers=3)
        z = torch.randn(2, 4, 256)
        
        output = rvq(z)
        
        assert isinstance(output, RVQOutput)
        assert output.quantized.shape == z.shape
        assert output.indices.shape == (2, 4, 3)  # 3 layers
        assert len(output.per_layer_quantized) == 3
    
    def test_progressive_decode(self):
        """渐进解码测试"""
        rvq = ResidualVQ(d_latent=128, codebook_size=256, num_layers=4)
        z = torch.randn(2, 4, 128)
        
        output = rvq(z)
        
        # 使用不同层数解码
        z1 = progressive_decode(output, num_layers=1)
        z2 = progressive_decode(output, num_layers=2)
        z4 = progressive_decode(output, num_layers=4)
        
        assert z1.shape == z.shape
        assert z2.shape == z.shape
        assert z4.shape == z.shape
        
        # 更多层应该更接近原始
        # (这里只验证形状，实际精度需要训练后验证)
    
    def test_encode_decode(self):
        """编码-解码一致性"""
        rvq = ResidualVQ(d_latent=128, codebook_size=256, num_layers=3)
        z = torch.randn(2, 4, 128)
        
        output = rvq(z)
        decoded = rvq.decode(output.indices)
        
        # 验证形状一致性
        assert decoded.shape == z.shape
        assert decoded.shape == output.quantized.shape
        
        # 验证 decode 是纯码本查找 (返回可微的张量)
        assert decoded.requires_grad  # 通过 embedding 层应保持梯度
    
    def test_codebook_usage(self):
        """码本利用率测试"""
        rvq = ResidualVQ(d_latent=128, codebook_size=256, num_layers=2)
        
        # 前向传播几次
        for _ in range(10):
            z = torch.randn(8, 4, 128)
            rvq(z)
        
        usages = rvq.get_codebook_usage()
        assert len(usages) == 2
        assert all(0 <= u <= 1 for u in usages)


class TestRVQOutput:
    """测试 RVQ 输出结构"""
    
    def test_output_fields(self):
        """输出字段完整性"""
        rvq = ResidualVQ(d_latent=64, codebook_size=128, num_layers=2)
        z = torch.randn(2, 4, 64)
        
        output = rvq(z)
        
        assert hasattr(output, 'quantized')
        assert hasattr(output, 'indices')
        assert hasattr(output, 'commitment_loss')
        assert hasattr(output, 'perplexity')
        assert hasattr(output, 'per_layer_quantized')
        assert hasattr(output, 'per_layer_losses')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
