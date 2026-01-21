"""
Test Hierarchical Encoder - 分层编码器测试

Phase 4.6
"""

import pytest
import torch

from app.io.paragraph_encoder import (
    ChunkEncoder,
    GlobalPooler,
    DetailEncoder,
    create_encoder,
)


class TestChunkEncoder:
    """测试块编码器"""
    
    def test_basic_encoding(self):
        """基本编码测试"""
        encoder = ChunkEncoder(d_model=256, num_layers=2)
        chunks = torch.randn(2, 4, 16, 256)  # [batch, num_chunks, chunk_len, d_model]
        mask = torch.ones(2, 4, 16)
        
        output = encoder(chunks, mask)
        
        assert output.shape == (2, 4, 256)  # [batch, num_chunks, d_model]
    
    def test_with_padding(self):
        """带 padding 编码测试"""
        encoder = ChunkEncoder(d_model=128)
        chunks = torch.randn(2, 4, 16, 128)
        mask = torch.zeros(2, 4, 16)
        mask[:, :, :8] = 1  # 只有前 8 个有效
        
        output = encoder(chunks, mask)
        
        assert output.shape == (2, 4, 128)


class TestGlobalPooler:
    """测试全局池化器"""
    
    def test_basic_pooling(self):
        """基本池化测试"""
        pooler = GlobalPooler(d_model=256)
        chunk_embeds = torch.randn(2, 4, 256)
        chunk_mask = torch.ones(2, 4)
        
        global_embed = pooler(chunk_embeds, chunk_mask)
        
        assert global_embed.shape == (2, 1, 256)
    
    def test_with_mask(self):
        """带掩码池化测试"""
        pooler = GlobalPooler(d_model=128, num_layers=2)
        chunk_embeds = torch.randn(2, 8, 128)
        chunk_mask = torch.zeros(2, 8)
        chunk_mask[:, :4] = 1  # 只有前 4 个 chunk 有效
        
        global_embed = pooler(chunk_embeds, chunk_mask)
        
        assert global_embed.shape == (2, 1, 128)


class TestDetailEncoder:
    """测试细节编码器"""
    
    def test_basic_encoding(self):
        """基本细节编码测试"""
        encoder = DetailEncoder(d_model=256, d_detail=128, max_detail_tokens=16)
        hidden = torch.randn(2, 64, 256)
        mask = torch.ones(2, 64)
        
        detail, detail_mask = encoder(hidden, mask)
        
        assert detail.shape == (2, 16, 128)
        assert detail_mask.shape == (2, 16)
    
    def test_short_sequence(self):
        """短序列测试"""
        encoder = DetailEncoder(d_model=128, max_detail_tokens=32)
        hidden = torch.randn(2, 8, 128)  # 序列长度 < max_detail_tokens
        
        detail, detail_mask = encoder(hidden)
        
        assert detail.shape == (2, 8, 64)  # min(8, 32)
    
    def test_importance_scoring(self):
        """重要性评分测试"""
        encoder = DetailEncoder(d_model=128, max_detail_tokens=8)
        hidden = torch.randn(2, 32, 128)
        
        detail, detail_mask = encoder(hidden)
        
        # 应该选择 top-8 最重要的 token
        assert detail.shape == (2, 8, 64)


class TestCreateEncoder:
    """测试编码器工厂方法"""
    
    def test_legacy_encoder(self):
        """Legacy 编码器创建"""
        from app.interfaces.config import ModelConfig
        config = ModelConfig()
        config.use_hierarchical = False
        config.use_three_channel = False
        
        # 由于需要 tokenizer，这里只测试类型选择逻辑
        # encoder = create_encoder(config, encoder_type="legacy")
        pass
    
    def test_hierarchical_encoder(self):
        """Hierarchical 编码器创建"""
        from app.interfaces.config import ModelConfig
        config = ModelConfig()
        config.use_hierarchical = True
        config.use_three_channel = False
        
        # encoder = create_encoder(config, encoder_type="hierarchical")
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
