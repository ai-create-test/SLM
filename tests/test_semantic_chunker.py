"""
Test Semantic Chunker - 语义分块器测试

Phase 2.3
"""

import pytest
import torch

from app.io.semantic_chunker import (
    SemanticChunker,
    BoundaryPredictor,
    ChunkerOutput,
    pool_chunks,
)


class TestBoundaryPredictor:
    """测试边界预测器"""
    
    def test_basic_prediction(self):
        """基本预测测试"""
        predictor = BoundaryPredictor(d_model=256)
        hidden = torch.randn(2, 32, 256)
        
        logits = predictor(hidden)
        
        assert logits.shape == (2, 32)


class TestSemanticChunker:
    """测试语义分块器"""
    
    def test_basic_chunking(self):
        """基本分块测试"""
        chunker = SemanticChunker(d_model=256, max_chunks=4, max_chunk_len=16)
        hidden = torch.randn(2, 32, 256)
        mask = torch.ones(2, 32)
        
        output = chunker(hidden, mask, hard=True)
        
        assert isinstance(output, ChunkerOutput)
        assert output.chunks.shape[0] == 2  # batch
        assert output.chunks.shape[1] == 4  # max_chunks
        assert output.chunks.shape[2] == 16  # max_chunk_len
        assert output.chunks.shape[3] == 256  # d_model
    
    def test_chunk_mask(self):
        """分块掩码测试"""
        chunker = SemanticChunker(d_model=128, max_chunks=8)
        hidden = torch.randn(2, 64, 128)
        
        output = chunker(hidden, hard=True)
        
        assert output.chunk_mask.shape == (2, 8)
        assert output.token_mask.shape[:3] == output.chunks.shape[:3]
    
    def test_num_chunks_adaptive(self):
        """自适应块数测试"""
        chunker = SemanticChunker(d_model=128, max_chunks=8, use_learned_boundaries=True)
        
        # 短序列
        short = torch.randn(1, 16, 128)
        out_short = chunker(short, hard=True)
        
        # 长序列
        long = torch.randn(1, 128, 128)
        out_long = chunker(long, hard=True)
        
        # 长序列通常产生更多块
        assert out_short.num_chunks.shape == (1,)
        assert out_long.num_chunks.shape == (1,)
    
    def test_fixed_boundaries(self):
        """固定边界分块测试"""
        chunker = SemanticChunker(
            d_model=128,
            max_chunks=4,
            max_chunk_len=16,
            use_learned_boundaries=False
        )
        hidden = torch.randn(2, 64, 128)
        
        output = chunker(hidden, hard=True)
        
        assert output.chunks.shape == (2, 4, 16, 128)


class TestPoolChunks:
    """测试块池化"""
    
    def test_mean_pooling(self):
        """均值池化测试"""
        chunks = torch.randn(2, 4, 16, 128)
        token_mask = torch.ones(2, 4, 16)
        
        pooled = pool_chunks(chunks, token_mask, pooling_type="mean")
        
        assert pooled.shape == (2, 4, 128)
    
    def test_first_pooling(self):
        """首 token 池化测试"""
        chunks = torch.randn(2, 4, 16, 128)
        token_mask = torch.ones(2, 4, 16)
        
        pooled = pool_chunks(chunks, token_mask, pooling_type="first")
        
        assert pooled.shape == (2, 4, 128)
        # 验证取的是第一个 token
        assert torch.allclose(pooled, chunks[:, :, 0, :])
    
    def test_masked_pooling(self):
        """带掩码池化测试"""
        chunks = torch.randn(2, 4, 16, 128)
        token_mask = torch.zeros(2, 4, 16)
        token_mask[:, :, :8] = 1  # 只有前 8 个有效
        
        pooled = pool_chunks(chunks, token_mask, pooling_type="mean")
        
        assert pooled.shape == (2, 4, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
