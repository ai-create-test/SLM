"""
Tokenizer Integration Tests

测试 TokenizerWrapper 和 ParagraphEncoder 的 Tokenizer 集成。
"""

import pytest
import torch


class TestTokenizerWrapper:
    """TokenizerWrapper 单元测试"""
    
    def test_fallback_tokenizer_encode(self):
        """测试备用 tokenizer 编码"""
        from app.io.tokenizer_wrapper import FallbackTokenizer
        
        tokenizer = FallbackTokenizer(max_length=64)
        
        # 单文本编码
        output = tokenizer.encode("Hello world")
        
        assert hasattr(output, 'input_ids')
        assert hasattr(output, 'attention_mask')
        assert output.input_ids.shape[0] == 1  # batch size
        assert output.attention_mask.shape == output.input_ids.shape
        
        print(f"✓ Fallback tokenizer encode: shape={output.input_ids.shape}")
    
    def test_fallback_tokenizer_batch(self):
        """测试备用 tokenizer 批量编码"""
        from app.io.tokenizer_wrapper import FallbackTokenizer
        
        tokenizer = FallbackTokenizer(max_length=64)
        
        # 批量编码
        texts = ["Hello", "World", "How are you?"]
        output = tokenizer.encode_batch(texts)
        
        assert output.input_ids.shape[0] == 3  # batch size
        assert output.attention_mask.shape[0] == 3
        
        print(f"✓ Fallback tokenizer batch: shape={output.input_ids.shape}")
    
    def test_fallback_tokenizer_decode(self):
        """测试备用 tokenizer 解码"""
        from app.io.tokenizer_wrapper import FallbackTokenizer
        
        tokenizer = FallbackTokenizer(max_length=64, use_tiktoken=False)
        
        # 编码再解码
        text = "Hello"
        output = tokenizer.encode(text)
        decoded = tokenizer.decode(output.input_ids[0])
        
        # 基本的字符应该保留
        assert len(decoded) > 0
        print(f"✓ Fallback tokenizer decode: '{text}' -> encoded -> '{decoded}'")
    
    def test_get_tokenizer_fallback(self):
        """测试 get_tokenizer 的 fallback 行为"""
        from app.io.tokenizer_wrapper import get_tokenizer, FallbackTokenizer
        
        # 使用一个不存在的模型名，应该 fallback
        tokenizer = get_tokenizer("nonexistent-model-xyz", fallback=True)
        
        assert tokenizer is not None
        assert isinstance(tokenizer, FallbackTokenizer)
        
        print(f"✓ get_tokenizer fallback works")
    
    def test_tokenizer_output_to_device(self):
        """测试 TokenizerOutput.to() 方法"""
        from app.io.tokenizer_wrapper import FallbackTokenizer
        
        tokenizer = FallbackTokenizer(max_length=32)
        output = tokenizer.encode("Test")
        
        # 移动到 CPU (应该不报错)
        output_cpu = output.to(torch.device("cpu"))
        assert output_cpu.input_ids.device.type == "cpu"
        
        print(f"✓ TokenizerOutput.to() works")


class TestHuggingFaceTokenizer:
    """HuggingFace Tokenizer 测试 (如果可用)"""
    
    @pytest.fixture
    def check_transformers(self):
        """检查 transformers 是否可用"""
        try:
            import transformers
            return True
        except ImportError:
            pytest.skip("transformers not installed")
            return False
    
    def test_load_bert_tokenizer(self, check_transformers):
        """测试加载 BERT tokenizer"""
        from app.io.tokenizer_wrapper import TokenizerWrapper
        
        tokenizer = TokenizerWrapper.from_pretrained("bert-base-uncased", max_length=128)
        
        assert tokenizer.vocab_size > 0
        assert tokenizer.pad_token_id is not None
        
        output = tokenizer.encode("Hello world!")
        assert output.input_ids.shape[0] == 1
        
        print(f"✓ BERT tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    
    def test_bert_encode_decode(self, check_transformers):
        """测试 BERT 编码解码"""
        from app.io.tokenizer_wrapper import TokenizerWrapper
        
        tokenizer = TokenizerWrapper.from_pretrained("bert-base-uncased")
        
        text = "The quick brown fox jumps over the lazy dog."
        output = tokenizer.encode(text)
        decoded = tokenizer.decode(output.input_ids[0])
        
        # 解码后应该接近原文
        assert "quick" in decoded.lower()
        assert "fox" in decoded.lower()
        
        print(f"✓ BERT encode/decode: '{text[:30]}...' -> '{decoded[:30]}...'")
    
    def test_bert_batch_encoding(self, check_transformers):
        """测试 BERT 批量编码"""
        from app.io.tokenizer_wrapper import TokenizerWrapper
        
        tokenizer = TokenizerWrapper.from_pretrained("bert-base-uncased")
        
        texts = [
            "First sentence.",
            "Second sentence is longer.",
            "Third.",
        ]
        output = tokenizer.encode_batch(texts)
        
        assert output.input_ids.shape[0] == 3
        assert output.attention_mask.shape[0] == 3
        
        # 所有序列应该被填充到相同长度
        assert output.input_ids.shape[1] == tokenizer.max_length
        
        print(f"✓ BERT batch encoding: {len(texts)} texts -> shape {output.input_ids.shape}")


class TestParagraphEncoderTokenizer:
    """ParagraphEncoder 的 Tokenizer 集成测试"""
    
    def test_encoder_with_fallback_tokenizer(self):
        """测试 Encoder 使用 fallback tokenizer"""
        from app.io import ParagraphEncoder
        from app.io.tokenizer_wrapper import FallbackTokenizer
        
        # 预先创建 fallback tokenizer
        fallback = FallbackTokenizer(max_length=64)
        
        encoder = ParagraphEncoder(
            d_model=256,  # 使用较小的维度加快测试
            d_latent=128,
            use_vq=False,  # 跳过 VQ 简化测试
            max_length=64,
            tokenizer=fallback,
        )
        
        # 测试分词方法
        output = encoder.tokenize("Hello world")
        assert output.input_ids.shape[0] == 1
        
        print(f"✓ ParagraphEncoder.tokenize() works with fallback")
    
    def test_encoder_vocab_size(self):
        """测试 Encoder 获取词表大小"""
        from app.io import ParagraphEncoder
        from app.io.tokenizer_wrapper import FallbackTokenizer
        
        fallback = FallbackTokenizer(vocab_size=30000, max_length=64)
        
        encoder = ParagraphEncoder(
            d_model=256,
            d_latent=128,
            use_vq=False,
            tokenizer=fallback,
        )
        
        assert encoder.vocab_size == 30000
        
        print(f"✓ ParagraphEncoder.vocab_size = {encoder.vocab_size}")
    
    def test_encoder_forward_with_text(self):
        """测试 Encoder 前向传播（使用文本输入）"""
        from app.io import ParagraphEncoder
        from app.io.tokenizer_wrapper import FallbackTokenizer
        
        fallback = FallbackTokenizer(max_length=32)
        
        encoder = ParagraphEncoder(
            d_model=128,
            d_latent=64,
            use_vq=False,
            tokenizer=fallback,
        )
        
        # 单文本
        output = encoder("Hello world!")
        assert output.latent.vector.shape == (1, 64)  # [batch, d_latent]
        
        # 批量文本
        output = encoder(["Hello", "World", "Test"])
        assert output.latent.vector.shape == (3, 64)
        
        print(f"✓ ParagraphEncoder forward with text: output shape = {output.latent.vector.shape}")


def run_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("Running Tokenizer Integration Tests")
    print("="*60 + "\n")
    
    # Fallback tokenizer tests
    print("--- FallbackTokenizer Tests ---")
    test_fallback = TestTokenizerWrapper()
    test_fallback.test_fallback_tokenizer_encode()
    test_fallback.test_fallback_tokenizer_batch()
    test_fallback.test_fallback_tokenizer_decode()
    test_fallback.test_get_tokenizer_fallback()
    test_fallback.test_tokenizer_output_to_device()
    
    # ParagraphEncoder integration tests
    print("\n--- ParagraphEncoder Integration Tests ---")
    test_encoder = TestParagraphEncoderTokenizer()
    test_encoder.test_encoder_with_fallback_tokenizer()
    test_encoder.test_encoder_vocab_size()
    test_encoder.test_encoder_forward_with_text()
    
    # HuggingFace tests (if available)
    print("\n--- HuggingFace Tokenizer Tests ---")
    try:
        import transformers
        test_hf = TestHuggingFaceTokenizer()
        test_hf.test_load_bert_tokenizer(True)
        test_hf.test_bert_encode_decode(True)
        test_hf.test_bert_batch_encoding(True)
    except ImportError:
        print("⚠ transformers not installed, skipping HuggingFace tests")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_tests()
