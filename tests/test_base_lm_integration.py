"""
BaseLM Integration Tests

测试预训练语言模型集成。
"""

import pytest
import torch


class TestFallbackLM:
    """FallbackLM 单元测试"""
    
    def test_fallback_lm_forward(self):
        """测试备用 LM 前向传播"""
        from app.io.base_lm import FallbackLM
        
        lm = FallbackLM(vocab_size=1000, d_model=256, num_layers=2)
        
        # 创建输入
        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32)
        
        # 前向传播
        output = lm(input_ids, attention_mask=attention_mask)
        
        assert hasattr(output, 'last_hidden_state')
        assert output.last_hidden_state.shape == (2, 32, 256)
        
        print(f"[OK] FallbackLM forward: shape={output.last_hidden_state.shape}")
    
    def test_fallback_lm_encode(self):
        """测试 encode 接口"""
        from app.io.base_lm import FallbackLM
        
        lm = FallbackLM(vocab_size=1000, d_model=128)
        
        input_ids = torch.randint(0, 1000, (3, 16))
        hidden = lm.encode(input_ids)
        
        assert hidden.shape == (3, 16, 128)
        
        print(f"[OK] FallbackLM encode: shape={hidden.shape}")
    
    def test_fallback_lm_freeze_unfreeze(self):
        """测试冻结和解冻"""
        from app.io.base_lm import FallbackLM
        
        lm = FallbackLM(d_model=128)
        
        # 初始状态是未冻结
        assert not lm.is_frozen
        
        # 冻结
        lm.freeze()
        assert lm.is_frozen
        
        # 验证参数不可训练
        for param in lm.parameters():
            assert not param.requires_grad
        
        # 解冻
        lm.unfreeze()
        assert not lm.is_frozen
        
        print("[OK] FallbackLM freeze/unfreeze works")


class TestGetBaseLM:
    """get_base_lm 测试"""
    
    def test_get_fallback(self):
        """测试获取 fallback LM"""
        from app.io.base_lm import get_base_lm, FallbackLM
        
        # 使用不存在的模型名，触发 fallback
        lm = get_base_lm("nonexistent-model", fallback=True, d_model=256)
        
        assert lm is not None
        
        print(f"[OK] get_base_lm fallback: type={type(lm).__name__}")


class TestHuggingFaceLM:
    """HuggingFace LM 测试 (如果可用)"""
    
    @pytest.fixture
    def check_transformers(self):
        try:
            import transformers
            return True
        except ImportError:
            pytest.skip("transformers not installed")
            return False
    
    def test_load_bert(self, check_transformers):
        """测试加载 BERT"""
        from app.io.base_lm import BaseLM
        
        lm = BaseLM.from_pretrained("bert-base-uncased", freeze=True)
        
        assert lm.d_model == 768
        assert lm.is_frozen
        
        print(f"[OK] BERT loaded: d_model={lm.d_model}")
    
    def test_bert_forward(self, check_transformers):
        """测试 BERT 前向传播"""
        from app.io.base_lm import BaseLM
        from app.io.tokenizer_wrapper import TokenizerWrapper
        
        # 加载模型和 tokenizer
        lm = BaseLM.from_pretrained("bert-base-uncased")
        tokenizer = TokenizerWrapper.from_pretrained("bert-base-uncased")
        
        # Tokenize
        output = tokenizer.encode("Hello world!")
        
        # 前向传播
        lm_output = lm(output.input_ids, attention_mask=output.attention_mask)
        
        assert lm_output.last_hidden_state.shape[0] == 1
        assert lm_output.last_hidden_state.shape[2] == 768
        
        print(f"[OK] BERT forward: shape={lm_output.last_hidden_state.shape}")


class TestParagraphEncoderWithLM:
    """ParagraphEncoder 与 BaseLM 集成测试"""
    
    def test_encoder_with_fallback_lm(self):
        """测试 Encoder 使用 fallback LM"""
        from app.io import ParagraphEncoder
        
        # 创建 encoder (使用 fallback)
        encoder = ParagraphEncoder(
            d_model=256,
            d_latent=128,
            use_vq=False,
            max_length=64,
        )
        
        # 检查基础模型信息
        info = encoder.get_base_model_info()
        print(f"[OK] Encoder base model info: {info}")
        
        # 前向传播
        output = encoder("Hello world")
        assert output.latent.vector.shape == (1, 128)
        
        print(f"[OK] Encoder forward with fallback LM: latent shape={output.latent.vector.shape}")
    
    def test_encoder_batch(self):
        """测试批量编码"""
        from app.io import ParagraphEncoder
        
        encoder = ParagraphEncoder(
            d_model=128,
            d_latent=64,
            use_vq=False,
        )
        
        texts = ["First text.", "Second text.", "Third text."]
        output = encoder(texts)
        
        assert output.latent.vector.shape == (3, 64)
        
        print(f"[OK] Encoder batch: latent shape={output.latent.vector.shape}")


def run_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("Running BaseLM Integration Tests")
    print("="*60 + "\n")
    
    # FallbackLM tests
    print("--- FallbackLM Tests ---")
    test_fallback = TestFallbackLM()
    test_fallback.test_fallback_lm_forward()
    test_fallback.test_fallback_lm_encode()
    test_fallback.test_fallback_lm_freeze_unfreeze()
    
    # get_base_lm tests
    print("\n--- get_base_lm Tests ---")
    test_get = TestGetBaseLM()
    test_get.test_get_fallback()
    
    # ParagraphEncoder integration
    print("\n--- ParagraphEncoder Integration Tests ---")
    test_encoder = TestParagraphEncoderWithLM()
    test_encoder.test_encoder_with_fallback_lm()
    test_encoder.test_encoder_batch()
    
    # HuggingFace tests
    print("\n--- HuggingFace LM Tests ---")
    try:
        import transformers
        test_hf = TestHuggingFaceLM()
        test_hf.test_load_bert(True)
        test_hf.test_bert_forward(True)
    except ImportError:
        print("! transformers not installed, skipping HuggingFace tests")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_tests()
