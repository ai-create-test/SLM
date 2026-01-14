"""
Tokenizer模块完整单元测试

覆盖范围：
1. TokenizerResult 数据结构
2. BaseTokenizer 抽象接口
3. BPETokenizer 编解码功能
4. TokenAttentionMixin 注意力权重计算
5. TokenizerFactory 工厂模式
6. 边界情况和错误处理
"""

import pytest
import json
import tempfile
import os
from pathlib import Path

# 导入待测试模块
from app.core import (
    BaseTokenizer,
    TokenizerResult,
    BatchTokenizerResult,
    BPETokenizer,
    TokenizerFactory,
    get_tokenizer,
    TokenAttentionMixin,
    AttentionConfig,
    AttentionStrategy,
    QueryFocusedAttention,
    PaddingStrategy,
    TruncationStrategy,
)


# ==================== TokenizerResult 测试 ====================

class TestTokenizerResult:
    """测试TokenizerResult数据结构"""
    
    def test_basic_creation(self):
        """测试基本创建"""
        result = TokenizerResult(
            token_ids=[1, 2, 3],
            tokens=["hello", "world", "!"]
        )
        assert result.token_ids == [1, 2, 3]
        assert result.tokens == ["hello", "world", "!"]
        assert result.attention_weights is None
        assert result.attention_mask is None
    
    def test_len(self):
        """测试长度计算"""
        result = TokenizerResult(
            token_ids=[1, 2, 3, 4, 5],
            tokens=["a", "b", "c", "d", "e"]
        )
        assert len(result) == 5
    
    def test_to_dict(self):
        """测试字典转换"""
        result = TokenizerResult(
            token_ids=[1, 2],
            tokens=["a", "b"],
            attention_weights=[0.5, 0.8],
            attention_mask=[1, 1],
            metadata={"test": "value"}
        )
        d = result.to_dict()
        assert d["token_ids"] == [1, 2]
        assert d["tokens"] == ["a", "b"]
        assert d["attention_weights"] == [0.5, 0.8]
        assert d["attention_mask"] == [1, 1]
        assert d["metadata"] == {"test": "value"}
    
    def test_to_dict_optional_fields(self):
        """测试可选字段的字典转换"""
        result = TokenizerResult(token_ids=[1], tokens=["a"])
        d = result.to_dict()
        assert "attention_weights" not in d
        assert "attention_mask" not in d
        assert "metadata" not in d  # 空字典不应该出现


class TestBatchTokenizerResult:
    """测试BatchTokenizerResult"""
    
    def test_basic_creation(self):
        """测试基本创建"""
        result = BatchTokenizerResult(
            token_ids=[[1, 2], [3, 4, 5]],
            tokens=[["a", "b"], ["c", "d", "e"]]
        )
        assert len(result) == 2
    
    def test_indexing(self):
        """测试索引访问"""
        result = BatchTokenizerResult(
            token_ids=[[1, 2], [3, 4]],
            tokens=[["a", "b"], ["c", "d"]],
            attention_mask=[[1, 1], [1, 0]]
        )
        single = result[0]
        assert isinstance(single, TokenizerResult)
        assert single.token_ids == [1, 2]
        assert single.tokens == ["a", "b"]
        assert single.attention_mask == [1, 1]


# ==================== BPETokenizer 测试 ====================

class TestBPETokenizer:
    """测试BPETokenizer"""
    
    @pytest.fixture
    def tokenizer(self):
        """创建默认分词器"""
        return BPETokenizer(encoding_name="cl100k_base")
    
    def test_initialization(self, tokenizer):
        """测试初始化"""
        assert tokenizer.vocab_size > 0
        assert tokenizer._tiktoken_encoder is not None
    
    def test_basic_encode(self, tokenizer):
        """测试基本编码"""
        result = tokenizer.encode("Hello, world!")
        assert len(result.token_ids) > 0
        assert len(result.tokens) == len(result.token_ids)
        assert result.attention_mask is not None
        assert all(m == 1 for m in result.attention_mask)
    
    def test_encode_without_special_tokens(self, tokenizer):
        """测试不添加特殊token的编码"""
        with_special = tokenizer.encode("test", add_special_tokens=True)
        without_special = tokenizer.encode("test", add_special_tokens=False)
        
        # 有特殊token时长度应该更长
        assert len(with_special.tokens) > len(without_special.tokens)
        assert "[CLS]" in with_special.tokens
        assert "[CLS]" not in without_special.tokens
    
    def test_decode(self, tokenizer):
        """测试解码"""
        original = "Hello world"
        result = tokenizer.encode(original, add_special_tokens=False)
        decoded = tokenizer.decode(result.token_ids)
        assert decoded == original
    
    def test_encode_decode_roundtrip(self, tokenizer):
        """测试编解码往返一致性"""
        texts = [
            "Simple text",
            "Numbers: 123456",
            "Special chars: @#$%",
            "Unicode: 你好世界",
            "Mixed: Hello 世界 123!",
        ]
        for text in texts:
            result = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(result.token_ids)
            assert decoded == text, f"Roundtrip failed for: {text}"
    
    def test_truncation(self, tokenizer):
        """测试截断功能"""
        long_text = "word " * 100
        result = tokenizer.encode(
            long_text, 
            max_length=10, 
            truncation=True,
            add_special_tokens=False
        )
        assert len(result.token_ids) <= 10
    
    def test_padding(self, tokenizer):
        """测试填充功能"""
        result = tokenizer.encode(
            "short",
            max_length=20,
            padding=True,
            add_special_tokens=False
        )
        assert len(result.token_ids) == 20
        assert result.attention_mask is not None
        # 填充部分的mask应该是0
        assert 0 in result.attention_mask
    
    def test_encode_with_query(self, tokenizer):
        """测试带问题的编码"""
        result = tokenizer.encode(
            "The weather is sunny today.",
            query="What is the weather?"
        )
        assert result.attention_weights is not None
        assert len(result.attention_weights) == len(result.tokens)
        # 权重应该在合理范围内
        assert all(0 <= w <= 2.0 for w in result.attention_weights)
    
    def test_encode_batch(self, tokenizer):
        """测试批量编码"""
        texts = ["Hello", "World", "Test"]
        result = tokenizer.encode_batch(texts, add_special_tokens=False)
        
        assert len(result) == 3
        assert len(result.token_ids) == 3
    
    def test_decode_batch(self, tokenizer):
        """测试批量解码"""
        texts = ["Hello", "World"]
        encoded = tokenizer.encode_batch(texts, add_special_tokens=False)
        decoded = tokenizer.decode_batch(encoded.token_ids)
        
        assert decoded == texts
    
    def test_empty_string(self, tokenizer):
        """测试空字符串处理"""
        result = tokenizer.encode("", add_special_tokens=False)
        assert result.token_ids == []
        assert result.tokens == []
    
    def test_repr(self, tokenizer):
        """测试字符串表示"""
        repr_str = repr(tokenizer)
        assert "BPETokenizer" in repr_str
        assert "cl100k_base" in repr_str


class TestBPETokenizerCustom:
    """测试自定义BPE分词器"""
    
    def test_custom_tokenizer_initialization(self):
        """测试自定义分词器初始化"""
        # 不使用tiktoken，使用默认词表
        tokenizer = BPETokenizer(encoding_name=None)
        assert tokenizer.vocab_size > 0
    
    def test_train_on_corpus(self):
        """测试在语料上训练"""
        tokenizer = BPETokenizer(encoding_name=None)
        corpus = [
            "hello world",
            "hello there",
            "world hello",
        ]
        tokenizer.train(corpus, vocab_size=30, min_frequency=1, show_progress=False)
        
        assert tokenizer.vocab_size >= 10
        assert len(tokenizer._merges) > 0
    
    def test_save_and_load(self):
        """测试保存和加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建并保存
            tokenizer = BPETokenizer(encoding_name="cl100k_base")
            tokenizer.save(tmpdir)
            
            # 验证文件存在
            assert os.path.exists(os.path.join(tmpdir, "config.json"))
            
            # 加载
            loaded = BPETokenizer.load(tmpdir)
            assert loaded.vocab_size == tokenizer.vocab_size


# ==================== TokenAttentionMixin 测试 ====================

class TestTokenAttentionMixin:
    """测试注意力权重计算"""
    
    @pytest.fixture
    def tokenizer(self):
        """创建带注意力功能的分词器"""
        return BPETokenizer(encoding_name="cl100k_base")
    
    def test_keyword_match_weights(self, tokenizer):
        """测试关键词匹配策略"""
        doc_tokens = ["the", "weather", "is", "sunny", "today"]
        weights = tokenizer.compute_attention_weights(
            doc_tokens, 
            "weather",
            strategy=AttentionStrategy.KEYWORD_MATCH
        )
        
        assert len(weights) == len(doc_tokens)
        # "weather" 应该有最高权重
        weather_idx = doc_tokens.index("weather")
        assert weights[weather_idx] >= max(weights[i] for i in range(len(doc_tokens)) if i != weather_idx)
    
    def test_bm25_weights(self, tokenizer):
        """测试BM25策略"""
        doc_tokens = ["python", "is", "a", "programming", "language"]
        weights = tokenizer.compute_attention_weights(
            doc_tokens,
            "What is Python?",
            strategy=AttentionStrategy.BM25
        )
        
        assert len(weights) == len(doc_tokens)
        # Python和is应该有较高权重
        python_idx = doc_tokens.index("python")
        assert weights[python_idx] > tokenizer.attention_config.min_weight
    
    def test_tfidf_weights(self, tokenizer):
        """测试TF-IDF策略"""
        doc_tokens = ["machine", "learning", "is", "powerful"]
        weights = tokenizer.compute_attention_weights(
            doc_tokens,
            "learning",
            strategy=AttentionStrategy.TFIDF
        )
        
        assert len(weights) == len(doc_tokens)
    
    def test_attention_config(self):
        """测试自定义注意力配置"""
        config = AttentionConfig(
            strategy=AttentionStrategy.KEYWORD_MATCH,
            normalize=True,
            min_weight=0.2,
            boost_factor=3.0
        )
        tokenizer = BPETokenizer(
            encoding_name="cl100k_base",
            attention_config=config
        )
        
        doc_tokens = ["hello", "world"]
        weights = tokenizer.compute_attention_weights(doc_tokens, "hello")
        
        # 最小权重应该是0.2
        assert all(w >= 0.2 for w in weights)
    
    def test_update_document_statistics(self, tokenizer):
        """测试文档统计更新"""
        documents = [
            ["hello", "world"],
            ["hello", "python"],
            ["world", "python", "programming"],
        ]
        tokenizer.update_document_statistics(documents)
        
        assert tokenizer._total_documents == 3
        assert tokenizer._avg_doc_length > 0
        assert "hello" in tokenizer._document_frequencies


class TestQueryFocusedAttention:
    """测试独立的注意力计算器"""
    
    def test_basic_compute(self):
        """测试基本计算"""
        calculator = QueryFocusedAttention()
        weights = calculator.compute(
            ["the", "cat", "sat"],
            "cat"
        )
        assert len(weights) == 3
    
    def test_apply_weights(self):
        """测试权重应用到embeddings"""
        calculator = QueryFocusedAttention()
        embeddings = [[1.0, 2.0], [3.0, 4.0]]
        weights = [0.5, 1.0]
        
        weighted = calculator.apply_weights(embeddings, weights)
        
        assert weighted[0] == [0.5, 1.0]
        assert weighted[1] == [3.0, 4.0]
    
    def test_apply_weights_length_mismatch(self):
        """测试长度不匹配时的错误处理"""
        calculator = QueryFocusedAttention()
        embeddings = [[1.0, 2.0], [3.0, 4.0]]
        weights = [0.5]  # 长度不匹配
        
        with pytest.raises(ValueError):
            calculator.apply_weights(embeddings, weights)


# ==================== TokenizerFactory 测试 ====================

class TestTokenizerFactory:
    """测试分词器工厂"""
    
    def test_create_bpe(self):
        """测试创建BPE分词器"""
        tokenizer = TokenizerFactory.create("bpe")
        assert isinstance(tokenizer, BPETokenizer)
    
    def test_create_with_shortcut(self):
        """测试使用快捷方式创建"""
        tokenizer = TokenizerFactory.create("gpt4")
        assert isinstance(tokenizer, BPETokenizer)
    
    def test_create_unknown_type(self):
        """测试创建未知类型时的错误"""
        with pytest.raises(ValueError) as excinfo:
            TokenizerFactory.create("unknown_type")
        assert "未知的分词器类型" in str(excinfo.value)
    
    def test_register_and_create(self):
        """测试注册和创建自定义分词器"""
        # 注册
        class CustomTokenizer(BPETokenizer):
            pass
        
        TokenizerFactory.register("custom", CustomTokenizer)
        
        # 创建
        tokenizer = TokenizerFactory.create("custom")
        assert isinstance(tokenizer, CustomTokenizer)
        
        # 清理
        TokenizerFactory.unregister("custom")
    
    def test_register_invalid_class(self):
        """测试注册无效类时的错误"""
        class NotATokenizer:
            pass
        
        with pytest.raises(TypeError):
            TokenizerFactory.register("invalid", NotATokenizer)
    
    def test_list_available(self):
        """测试列出可用分词器"""
        available = TokenizerFactory.list_available()
        
        assert "bpe" in available
        assert "gpt4" in available
        assert len(available) >= 4
    
    def test_from_config(self):
        """测试从配置文件创建"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "type": "bpe",
                "params": {"encoding_name": "cl100k_base"}
            }, f)
            config_path = f.name
        
        try:
            tokenizer = TokenizerFactory.from_config(config_path)
            assert isinstance(tokenizer, BPETokenizer)
        finally:
            os.unlink(config_path)
    
    def test_get_tokenizer_function(self):
        """测试便捷函数"""
        tokenizer = get_tokenizer("gpt4")
        assert isinstance(tokenizer, BPETokenizer)


# ==================== 边界情况和错误处理测试 ====================

class TestEdgeCases:
    """测试边界情况"""
    
    @pytest.fixture
    def tokenizer(self):
        return BPETokenizer(encoding_name="cl100k_base")
    
    def test_very_long_text(self, tokenizer):
        """测试非常长的文本"""
        long_text = "word " * 10000
        result = tokenizer.encode(long_text, add_special_tokens=False)
        assert len(result.token_ids) > 0
    
    def test_special_characters(self, tokenizer):
        """测试特殊字符"""
        text = "Hello! @#$%^&*()_+-=[]{}|;':\",./<>?"
        result = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(result.token_ids)
        assert decoded == text
    
    def test_unicode_text(self, tokenizer):
        """测试Unicode文本"""
        texts = [
            "中文测试",
            "日本語テスト",
            "한국어 테스트",
            "مرحبا",
            "🚀💻🎉",
        ]
        for text in texts:
            result = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(result.token_ids)
            assert decoded == text, f"Failed for: {text}"
    
    def test_whitespace_handling(self, tokenizer):
        """测试空白字符处理"""
        text = "  multiple   spaces   "
        result = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(result.token_ids)
        # 解码后应该保持一致
        assert text in decoded or decoded.strip() == text.strip()
    
    def test_query_with_no_match(self, tokenizer):
        """测试问题与文档无匹配时"""
        result = tokenizer.encode(
            "The cat sat on the mat",
            query="quantum physics"
        )
        # 没有匹配时，所有权重应该相等（都是归一化后的值）
        assert result.attention_weights is not None
        weights = result.attention_weights
        # 检查所有权重值相同（因为没有匹配，都应该是相同的基础权重）
        non_special_weights = [w for w in weights if w is not None]
        assert len(set(non_special_weights)) <= 2  # 最多2种不同值（特殊token可能不同）


# ==================== 集成测试 ====================

class TestIntegration:
    """集成测试"""
    
    def test_full_workflow(self):
        """测试完整工作流程"""
        # 1. 使用工厂创建分词器
        tokenizer = get_tokenizer("gpt4")
        
        # 2. 编码文本
        text = "Machine learning is transforming the world."
        query = "What is machine learning?"
        result = tokenizer.encode(text, query=query)
        
        # 3. 验证结果
        assert len(result.token_ids) > 0
        assert result.attention_weights is not None
        assert result.attention_mask is not None
        assert result.metadata.get("encoding") == "cl100k_base"
        
        # 4. 解码验证
        decoded = tokenizer.decode(result.token_ids)
        assert "machine" in decoded.lower()
    
    def test_batch_workflow(self):
        """测试批量处理工作流程"""
        tokenizer = get_tokenizer("gpt4")
        
        texts = [
            "First document about Python",
            "Second document about Java",
            "Third document about JavaScript",
        ]
        queries = [
            "Python programming",
            "Java development",
            "JavaScript frameworks",
        ]
        
        results = tokenizer.encode_batch(texts, queries=queries)
        
        assert len(results) == 3
        for i in range(3):
            single = results[i]
            assert single.attention_weights is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
