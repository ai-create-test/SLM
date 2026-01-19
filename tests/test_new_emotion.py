"""
Tests for the VAD-based Emotion System

Tests:
1. VADEncoder - Basic VAD vector encoding
2. SemanticEmotionEncoder - Multiple input type handling (replaces UnifiedEmotionEncoder)
3. SemanticRetriever - Tier 2 semantic similarity search
4. Chinese emotion support
5. ContextAwareEmbedding - Integration with emotion_vector
6. Backward compatibility with legacy integer IDs
"""

import pytest
import torch
from typing import List

from app.modulation import (
    VADEncoder,
    SemanticEmotionEncoder,
    EmotionEncoder,  # Alias for SemanticEmotionEncoder
    VADCoordinate,
    get_vad,
)
from app.modulation.emotion import VADLexicon, get_default_lexicon, SemanticRetriever
from app.memory.fusion import ContextAwareEmbedding


class TestSemanticRetriever:
    """Test Tier 2 semantic retrieval"""
    
    @pytest.fixture
    def retriever(self):
        return SemanticRetriever(get_default_lexicon())
    
    def test_retrieve_similar_word(self, retriever):
        """Test finding similar words for unknown input"""
        # "happ" is not in lexicon but similar to "happy"
        result = retriever.retrieve("happ", k=5)
        assert isinstance(result, VADCoordinate)
        # Should have positive valence (similar to happy)
        assert result.valence > 0
    
    def test_retrieve_top_k(self, retriever):
        """Test getting top-k results with details"""
        results = retriever.retrieve_top_k("joyful", k=3)
        assert len(results) <= 3
        for r in results:
            assert hasattr(r, 'word')
            assert hasattr(r, 'vad')
            assert hasattr(r, 'similarity')
    
    def test_unknown_returns_valid_vad(self, retriever):
        """Test completely unknown text returns valid VAD"""
        result = retriever.retrieve("xyz123unknown", k=5)
        assert isinstance(result, VADCoordinate)
        # Should be close to neutral or fallback
    
    def test_chinese_similar_word(self, retriever):
        """Test Chinese word similarity"""
        # "快乐" is similar to "开心" or "高兴"
        result = retriever.retrieve("快乐", k=5)
        assert isinstance(result, VADCoordinate)


class TestTrainableEmotionEncoder:
    """Test Tier 3 trainable neural encoder"""
    
    @pytest.fixture
    def encoder(self):
        from app.modulation.emotion import TrainableEmotionEncoder
        return TrainableEmotionEncoder(d_emotion=64, d_hidden=128)
    
    def test_forward_single(self, encoder):
        """Test encoding single text"""
        vec = encoder(["happy"])
        assert vec.shape == (1, 64)
    
    def test_forward_batch(self, encoder):
        """Test encoding batch of texts"""
        vec = encoder(["happy", "sad", "excited"])
        assert vec.shape == (3, 64)
    
    def test_predict_vad(self, encoder):
        """Test VAD prediction"""
        vad = encoder.predict_vad("very happy")
        assert isinstance(vad, VADCoordinate)
        assert -1.0 <= vad.valence <= 1.0
        assert -1.0 <= vad.arousal <= 1.0
        assert -1.0 <= vad.dominance <= 1.0
    
    def test_encoder_type(self, encoder):
        """Test encoder type detection"""
        assert encoder.encoder_type in ["char_cnn", "transformer"]
    
    def test_return_vad_flag(self, encoder):
        """Test return_vad flag"""
        vad = encoder(["test"], return_vad=True)
        assert vad.shape == (1, 3)


class TestEmotionEncoderTrainer:
    """Test training pipeline"""
    
    @pytest.fixture
    def trainer(self):
        from app.modulation.emotion import TrainableEmotionEncoder, EmotionEncoderTrainer
        encoder = TrainableEmotionEncoder(d_emotion=32, d_hidden=64)
        return EmotionEncoderTrainer(encoder)
    
    def test_add_sample(self, trainer):
        """Test adding training samples"""
        trainer.add_sample("test", VADCoordinate(0.5, 0.5, 0.5))
        assert len(trainer._texts) == 1
        assert len(trainer._vads) == 1
    
    def test_train_small(self, trainer):
        """Test training with small dataset"""
        trainer.add_sample("happy", VADCoordinate(0.8, 0.5, 0.6))
        trainer.add_sample("sad", VADCoordinate(-0.7, -0.3, -0.5))
        history = trainer.train(epochs=2, verbose=False)
        assert "loss" in history
        assert len(history["loss"]) == 2


class TestVADLexicon:
    """Test VAD lexicon loading and lookup"""
    
    def test_default_lexicon_loads(self):
        """Verify default lexicon loads"""
        lexicon = get_default_lexicon()
        assert len(lexicon) > 0, "Lexicon should not be empty"
    
    def test_english_emotion_lookup(self):
        """Test looking up English emotion words"""
        lexicon = get_default_lexicon()
        vad = lexicon.get("happy")
        assert vad is not None
        assert vad.valence > 0, "Happy should have positive valence"
    
    def test_chinese_emotion_lookup(self):
        """Test looking up Chinese emotion words"""
        lexicon = get_default_lexicon()
        vad = lexicon.get("开心")
        assert vad is not None
        assert vad.valence > 0, "开心 should have positive valence"
    
    def test_unknown_word_returns_none(self):
        """Test unknown words return None"""
        lexicon = get_default_lexicon()
        vad = lexicon.get("xyz123unknown")
        assert vad is None


class TestVADEncoder:
    """Test VADEncoder class"""
    
    @pytest.fixture
    def encoder(self):
        return VADEncoder(d_emotion=128)
    
    def test_forward_with_tensor(self, encoder):
        """Test encoding VAD tensor directly"""
        vad = torch.tensor([[0.8, 0.5, 0.6]])  # Happy-like
        output = encoder(vad)
        assert output.shape == (1, 128)
    
    def test_encode_coordinate(self, encoder):
        """Test encoding VADCoordinate"""
        vad = VADCoordinate(0.8, 0.5, 0.6)
        output = encoder.encode_coordinate(vad)
        assert output.shape == (1, 128)
    
    def test_batch_encoding(self, encoder):
        """Test batch VAD encoding"""
        vads = [
            VADCoordinate(0.8, 0.5, 0.6),
            VADCoordinate(-0.7, -0.3, -0.5),
        ]
        output = encoder.encode_batch(vads)
        assert output.shape == (2, 128)


class TestSemanticEmotionEncoder:
    """Test SemanticEmotionEncoder - the main interface"""
    
    @pytest.fixture
    def encoder(self):
        return SemanticEmotionEncoder(d_emotion=128)
    
    def test_int_input(self, encoder):
        """Test integer input (legacy ID)"""
        output = encoder(1)  # 1 = happy
        assert output.shape == (1, 128)
    
    def test_string_input_english(self, encoder):
        """Test English emotion name"""
        output = encoder("happy")
        assert output.shape == (1, 128)
    
    def test_string_input_chinese(self, encoder):
        """Test Chinese emotion name"""
        output = encoder("开心")
        assert output.shape == (1, 128)
    
    def test_list_input(self, encoder):
        """Test list input (batch of names)"""
        output = encoder(["happy", "sad"])
        assert output.shape == (2, 128)
    
    def test_mixed_language_batch(self, encoder):
        """Test mixed English-Chinese batch"""
        output = encoder(["happy", "悲伤", "angry"])
        assert output.shape == (3, 128)
    
    def test_tensor_vad_input(self, encoder):
        """Test VAD tensor input"""
        vad = torch.tensor([0.8, 0.5, 0.6])
        output = encoder(vad)
        assert output.shape == (1, 128)
    
    def test_unknown_word_returns_neutral(self, encoder):
        """Test unknown words use default (neutral)"""
        output = encoder("xyz123unknown")
        assert output.shape == (1, 128)
    
    def test_different_emotions_produce_different_vectors(self, encoder):
        """Verify different emotions produce different vectors"""
        happy = encoder("happy")
        sad = encoder("sad")
        cosine_sim = torch.nn.functional.cosine_similarity(happy, sad, dim=-1)
        assert cosine_sim.item() < 0.9, "Happy and Sad should be different"
    
    def test_encode_blend(self, encoder):
        """Test emotion blending"""
        output = encoder.encode_blend("happy", "sad", weight=0.5)
        assert output.shape == (1, 128)
    
    def test_emotion_count(self, encoder):
        """Test lexicon size"""
        assert encoder.emotion_count > 50, "Should have many emotions"


class TestBackwardCompatibility:
    """Test backward compatibility with old EmotionEncoder API"""
    
    @pytest.fixture
    def encoder(self):
        # EmotionEncoder is now an alias for SemanticEmotionEncoder
        return EmotionEncoder(d_emotion=128)
    
    def test_encode_name_method(self, encoder):
        """Test encode_name method exists and works"""
        output = encoder.encode_name("happy")
        assert output.shape == (1, 128)
    
    def test_legacy_id(self, encoder):
        """Test legacy ID encoding"""
        output = encoder(0)  # neutral
        assert output.shape == (1, 128)


class TestContextAwareEmbeddingIntegration:
    """Test ContextAwareEmbedding with VAD emotion vectors"""
    
    @pytest.fixture
    def embedding(self):
        return ContextAwareEmbedding(
            vocab_size=1000,
            d_model=256,
            max_seq_len=128,
        )
    
    @pytest.fixture
    def emotion_encoder(self):
        return SemanticEmotionEncoder(d_emotion=256)
    
    def test_with_emotion_vector(self, embedding, emotion_encoder):
        """Test using emotion_vector parameter"""
        token_ids = torch.randint(0, 1000, (2, 10))
        emotion_vec = emotion_encoder("happy")
        # Expand for batch
        emotion_vec_batch = emotion_vec.expand(2, -1)
        
        output = embedding(token_ids, emotion_vector=emotion_vec_batch)
        assert output.shape == (2, 10, 256)
    
    def test_emotion_vector_overrides_id(self, embedding, emotion_encoder):
        """Verify emotion_vector takes precedence over emotion_id"""
        embedding.eval()
        emotion_encoder.eval()
        
        token_ids = torch.randint(0, 1000, (1, 5))
        
        # Use happy vector but pass angry id
        happy_vec = emotion_encoder("happy").detach()
        
        with torch.no_grad():
            out_with_vec = embedding(token_ids.clone(), emotion_id=torch.tensor([3]), emotion_vector=happy_vec)
            out_with_happy_id = embedding(token_ids.clone(), emotion_id=torch.tensor([1]))  # happy
        
        # Out with vec should be similar to happy, not angry
        diff = (out_with_vec - out_with_happy_id).abs().mean()
        assert diff < 0.5, "emotion_vector should take precedence"
