"""
Semantic Retriever - Tier 2 语义相似度检索

当 Tier 1 词典精确匹配失败时，使用语义相似度检索最接近的情感词。

策略:
1. 如果安装了 sentence-transformers: 使用预训练模型编码
2. 否则: 使用字符级 n-gram 相似度作为轻量级回退

设计原则:
- 懒加载: 只在第一次调用时构建索引
- 可选依赖: 不强制要求 sentence-transformers
- 缓存: 缓存编码结果避免重复计算
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import torch

from .vad_lexicon import VADLexicon, VADCoordinate, get_default_lexicon


@dataclass
class RetrievalResult:
    """检索结果"""
    word: str
    vad: VADCoordinate
    similarity: float


class SemanticRetriever:
    """
    语义相似度检索器
    
    当精确匹配失败时，找到语义最相似的词并返回其 VAD。
    
    用法:
        retriever = SemanticRetriever(lexicon)
        vad = retriever.retrieve("elated")  # 找到 "happy" 或 "excited"
        vad = retriever.retrieve("难过")    # 找到 "悲伤"
    """
    
    def __init__(
        self,
        lexicon: Optional[VADLexicon] = None,
        use_neural: bool = True,
        cache_size: int = 1000,
    ):
        """
        Args:
            lexicon: VAD 词典 (默认使用全局词典)
            use_neural: 是否尝试使用神经网络编码器
            cache_size: 缓存大小
        """
        self.lexicon = lexicon or get_default_lexicon()
        self.use_neural = use_neural
        self.cache_size = cache_size
        
        # 懒加载
        self._encoder = None
        self._word_vectors: Optional[torch.Tensor] = None
        self._word_list: Optional[List[str]] = None
        self._cache: Dict[str, VADCoordinate] = {}
        self._initialized = False
    
    def _initialize(self) -> None:
        """懒加载初始化"""
        if self._initialized:
            return
        
        self._word_list = self.lexicon.all_words()
        
        # 尝试加载神经网络编码器
        if self.use_neural:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                self._word_vectors = self._encode_words_neural(self._word_list)
            except ImportError:
                self._encoder = None
                self._word_vectors = None
        
        self._initialized = True
    
    def _encode_words_neural(self, words: List[str]) -> torch.Tensor:
        """使用神经网络编码词列表"""
        embeddings = self._encoder.encode(words, convert_to_tensor=True)
        return embeddings
    
    def _encode_text_neural(self, text: str) -> torch.Tensor:
        """使用神经网络编码单个文本"""
        return self._encoder.encode([text], convert_to_tensor=True)[0]
    
    def retrieve(self, text: str, k: int = 5) -> VADCoordinate:
        """
        检索最相似情感词的 VAD
        
        Args:
            text: 查询文本
            k: 考虑的最相似词数量
            
        Returns:
            加权平均后的 VADCoordinate
        """
        # 检查缓存
        cache_key = text.lower().strip()
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 初始化 (懒加载)
        self._initialize()
        
        # 选择检索策略
        if self._encoder is not None and self._word_vectors is not None:
            result = self._retrieve_neural(text, k)
        else:
            result = self._retrieve_ngram(text, k)
        
        # 缓存结果
        if len(self._cache) < self.cache_size:
            self._cache[cache_key] = result
        
        return result
    
    def _retrieve_neural(self, text: str, k: int) -> VADCoordinate:
        """使用神经网络嵌入进行检索"""
        query_vec = self._encode_text_neural(text)
        
        # 计算余弦相似度
        similarities = torch.nn.functional.cosine_similarity(
            query_vec.unsqueeze(0), 
            self._word_vectors, 
            dim=1
        )
        
        # 获取 top-k
        top_k_values, top_k_indices = torch.topk(similarities, min(k, len(self._word_list)))
        
        # 加权平均 VAD
        return self._weighted_average_vad(top_k_indices.tolist(), top_k_values.tolist())
    
    def _retrieve_ngram(self, text: str, k: int) -> VADCoordinate:
        """使用 n-gram 相似度进行检索 (轻量级回退)"""
        query_ngrams = self._get_ngrams(text.lower())
        
        similarities = []
        for word in self._word_list:
            word_ngrams = self._get_ngrams(word.lower())
            sim = self._jaccard_similarity(query_ngrams, word_ngrams)
            similarities.append(sim)
        
        # 获取 top-k
        indexed_sims = list(enumerate(similarities))
        indexed_sims.sort(key=lambda x: x[1], reverse=True)
        top_k = indexed_sims[:min(k, len(indexed_sims))]
        
        indices = [idx for idx, _ in top_k]
        scores = [score for _, score in top_k]
        
        return self._weighted_average_vad(indices, scores)
    
    def _get_ngrams(self, text: str, n: int = 2) -> set:
        """获取字符 n-grams"""
        text = text.lower().strip()
        if len(text) < n:
            return {text}
        return set(text[i:i+n] for i in range(len(text) - n + 1))
    
    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """计算 Jaccard 相似度"""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def _weighted_average_vad(
        self, 
        indices: List[int], 
        scores: List[float]
    ) -> VADCoordinate:
        """计算加权平均 VAD"""
        if not indices or not scores or sum(scores) == 0:
            # 没有匹配，返回 neutral
            return VADCoordinate(0.0, 0.0, 0.0, source="fallback")
        
        # 归一化权重
        total = sum(scores)
        weights = [s / total for s in scores]
        
        # 加权平均
        v_sum = 0.0
        a_sum = 0.0
        d_sum = 0.0
        
        for idx, weight in zip(indices, weights):
            word = self._word_list[idx]
            vad = self.lexicon.get(word)
            if vad:
                v_sum += vad.valence * weight
                a_sum += vad.arousal * weight
                d_sum += vad.dominance * weight
        
        return VADCoordinate(
            valence=v_sum,
            arousal=a_sum,
            dominance=d_sum,
            source="semantic_retrieval",
        )
    
    def retrieve_top_k(self, text: str, k: int = 5) -> List[RetrievalResult]:
        """返回 top-k 检索结果 (带详情)"""
        self._initialize()
        
        if self._encoder is not None and self._word_vectors is not None:
            query_vec = self._encode_text_neural(text)
            similarities = torch.nn.functional.cosine_similarity(
                query_vec.unsqueeze(0), 
                self._word_vectors, 
                dim=1
            )
            top_k_values, top_k_indices = torch.topk(similarities, min(k, len(self._word_list)))
            
            results = []
            for idx, score in zip(top_k_indices.tolist(), top_k_values.tolist()):
                word = self._word_list[idx]
                vad = self.lexicon.get(word)
                if vad:
                    results.append(RetrievalResult(word=word, vad=vad, similarity=score))
            return results
        else:
            # N-gram fallback
            query_ngrams = self._get_ngrams(text.lower())
            indexed_sims = []
            for i, word in enumerate(self._word_list):
                sim = self._jaccard_similarity(query_ngrams, self._get_ngrams(word.lower()))
                indexed_sims.append((i, sim))
            
            indexed_sims.sort(key=lambda x: x[1], reverse=True)
            top_k = indexed_sims[:min(k, len(indexed_sims))]
            
            results = []
            for idx, score in top_k:
                word = self._word_list[idx]
                vad = self.lexicon.get(word)
                if vad:
                    results.append(RetrievalResult(word=word, vad=vad, similarity=score))
            return results
    
    @property
    def is_neural_available(self) -> bool:
        """检查神经网络编码器是否可用"""
        self._initialize()
        return self._encoder is not None
