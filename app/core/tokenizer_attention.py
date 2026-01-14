"""
Token Attention Module - 问题导向注意力权重计算

核心设计思想：
当用户提问时，根据问题关键词为文档tokens分配不同权重，
使模型在理解下文时有不同侧重点。

支持多种计算策略：
1. 关键词匹配 (keyword_match) - 快速但精度有限
2. TF-IDF权重 (tfidf) - 考虑词频和逆文档频率
3. BM25算法 (bm25) - 工业级检索算法
4. 语义相似度 (semantic) - 基于embedding的高级策略（预留）
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math
import re
from collections import Counter


class AttentionStrategy(Enum):
    """注意力权重计算策略枚举"""
    KEYWORD_MATCH = "keyword_match"
    TFIDF = "tfidf"
    BM25 = "bm25"
    SEMANTIC = "semantic"  # 预留：需要embedding模型支持


@dataclass
class AttentionConfig:
    """注意力权重计算配置"""
    strategy: AttentionStrategy = AttentionStrategy.BM25
    normalize: bool = True  # 是否归一化权重
    min_weight: float = 0.1  # 最小权重值（避免完全忽略）
    boost_factor: float = 2.0  # 匹配时的提升因子
    
    # BM25 参数（工业标准值）
    bm25_k1: float = 1.5  # 词频饱和参数
    bm25_b: float = 0.75  # 文档长度归一化参数


class TokenAttentionMixin:
    """
    问题导向注意力权重混入类
    
    此Mixin为Tokenizer提供注意力权重计算能力，
    是实现"问题导向检索阅览"的关键组件。
    
    使用方式：
        class MyTokenizer(BaseTokenizer, TokenAttentionMixin):
            def encode(self, text, query=None, ...):
                result = self._base_encode(text)
                if query:
                    result.attention_weights = self.compute_attention_weights(
                        result.tokens, query
                    )
                return result
    """
    
    def __init__(self, attention_config: Optional[AttentionConfig] = None, **kwargs):
        self.attention_config = attention_config or AttentionConfig()
        # 用于TF-IDF/BM25的文档频率统计（可后续更新）
        self._document_frequencies: Dict[str, int] = {}
        self._total_documents: int = 0
        self._avg_doc_length: float = 0.0
        super().__init__(**kwargs)
    
    def compute_attention_weights(
        self,
        doc_tokens: List[str],
        query: str,
        strategy: Optional[AttentionStrategy] = None,
    ) -> List[float]:
        """
        计算问题导向的注意力权重
        
        Args:
            doc_tokens: 文档的token列表
            query: 用户问题文本
            strategy: 计算策略（默认使用配置中的策略）
            
        Returns:
            每个token的注意力权重列表
        """
        strategy = strategy or self.attention_config.strategy
        
        # 对query进行简单分词（后续可使用相同tokenizer）
        query_tokens = self._simple_tokenize(query)
        
        if strategy == AttentionStrategy.KEYWORD_MATCH:
            weights = self._keyword_match_weights(doc_tokens, query_tokens)
        elif strategy == AttentionStrategy.TFIDF:
            weights = self._tfidf_weights(doc_tokens, query_tokens)
        elif strategy == AttentionStrategy.BM25:
            weights = self._bm25_weights(doc_tokens, query_tokens)
        elif strategy == AttentionStrategy.SEMANTIC:
            weights = self._semantic_weights(doc_tokens, query_tokens)
        else:
            # 默认均匀权重
            weights = [1.0] * len(doc_tokens)
        
        # 归一化
        if self.attention_config.normalize and weights:
            max_weight = max(weights)
            if max_weight > 0:
                weights = [w / max_weight for w in weights]
        
        # 应用最小权重
        weights = [max(w, self.attention_config.min_weight) for w in weights]
        
        return weights
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """简单分词（用于query处理）"""
        # 转小写并按非字母数字分割
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _keyword_match_weights(
        self, 
        doc_tokens: List[str], 
        query_tokens: List[str]
    ) -> List[float]:
        """
        关键词匹配策略
        
        简单但高效：匹配的token获得更高权重
        """
        query_set = set(t.lower() for t in query_tokens)
        weights = []
        
        for token in doc_tokens:
            token_lower = token.lower()
            if token_lower in query_set:
                weights.append(self.attention_config.boost_factor)
            else:
                weights.append(1.0)
        
        return weights
    
    def _tfidf_weights(
        self, 
        doc_tokens: List[str], 
        query_tokens: List[str]
    ) -> List[float]:
        """
        TF-IDF权重策略
        
        考虑词频(TF)和逆文档频率(IDF)
        TF = 词在文档中出现次数 / 文档总词数
        IDF = log(总文档数 / 包含该词的文档数)
        """
        query_set = set(t.lower() for t in query_tokens)
        doc_tokens_lower = [t.lower() for t in doc_tokens]
        
        # 计算词频
        tf_counter = Counter(doc_tokens_lower)
        doc_length = len(doc_tokens_lower)
        
        weights = []
        for token in doc_tokens_lower:
            if token in query_set:
                # TF分量
                tf = tf_counter[token] / doc_length if doc_length > 0 else 0
                
                # IDF分量（如果没有统计数据，使用默认值）
                if self._total_documents > 0 and token in self._document_frequencies:
                    idf = math.log(self._total_documents / (1 + self._document_frequencies[token]))
                else:
                    idf = 1.0  # 默认IDF
                
                weight = (1 + tf) * idf * self.attention_config.boost_factor
            else:
                weight = 1.0
            
            weights.append(weight)
        
        return weights
    
    def _bm25_weights(
        self, 
        doc_tokens: List[str], 
        query_tokens: List[str]
    ) -> List[float]:
        """
        BM25权重策略 - 工业级检索算法
        
        BM25是信息检索领域的标准算法，在Elasticsearch、Lucene等系统中广泛使用。
        
        公式：
        score(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl))
        
        其中：
        - f(qi,D): 词qi在文档D中的频率
        - |D|: 文档长度
        - avgdl: 平均文档长度
        - k1, b: 调节参数
        """
        k1 = self.attention_config.bm25_k1
        b = self.attention_config.bm25_b
        
        query_set = set(t.lower() for t in query_tokens)
        doc_tokens_lower = [t.lower() for t in doc_tokens]
        
        # 计算词频
        tf_counter = Counter(doc_tokens_lower)
        doc_length = len(doc_tokens_lower)
        
        # 平均文档长度（如果没有统计，使用当前文档长度）
        avg_dl = self._avg_doc_length if self._avg_doc_length > 0 else doc_length
        
        weights = []
        for token in doc_tokens_lower:
            if token in query_set:
                # 词频
                tf = tf_counter[token]
                
                # IDF分量
                if self._total_documents > 0 and token in self._document_frequencies:
                    N = self._total_documents
                    n = self._document_frequencies.get(token, 0)
                    idf = math.log((N - n + 0.5) / (n + 0.5) + 1)
                else:
                    idf = 1.0
                
                # BM25分数
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_length / avg_dl)
                weight = idf * (numerator / denominator) * self.attention_config.boost_factor
            else:
                weight = 1.0
            
            weights.append(weight)
        
        return weights
    
    def _semantic_weights(
        self, 
        doc_tokens: List[str], 
        query_tokens: List[str]
    ) -> List[float]:
        """
        语义相似度策略（预留接口）
        
        需要embedding模型支持，后续可扩展实现。
        可以使用：
        - Sentence-BERT
        - OpenAI Embeddings
        - 自定义embedding模型
        """
        # TODO: 实现语义相似度计算
        # 当前返回均匀权重
        return [1.0] * len(doc_tokens)
    
    def update_document_statistics(
        self, 
        documents: List[List[str]]
    ) -> None:
        """
        更新文档统计信息（用于TF-IDF和BM25）
        
        Args:
            documents: 文档列表，每个文档是token列表
        """
        self._total_documents = len(documents)
        
        total_length = 0
        df_counter: Dict[str, int] = {}
        
        for doc in documents:
            doc_tokens_lower = [t.lower() for t in doc]
            total_length += len(doc_tokens_lower)
            
            # 每个文档中的唯一词
            unique_tokens = set(doc_tokens_lower)
            for token in unique_tokens:
                df_counter[token] = df_counter.get(token, 0) + 1
        
        self._document_frequencies = df_counter
        self._avg_doc_length = total_length / self._total_documents if self._total_documents > 0 else 0


class QueryFocusedAttention:
    """
    问题聚焦注意力计算器
    
    独立的注意力计算类，可以在不修改Tokenizer的情况下使用。
    适用于：
    - 后处理已有的token序列
    - 与第三方Tokenizer配合使用
    """
    
    def __init__(self, config: Optional[AttentionConfig] = None):
        self.config = config or AttentionConfig()
        self._mixin = TokenAttentionMixin(attention_config=self.config)
    
    def compute(
        self,
        doc_tokens: List[str],
        query: str,
        strategy: Optional[AttentionStrategy] = None,
    ) -> List[float]:
        """计算注意力权重"""
        return self._mixin.compute_attention_weights(doc_tokens, query, strategy)
    
    def apply_weights(
        self,
        embeddings: List[List[float]],
        weights: List[float],
    ) -> List[List[float]]:
        """
        将权重应用到embedding上
        
        Args:
            embeddings: token embeddings列表
            weights: 注意力权重列表
            
        Returns:
            加权后的embeddings
        """
        if len(embeddings) != len(weights):
            raise ValueError("Embeddings和weights长度必须相同")
        
        weighted = []
        for emb, w in zip(embeddings, weights):
            weighted.append([e * w for e in emb])
        
        return weighted
