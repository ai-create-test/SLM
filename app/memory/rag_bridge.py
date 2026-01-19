"""
RAG Bridge - Graph-Augmented Retrieval

核心功能: 将向量检索与图结构查询结合，实现 GraphRAG

工作流程:
1. Vector Step: 从 LatentMemoryBank 检索 Top-K 相似向量
2. Graph Step: 从 GraphMemory 扩展 1-hop 邻居节点
3. Merge: 合并、去重、排序
4. Return: 返回增强的 MemoryItem 列表
"""

from typing import List, Optional, Dict, Set, Any
from dataclasses import dataclass
import torch

from ..interfaces.base_module import MemoryItem
from .latent_memory_bank import LatentMemoryBank
from .graph_memory import GraphMemory


@dataclass
class RAGConfig:
    """RAG 检索配置"""
    vector_k: int = 5               # 向量检索的 Top-K
    graph_hops: int = 1             # 图扩展的跳数
    max_total_results: int = 15     # 最大返回结果数
    graph_weight: float = 0.5       # 图结果的权重 (vs 向量结果)
    dedupe_by_content: bool = True  # 是否根据内容去重


class RAGRetriever:
    """
    Graph-Augmented Retrieval (GraphRAG) 检索器
    
    融合向量相似度检索和图结构扩展：
    1. 向量检索找到语义相似的片段
    2. 图扩展找到结构关联的片段
    3. 合并结果，返回增强的上下文
    
    使用示例:
        memory_bank = LatentMemoryBank(d_latent=512)
        graph = GraphMemory(d_node=512)
        
        retriever = RAGRetriever(memory_bank, graph)
        
        # 检索
        query = encoder(current_text)
        context = retriever.retrieve(query, k=10)
        
        for item in context:
            print(item.content, item.metadata.get("source"))
    """
    
    def __init__(
        self,
        memory_bank: LatentMemoryBank,
        graph_memory: GraphMemory,
        config: Optional[RAGConfig] = None,
    ):
        """
        Args:
            memory_bank: 向量记忆库
            graph_memory: 图记忆库
            config: 检索配置
        """
        self.memory_bank = memory_bank
        self.graph_memory = graph_memory
        self.config = config or RAGConfig()
    
    def retrieve(
        self,
        query: torch.Tensor,
        k: Optional[int] = None,
        use_graph: bool = True,
    ) -> List[MemoryItem]:
        """
        执行 GraphRAG 检索
        
        Args:
            query: 查询向量 [d_latent]
            k: 返回结果数量 (默认使用 config.max_total_results)
            use_graph: 是否启用图扩展
            
        Returns:
            增强的 MemoryItem 列表
        """
        k = k or self.config.max_total_results
        
        # Step 1: 向量检索
        vector_results = self._vector_retrieve(query)
        
        if not use_graph or len(self.graph_memory._nodes) == 0:
            # 不使用图扩展
            return vector_results[:k]
        
        # Step 2: 提取 node_ids
        node_ids = self._extract_node_ids(vector_results)
        
        # Step 3: 图扩展
        graph_results = self._graph_expand(node_ids)
        
        # Step 4: 合并和排序
        merged = self._merge_results(vector_results, graph_results, query)
        
        return merged[:k]
    
    def _vector_retrieve(self, query: torch.Tensor) -> List[MemoryItem]:
        """向量检索"""
        return self.memory_bank.retrieve(query, k=self.config.vector_k)
    
    def _extract_node_ids(self, items: List[MemoryItem]) -> List[str]:
        """从 MemoryItem 中提取 node_id"""
        node_ids = []
        for item in items:
            node_id = item.metadata.get("node_id")
            if node_id is not None:
                node_ids.append(node_id)
        return node_ids
    
    def _graph_expand(self, node_ids: List[str]) -> List[MemoryItem]:
        """
        图扩展：获取 node_ids 的邻居节点
        
        Args:
            node_ids: 种子节点 ID 列表
            
        Returns:
            邻居节点对应的 MemoryItem 列表
        """
        if not node_ids:
            return []
        
        # 收集邻居
        neighbor_ids: Set[str] = set()
        for node_id in node_ids:
            neighbors = self.graph_memory.get_neighbors(
                node_id,
                direction="both",
            )
            neighbor_ids.update(neighbors)
        
        # 排除种子节点本身
        neighbor_ids -= set(node_ids)
        
        # 转换为 MemoryItem
        results = []
        for nid in neighbor_ids:
            node = self.graph_memory.get_node(nid)
            if node is not None:
                results.append(MemoryItem(
                    vector=node.vector,
                    content=node.metadata.get("content", node.name),
                    timestamp=0.0,  # 图节点没有时间戳
                    importance=1.0,
                    metadata={
                        "node_id": nid,
                        "source": "graph",
                        "node_type": node.node_type,
                    },
                ))
        
        return results
    
    def _merge_results(
        self,
        vector_results: List[MemoryItem],
        graph_results: List[MemoryItem],
        query: torch.Tensor,
    ) -> List[MemoryItem]:
        """
        合并向量和图检索结果
        
        使用相关性评分排序：
        - 向量结果：使用余弦相似度
        - 图结果：使用加权评分
        """
        # 标记来源
        for item in vector_results:
            if "source" not in item.metadata:
                item.metadata["source"] = "vector"
        
        # 去重
        seen_contents: Set[str] = set()
        seen_node_ids: Set[str] = set()
        merged = []
        
        for item in vector_results + graph_results:
            # 内容去重
            if self.config.dedupe_by_content and item.content:
                if item.content in seen_contents:
                    continue
                seen_contents.add(item.content)
            
            # node_id 去重
            node_id = item.metadata.get("node_id")
            if node_id and node_id in seen_node_ids:
                continue
            if node_id:
                seen_node_ids.add(node_id)
            
            merged.append(item)
        
        # 计算相关性评分并排序
        def score(item: MemoryItem) -> float:
            # 余弦相似度
            if isinstance(query, torch.Tensor):
                q = query.detach().float()
                v = item.vector.detach().float()
                sim = torch.cosine_similarity(q.unsqueeze(0), v.unsqueeze(0)).item()
            else:
                sim = 0.5
            
            # 来源权重
            source_weight = 1.0 if item.metadata.get("source") == "vector" else self.config.graph_weight
            
            return sim * source_weight
        
        merged.sort(key=score, reverse=True)
        
        return merged
    
    def add_memory_with_node(
        self,
        vector: torch.Tensor,
        content: str,
        node_id: str,
        node_type: str = "paragraph",
        importance: float = 1.0,
        related_nodes: Optional[List[tuple]] = None,
    ) -> int:
        """
        同时添加记忆到向量库和图库
        
        Args:
            vector: 向量表示
            content: 文本内容
            node_id: 节点 ID
            node_type: 节点类型
            importance: 重要度
            related_nodes: 相关节点列表 [(target_id, relation), ...]
            
        Returns:
            向量库索引
        """
        # 添加到向量库
        idx = self.memory_bank.add(
            vector=vector,
            content=content,
            importance=importance,
            node_id=node_id,
        )
        
        # 添加到图库
        self.graph_memory.add_node(
            name=node_id,
            vector=vector,
            node_type=node_type,
            metadata={"content": content, "vector_idx": idx},
        )
        
        # 添加边
        if related_nodes:
            for target_id, relation in related_nodes:
                try:
                    self.graph_memory.add_edge(node_id, relation, target_id)
                except ValueError:
                    # 目标节点不存在，跳过
                    pass
        
        return idx
