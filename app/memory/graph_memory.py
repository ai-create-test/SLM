"""
Graph Memory - 类图结构知识存储

核心功能：以图结构存储知识，支持关系查询

节点: 概念/实体/段落
边: 关系 (因果、相似、对比等)
"""

from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
import torch


@dataclass
class GraphNode:
    """图节点"""
    name: str                           # 节点名称/ID
    vector: torch.Tensor                # 向量表示
    node_type: str = "entity"           # 节点类型
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return self.name == other.name


@dataclass
class GraphEdge:
    """图边"""
    source: str                         # 源节点名称
    target: str                         # 目标节点名称
    relation: str                       # 关系类型
    weight: float = 1.0                 # 边权重
    metadata: Dict[str, Any] = field(default_factory=dict)


class GraphMemory:
    """
    类图结构的知识存储
    
    支持:
    1. 节点增删改查
    2. 关系管理
    3. 子图检索
    4. 向量相似度搜索
    
    使用示例:
        graph = GraphMemory(d_node=512)
        
        # 添加节点
        graph.add_node("python", vector, node_type="concept")
        graph.add_node("programming", vector, node_type="concept")
        
        # 添加边
        graph.add_edge("python", "is_a", "programming")
        
        # 查询
        neighbors = graph.get_neighbors("python", relation="is_a")
        subgraph = graph.subgraph_retrieval(query_vector, hops=2)
    """
    
    # 预定义的关系类型
    RELATION_TYPES = [
        "is_a",         # 上下位关系
        "part_of",      # 部分-整体
        "causes",       # 因果关系
        "similar_to",   # 相似关系
        "contrast_to",  # 对比关系
        "follows",      # 顺序关系
        "requires",     # 依赖关系
        "related_to",   # 一般关联
    ]
    
    def __init__(self, d_node: int = 512):
        """
        Args:
            d_node: 节点向量维度
        """
        self.d_node = d_node
        
        # 存储
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: List[GraphEdge] = []
        
        # 邻接表 (用于快速查询)
        self._outgoing: Dict[str, List[GraphEdge]] = {}
        self._incoming: Dict[str, List[GraphEdge]] = {}
    
    def add_node(
        self,
        name: str,
        vector: torch.Tensor,
        node_type: str = "entity",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        添加节点
        
        Args:
            name: 节点名称 (唯一标识)
            vector: 向量表示
            node_type: 节点类型
            metadata: 额外元数据
        """
        if name in self._nodes:
            # 更新现有节点
            self._nodes[name].vector = vector
            self._nodes[name].metadata.update(metadata or {})
        else:
            # 创建新节点
            self._nodes[name] = GraphNode(
                name=name,
                vector=vector,
                node_type=node_type,
                metadata=metadata or {},
            )
            self._outgoing[name] = []
            self._incoming[name] = []
    
    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        添加边
        
        Args:
            source: 源节点名称
            relation: 关系类型
            target: 目标节点名称
            weight: 边权重
            metadata: 额外元数据
        """
        if source not in self._nodes or target not in self._nodes:
            raise ValueError(f"Nodes must exist: {source}, {target}")
        
        edge = GraphEdge(
            source=source,
            target=target,
            relation=relation,
            weight=weight,
            metadata=metadata or {},
        )
        
        self._edges.append(edge)
        self._outgoing[source].append(edge)
        self._incoming[target].append(edge)
    
    def get_node(self, name: str) -> Optional[GraphNode]:
        """获取节点"""
        return self._nodes.get(name)
    
    def get_neighbors(
        self,
        node: str,
        relation: Optional[str] = None,
        direction: str = "out",
    ) -> List[str]:
        """
        获取邻居节点
        
        Args:
            node: 节点名称
            relation: 可选，限定关系类型
            direction: 方向 ('out', 'in', 'both')
            
        Returns:
            邻居节点名称列表
        """
        neighbors = []
        
        if direction in ("out", "both"):
            for edge in self._outgoing.get(node, []):
                if relation is None or edge.relation == relation:
                    neighbors.append(edge.target)
        
        if direction in ("in", "both"):
            for edge in self._incoming.get(node, []):
                if relation is None or edge.relation == relation:
                    neighbors.append(edge.source)
        
        return list(set(neighbors))
    
    def get_path(
        self,
        start: str,
        end: str,
        max_hops: int = 5,
    ) -> Optional[List[str]]:
        """
        查找两节点间的路径 (BFS)
        """
        if start not in self._nodes or end not in self._nodes:
            return None
        
        visited = {start}
        queue = [(start, [start])]
        
        while queue:
            current, path = queue.pop(0)
            
            if current == end:
                return path
            
            if len(path) >= max_hops:
                continue
            
            for neighbor in self.get_neighbors(current, direction="out"):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def subgraph_retrieval(
        self,
        query_vector: torch.Tensor,
        hops: int = 2,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        基于向量相似度检索相关子图
        
        Args:
            query_vector: 查询向量
            hops: 扩展跳数
            top_k: 初始检索的节点数
            
        Returns:
            子图信息 {nodes: [...], edges: [...]}
        """
        # 1. 找到最相似的 k 个节点
        similarities = []
        for name, node in self._nodes.items():
            sim = torch.cosine_similarity(
                query_vector.unsqueeze(0),
                node.vector.unsqueeze(0),
            ).item()
            similarities.append((sim, name))
        
        similarities.sort(reverse=True)
        seed_nodes = [name for _, name in similarities[:top_k]]
        
        # 2. 扩展子图
        subgraph_nodes = set(seed_nodes)
        for _ in range(hops):
            new_nodes = set()
            for node in subgraph_nodes:
                neighbors = self.get_neighbors(node, direction="both")
                new_nodes.update(neighbors)
            subgraph_nodes.update(new_nodes)
        
        # 3. 收集边
        subgraph_edges = []
        for edge in self._edges:
            if edge.source in subgraph_nodes and edge.target in subgraph_nodes:
                subgraph_edges.append(edge)
        
        return {
            "nodes": [self._nodes[name] for name in subgraph_nodes],
            "edges": subgraph_edges,
            "seed_nodes": seed_nodes,
        }
    
    def remove_node(self, name: str) -> None:
        """删除节点及其相关边"""
        if name not in self._nodes:
            return
        
        # 删除相关边
        self._edges = [e for e in self._edges if e.source != name and e.target != name]
        
        # 更新邻接表
        del self._outgoing[name]
        del self._incoming[name]
        
        for edges in self._outgoing.values():
            edges[:] = [e for e in edges if e.target != name]
        for edges in self._incoming.values():
            edges[:] = [e for e in edges if e.source != name]
        
        # 删除节点
        del self._nodes[name]
    
    def clear(self) -> None:
        """清空图"""
        self._nodes.clear()
        self._edges.clear()
        self._outgoing.clear()
        self._incoming.clear()
    
    @property
    def num_nodes(self) -> int:
        return len(self._nodes)
    
    @property
    def num_edges(self) -> int:
        return len(self._edges)
    
    def __repr__(self) -> str:
        return f"GraphMemory(nodes={self.num_nodes}, edges={self.num_edges})"
