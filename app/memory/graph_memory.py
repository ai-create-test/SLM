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
    
    # ============================================================
    # AMHVQ+ 结构通道集成
    # ============================================================
    
    def store_structure(
        self,
        structure_id: str,
        nodes: List[Dict[str, Any]],
        edges: List[Tuple[str, str, str]],
        summary_vector: Optional[torch.Tensor] = None,
        structure_type: str = "ast",
    ) -> List[str]:
        """
        存储结构图 (用于 AMHVQ+ 结构通道)
        
        Args:
            structure_id: 结构的唯一标识符
            nodes: 节点列表 [{"id": str, "type": str, "content": str, "slot_id": int?}, ...]
            edges: 边列表 [(source, relation, target), ...]
            summary_vector: 结构摘要向量
            structure_type: 结构类型 ("ast", "syntax", "custom")
            
        Returns:
            创建的节点 ID 列表
        """
        created_node_ids = []
        
        # 添加根节点 (结构摘要)
        root_name = f"structure_{structure_id}"
        if summary_vector is None:
            summary_vector = torch.zeros(self.d_node)
        
        self.add_node(
            name=root_name,
            vector=summary_vector,
            node_type=f"structure_root_{structure_type}",
            metadata={
                "structure_id": structure_id,
                "structure_type": structure_type,
                "num_nodes": len(nodes),
            }
        )
        created_node_ids.append(root_name)
        
        # 添加结构节点
        for node_info in nodes:
            node_name = f"{structure_id}_{node_info['id']}"
            
            # 如果没有向量，使用零向量
            node_vector = node_info.get("vector", torch.zeros(self.d_node))
            
            self.add_node(
                name=node_name,
                vector=node_vector,
                node_type=f"structure_{node_info.get('type', 'node')}",
                metadata={
                    "structure_id": structure_id,
                    "original_id": node_info["id"],
                    "content": node_info.get("content", ""),
                    "slot_id": node_info.get("slot_id"),
                    "span": node_info.get("span", (0, 0)),
                }
            )
            created_node_ids.append(node_name)
            
            # 连接到根节点
            self.add_edge(root_name, "contains", node_name)
        
        # 添加结构边
        for source, relation, target in edges:
            source_name = f"{structure_id}_{source}"
            target_name = f"{structure_id}_{target}"
            if source_name in self._nodes and target_name in self._nodes:
                self.add_edge(source_name, relation, target_name)
        
        return created_node_ids
    
    def retrieve_skeleton(
        self,
        query_vector: torch.Tensor,
        top_k: int = 1,
        structure_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        检索结构骨架 (用于 AMHVQ+ 解码时的结构引导)
        
        Args:
            query_vector: 查询向量 (通常是语义向量)
            top_k: 返回最相似的结构数量
            structure_type: 可选的结构类型过滤
            
        Returns:
            {
                "structure_id": str,
                "nodes": List[Dict],  # 节点信息
                "slots": List[Dict],  # 槽位信息
                "skeleton_str": str,   # 骨架字符串表示
            }
        """
        # 找到所有结构根节点
        root_nodes = []
        for name, node in self._nodes.items():
            if node.node_type.startswith("structure_root"):
                if structure_type is None or structure_type in node.node_type:
                    root_nodes.append(node)
        
        if not root_nodes:
            return {
                "structure_id": None,
                "nodes": [],
                "slots": [],
                "skeleton_str": "",
            }
        
        # 计算相似度
        similarities = []
        for node in root_nodes:
            if query_vector.dim() == 1:
                sim = torch.cosine_similarity(
                    query_vector.unsqueeze(0),
                    node.vector.unsqueeze(0)
                ).item()
            else:
                sim = torch.cosine_similarity(
                    query_vector.mean(dim=0, keepdim=True),
                    node.vector.unsqueeze(0)
                ).item()
            similarities.append((node, sim))
        
        # 排序并选择 top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        best_root = similarities[0][0] if similarities else None
        
        if best_root is None:
            return {
                "structure_id": None,
                "nodes": [],
                "slots": [],
                "skeleton_str": "",
            }
        
        structure_id = best_root.metadata.get("structure_id", "unknown")
        
        # 收集该结构的所有节点
        structure_nodes = []
        slots = []
        
        for name, node in self._nodes.items():
            if node.metadata.get("structure_id") == structure_id and name != best_root.name:
                node_info = {
                    "id": node.metadata.get("original_id", name),
                    "type": node.node_type.replace("structure_", ""),
                    "content": node.metadata.get("content", ""),
                    "slot_id": node.metadata.get("slot_id"),
                }
                structure_nodes.append(node_info)
                
                if node_info["slot_id"] is not None:
                    slots.append({
                        "slot_id": node_info["slot_id"],
                        "slot_type": node_info["type"],
                        "content": node_info["content"],
                    })
        
        # 生成骨架字符串
        skeleton_str = self._generate_skeleton_str(structure_nodes)
        
        return {
            "structure_id": structure_id,
            "nodes": structure_nodes,
            "slots": sorted(slots, key=lambda x: x["slot_id"]),
            "skeleton_str": skeleton_str,
        }
    
    def _generate_skeleton_str(self, nodes: List[Dict]) -> str:
        """生成骨架字符串表示"""
        parts = []
        for node in nodes:
            if node["slot_id"] is not None:
                parts.append(f"□{node['slot_id']}")
            elif node["content"]:
                parts.append(node["content"][:10])
            else:
                parts.append(f"[{node['type']}]")
        return " ".join(parts[:10])  # 限制长度
