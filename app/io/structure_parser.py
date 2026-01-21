"""
Structure Parser - 结构解析器

AMHVQ+ 结构通道：解析文本结构 (代码 AST / 句法树) 并生成骨架。

场景:
    - coding: 解析 Python/JS AST，提取函数、变量、控制流
    - formal: 解析句法结构，提取主谓宾
    - chat: 简化结构或跳过
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import torch
import torch.nn as nn
import re


@dataclass
class StructureNode:
    """结构节点"""
    node_id: str                       # 唯一 ID
    node_type: str                     # 类型 (function, variable, statement, etc.)
    content: str = ""                  # 原始内容
    slot_id: Optional[int] = None      # 槽位 ID (如果是可填充位置)
    children: List[str] = field(default_factory=list)  # 子节点 ID
    parent: Optional[str] = None       # 父节点 ID
    span: Tuple[int, int] = (0, 0)     # 在原文中的位置
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructureGraph:
    """结构图"""
    nodes: Dict[str, StructureNode] = field(default_factory=dict)
    edges: List[Tuple[str, str, str]] = field(default_factory=list)  # (source, relation, target)
    root_id: Optional[str] = None
    structure_type: str = "generic"
    
    def add_node(self, node: StructureNode) -> None:
        self.nodes[node.node_id] = node
    
    def add_edge(self, source: str, relation: str, target: str) -> None:
        self.edges.append((source, relation, target))
    
    def get_slots(self) -> List[Tuple[int, str, str]]:
        """获取所有槽位: [(slot_id, slot_type, parent_type), ...]"""
        slots = []
        for node in self.nodes.values():
            if node.slot_id is not None:
                parent_type = self.nodes[node.parent].node_type if node.parent else "root"
                slots.append((node.slot_id, node.node_type, parent_type))
        return sorted(slots, key=lambda x: x[0])
    
    def to_skeleton_string(self) -> str:
        """生成骨架字符串 (用于调试)"""
        if not self.root_id:
            return ""
        
        def _format(node_id: str, indent: int = 0) -> str:
            node = self.nodes.get(node_id)
            if not node:
                return ""
            
            prefix = "  " * indent
            if node.slot_id is not None:
                content = f"□{node.slot_id}"
            else:
                content = node.content[:20] if node.content else node.node_type
            
            result = f"{prefix}{node.node_type}: {content}\n"
            for child_id in node.children:
                result += _format(child_id, indent + 1)
            return result
        
        return _format(self.root_id)


class BaseStructureParser(ABC):
    """结构解析器基类"""
    
    @abstractmethod
    def parse(self, text: str) -> StructureGraph:
        """解析文本为结构图"""
        pass
    
    @abstractmethod
    def can_parse(self, text: str) -> bool:
        """检查是否能解析该文本"""
        pass


class CodeStructureParser(BaseStructureParser):
    """
    代码结构解析器
    
    解析 Python 代码的 AST 结构。
    """
    
    def __init__(self):
        self.slot_counter = 0
    
    def can_parse(self, text: str) -> bool:
        """检查是否像代码"""
        code_patterns = [
            r'def\s+\w+',      # Python function
            r'class\s+\w+',    # Python class
            r'import\s+\w+',   # Import
            r'=\s*\w+\(',      # Function call assignment
            r'if\s+.*:',       # If statement
            r'for\s+\w+\s+in', # For loop
        ]
        for pattern in code_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def parse(self, text: str) -> StructureGraph:
        """解析代码结构"""
        graph = StructureGraph(structure_type="ast")
        self.slot_counter = 0
        
        try:
            import ast
            tree = ast.parse(text)
            root_node = self._parse_ast_node(tree, graph)
            graph.root_id = root_node.node_id
        except:
            # 如果 AST 解析失败，使用简单的正则解析
            graph = self._simple_parse(text)
        
        return graph
    
    def _parse_ast_node(
        self,
        node,
        graph: StructureGraph,
        parent_id: Optional[str] = None,
    ) -> StructureNode:
        """递归解析 AST 节点"""
        import ast
        
        node_id = f"node_{len(graph.nodes)}"
        node_type = node.__class__.__name__
        
        # 识别需要作为槽位的节点
        slot_id = None
        content = ""
        
        if isinstance(node, ast.Name):
            slot_id = self._next_slot()
            content = node.id
        elif isinstance(node, ast.Constant):
            slot_id = self._next_slot()
            content = str(node.value)[:50]
        elif isinstance(node, ast.FunctionDef):
            content = f"def {node.name}"
        elif isinstance(node, ast.ClassDef):
            content = f"class {node.name}"
        elif isinstance(node, ast.Assign):
            content = "assignment"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                content = f"call {node.func.id}"
        
        struct_node = StructureNode(
            node_id=node_id,
            node_type=node_type,
            content=content,
            slot_id=slot_id,
            parent=parent_id,
        )
        graph.add_node(struct_node)
        
        # 处理子节点
        for child in ast.iter_child_nodes(node):
            child_node = self._parse_ast_node(child, graph, node_id)
            struct_node.children.append(child_node.node_id)
            graph.add_edge(node_id, "contains", child_node.node_id)
        
        return struct_node
    
    def _simple_parse(self, text: str) -> StructureGraph:
        """简单的正则解析 (AST 失败时的回退)"""
        graph = StructureGraph(structure_type="simple")
        
        root = StructureNode(node_id="root", node_type="code_block")
        graph.add_node(root)
        graph.root_id = "root"
        
        # 提取标识符
        identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', text)
        for i, ident in enumerate(set(identifiers)):
            if ident not in ['def', 'class', 'if', 'for', 'while', 'return', 'import']:
                node = StructureNode(
                    node_id=f"ident_{i}",
                    node_type="identifier",
                    content=ident,
                    slot_id=self._next_slot(),
                    parent="root",
                )
                graph.add_node(node)
                root.children.append(node.node_id)
        
        return graph
    
    def _next_slot(self) -> int:
        slot_id = self.slot_counter
        self.slot_counter += 1
        return slot_id


class TextStructureParser(BaseStructureParser):
    """
    文本结构解析器
    
    解析自然语言的句法结构。
    """
    
    def can_parse(self, text: str) -> bool:
        """总是可以解析文本"""
        return True
    
    def parse(self, text: str) -> StructureGraph:
        """解析文本结构 (简化版句法)"""
        graph = StructureGraph(structure_type="syntax")
        
        root = StructureNode(node_id="root", node_type="paragraph")
        graph.add_node(root)
        graph.root_id = "root"
        
        # 按句子切分
        sentences = re.split(r'[。！？.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        for i, sent in enumerate(sentences):
            sent_node = StructureNode(
                node_id=f"sent_{i}",
                node_type="sentence",
                content=sent[:50],
                parent="root",
            )
            graph.add_node(sent_node)
            root.children.append(sent_node.node_id)
            
            # 简单提取关键词作为槽位
            words = sent.split()
            for j, word in enumerate(words[:5]):  # 限制数量
                word_node = StructureNode(
                    node_id=f"word_{i}_{j}",
                    node_type="token",
                    content=word,
                    slot_id=i * 10 + j,
                    parent=sent_node.node_id,
                )
                graph.add_node(word_node)
                sent_node.children.append(word_node.node_id)
        
        return graph


class StructureSummaryEncoder(nn.Module):
    """
    结构摘要编码器
    
    将结构图编码为向量。
    """
    
    def __init__(
        self,
        d_model: int,
        max_nodes: int = 32,
        num_node_types: int = 16,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_nodes = max_nodes
        
        # 节点类型嵌入
        self.type_embedding = nn.Embedding(num_node_types, d_model)
        
        # 节点编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # 图级别池化
        self.graph_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )
        
        # 类型映射
        self.type_to_id = {}
        self.next_type_id = 0
    
    def forward(
        self,
        graph: StructureGraph,
        text_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        编码结构图
        
        Args:
            graph: 结构图
            text_embeddings: 可选的文本嵌入 [seq_len, d_model]
            
        Returns:
            summary: [d_model]
        """
        device = self.type_embedding.weight.device
        
        if not graph.nodes:
            return torch.zeros(self.d_model, device=device)
        
        # 编码每个节点
        node_embeds = []
        for node_id, node in list(graph.nodes.items())[:self.max_nodes]:
            type_id = self._get_type_id(node.node_type)
            type_embed = self.type_embedding(torch.tensor(type_id, device=device))
            
            # 如果有文本嵌入，结合位置信息
            if text_embeddings is not None and node.span[1] > 0:
                start, end = node.span
                end = min(end, text_embeddings.shape[0])
                if start < end:
                    content_embed = text_embeddings[start:end].mean(dim=0)
                else:
                    content_embed = torch.zeros(self.d_model, device=device)
            else:
                content_embed = torch.zeros(self.d_model, device=device)
            
            # 拼接并编码
            combined = torch.cat([type_embed, content_embed])
            node_embed = self.node_encoder(combined)
            node_embeds.append(node_embed)
        
        # 堆叠并池化
        if node_embeds:
            node_tensor = torch.stack(node_embeds)  # [num_nodes, d_model]
            summary = self.graph_pool(node_tensor.mean(dim=0))
        else:
            summary = torch.zeros(self.d_model, device=device)
        
        return summary
    
    def _get_type_id(self, node_type: str) -> int:
        if node_type not in self.type_to_id:
            self.type_to_id[node_type] = min(self.next_type_id, 15)
            self.next_type_id += 1
        return self.type_to_id[node_type]


# ============================================================
# 工厂函数
# ============================================================

def get_structure_parser(parser_type: str = "auto") -> BaseStructureParser:
    """获取结构解析器"""
    if parser_type == "code" or parser_type == "ast":
        return CodeStructureParser()
    elif parser_type == "text" or parser_type == "syntax":
        return TextStructureParser()
    else:
        # auto: 返回代码解析器 (会自动 fallback)
        return CodeStructureParser()


def parse_structure(
    text: str,
    parser_type: str = "auto",
) -> StructureGraph:
    """解析文本结构"""
    if parser_type == "auto":
        code_parser = CodeStructureParser()
        if code_parser.can_parse(text):
            return code_parser.parse(text)
        else:
            return TextStructureParser().parse(text)
    else:
        parser = get_structure_parser(parser_type)
        return parser.parse(text)
