"""
Test Structure Channel - 结构通道测试

Phase 5.6
"""

import pytest
import torch

from app.io.structure_parser import (
    CodeStructureParser,
    TextStructureParser,
    StructureGraph,
    StructureNode,
    StructureSummaryEncoder,
    parse_structure,
    get_structure_parser,
)
from app.memory.graph_memory import GraphMemory


class TestCodeStructureParser:
    """测试代码结构解析器"""
    
    def test_can_parse_detection(self):
        """代码检测测试"""
        parser = CodeStructureParser()
        
        assert parser.can_parse("def hello(): pass")
        assert parser.can_parse("class MyClass:")
        assert parser.can_parse("import os")
        assert not parser.can_parse("Hello, how are you?")
    
    def test_parse_simple_code(self):
        """简单代码解析测试"""
        parser = CodeStructureParser()
        code = "x = 1"
        
        graph = parser.parse(code)
        
        assert isinstance(graph, StructureGraph)
        assert len(graph.nodes) > 0
    
    def test_parse_function(self):
        """函数解析测试"""
        parser = CodeStructureParser()
        code = """
def greet(name):
    return "Hello, " + name
"""
        
        graph = parser.parse(code)
        
        assert graph.root_id is not None
        assert len(graph.nodes) > 3
    
    def test_slot_extraction(self):
        """槽位提取测试"""
        parser = CodeStructureParser()
        code = "user_name = get_user(id)"
        
        graph = parser.parse(code)
        slots = graph.get_slots()
        
        assert len(slots) > 0
        # 应该有变量名、函数名等槽位


class TestTextStructureParser:
    """测试文本结构解析器"""
    
    def test_can_parse(self):
        """总是可以解析测试"""
        parser = TextStructureParser()
        
        assert parser.can_parse("Any text works")
        assert parser.can_parse("def foo(): pass")  # 也可以解析代码
    
    def test_parse_sentences(self):
        """句子解析测试"""
        parser = TextStructureParser()
        text = "Hello world. How are you? I am fine."
        
        graph = parser.parse(text)
        
        assert len(graph.nodes) > 1


class TestStructureGraph:
    """测试结构图"""
    
    def test_add_node(self):
        """添加节点测试"""
        graph = StructureGraph()
        node = StructureNode(node_id="test", node_type="test_type")
        
        graph.add_node(node)
        
        assert "test" in graph.nodes
    
    def test_skeleton_string(self):
        """骨架字符串测试"""
        graph = StructureGraph()
        graph.root_id = "root"
        graph.add_node(StructureNode(node_id="root", node_type="block"))
        graph.add_node(StructureNode(
            node_id="var",
            node_type="identifier",
            slot_id=0,
            parent="root"
        ))
        graph.nodes["root"].children.append("var")
        
        skeleton = graph.to_skeleton_string()
        
        assert "□0" in skeleton or "identifier" in skeleton


class TestStructureSummaryEncoder:
    """测试结构摘要编码器"""
    
    def test_basic_encoding(self):
        """基本编码测试"""
        encoder = StructureSummaryEncoder(d_model=256)
        
        graph = StructureGraph()
        graph.add_node(StructureNode(node_id="n1", node_type="function"))
        graph.add_node(StructureNode(node_id="n2", node_type="variable"))
        
        summary = encoder(graph)
        
        assert summary.shape == (256,)
    
    def test_empty_graph(self):
        """空图测试"""
        encoder = StructureSummaryEncoder(d_model=128)
        graph = StructureGraph()
        
        summary = encoder(graph)
        
        assert summary.shape == (128,)
        assert torch.allclose(summary, torch.zeros(128))


class TestGraphMemoryIntegration:
    """测试 GraphMemory 结构集成"""
    
    def test_store_structure(self):
        """存储结构测试"""
        graph_memory = GraphMemory(d_node=256)
        
        nodes = [
            {"id": "n1", "type": "function", "content": "def foo"},
            {"id": "n2", "type": "variable", "content": "x", "slot_id": 0},
        ]
        edges = [("n1", "contains", "n2")]
        
        created = graph_memory.store_structure(
            structure_id="test_001",
            nodes=nodes,
            edges=edges,
            summary_vector=torch.randn(256),
        )
        
        assert len(created) == 3  # root + 2 nodes
        assert graph_memory.num_nodes == 3
    
    def test_retrieve_skeleton(self):
        """检索骨架测试"""
        graph_memory = GraphMemory(d_node=128)
        
        # 存储一个结构
        nodes = [
            {"id": "n1", "type": "assignment", "content": "x = 1"},
            {"id": "n2", "type": "name", "content": "x", "slot_id": 0},
        ]
        summary_vec = torch.randn(128)
        graph_memory.store_structure(
            structure_id="struct_001",
            nodes=nodes,
            edges=[],
            summary_vector=summary_vec,
        )
        
        # 使用相似向量检索
        query = summary_vec + 0.1 * torch.randn(128)  # 略微扰动
        result = graph_memory.retrieve_skeleton(query)
        
        assert result["structure_id"] == "struct_001"
        assert len(result["slots"]) >= 1


class TestParseStructure:
    """测试便捷解析函数"""
    
    def test_auto_detect_code(self):
        """自动检测代码"""
        graph = parse_structure("def hello(): pass", parser_type="auto")
        
        assert graph.structure_type in ["ast", "simple"]
    
    def test_auto_detect_text(self):
        """自动检测文本"""
        graph = parse_structure("Hello, how are you today?", parser_type="auto")
        
        assert graph.structure_type == "syntax"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
