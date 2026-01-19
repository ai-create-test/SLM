"""
Tests for GraphRAG Memory System

Tests:
1. LatentMemoryBank with node_id
2. RAGRetriever - Vector + Graph retrieval
3. Graph expansion
"""

import pytest
import torch

from app.memory import (
    LatentMemoryBank,
    GraphMemory,
    RAGRetriever,
    RAGConfig,
)


class TestLatentMemoryBankNodeId:
    """Test node_id support in LatentMemoryBank"""
    
    @pytest.fixture
    def memory(self):
        return LatentMemoryBank(d_latent=64, max_size=100)
    
    def test_add_with_node_id(self, memory):
        """Test adding memory with node_id"""
        vec = torch.randn(64)
        idx = memory.add(vec, content="test", node_id="node_1")
        assert idx == 0
        
        # Verify node_id is stored
        assert memory._metadata[0]["node_id"] == "node_1"
    
    def test_add_without_node_id(self, memory):
        """Test backward compatibility without node_id"""
        vec = torch.randn(64)
        idx = memory.add(vec, content="test")
        assert idx == 0
        assert memory._metadata[0]["node_id"] is None
    
    def test_retrieve_returns_node_id(self, memory):
        """Test that retrieved items have node_id in metadata"""
        vec = torch.randn(64)
        memory.add(vec, content="test", node_id="node_1")
        
        results = memory.retrieve(vec, k=1)
        assert len(results) == 1
        assert results[0].metadata.get("node_id") == "node_1"


class TestRAGRetriever:
    """Test RAGRetriever GraphRAG integration"""
    
    @pytest.fixture
    def setup(self):
        memory = LatentMemoryBank(d_latent=64, max_size=100)
        graph = GraphMemory(d_node=64)
        return memory, graph
    
    def test_vector_only_retrieve(self, setup):
        """Test retrieval with graph disabled"""
        memory, graph = setup
        retriever = RAGRetriever(memory, graph)
        
        # Add some memories
        v1 = torch.randn(64)
        memory.add(v1, content="content1", node_id="n1")
        
        results = retriever.retrieve(v1, use_graph=False)
        assert len(results) >= 1
        assert results[0].content == "content1"
    
    def test_graph_expansion(self, setup):
        """Test that graph neighbors are included in results"""
        memory, graph = setup
        
        # Add nodes to graph
        v1, v2, v3 = torch.randn(64), torch.randn(64), torch.randn(64)
        graph.add_node("n1", v1, metadata={"content": "Node 1"})
        graph.add_node("n2", v2, metadata={"content": "Node 2 (neighbor)"})
        graph.add_node("n3", v3, metadata={"content": "Node 3 (unrelated)"})
        
        # Add edge: n1 -> n2
        graph.add_edge("n1", "related_to", "n2")
        
        # Add n1 to memory bank
        memory.add(v1, content="Node 1", node_id="n1")
        
        retriever = RAGRetriever(memory, graph)
        
        # Query with v1 - should return n1 (vector) and n2 (graph neighbor)
        results = retriever.retrieve(v1, k=5, use_graph=True)
        
        contents = [r.content for r in results]
        assert "Node 1" in contents
        # n2 should be included via graph expansion
        assert any("neighbor" in c.lower() or "2" in c for c in contents)
    
    def test_add_memory_with_node(self, setup):
        """Test convenience method for synchronized add"""
        memory, graph = setup
        retriever = RAGRetriever(memory, graph)
        
        vec = torch.randn(64)
        idx = retriever.add_memory_with_node(
            vector=vec,
            content="combined content",
            node_id="combined_1",
            node_type="paragraph",
        )
        
        # Check memory bank
        assert len(memory) == 1
        assert memory._metadata[0]["node_id"] == "combined_1"
        
        # Check graph
        assert graph.num_nodes == 1
        node = graph.get_node("combined_1")
        assert node is not None
        assert node.metadata.get("content") == "combined content"
    
    def test_deduplication(self, setup):
        """Test that results are deduplicated between vector and graph sources"""
        memory, graph = setup
        
        # Add a node with content "shared"
        v1 = torch.randn(64)
        memory.add(v1, content="shared_content", node_id="n1")
        
        # Also add same content as a graph neighbor
        graph.add_node("n1", v1, metadata={"content": "shared_content"})
        graph.add_node("n2", torch.randn(64), metadata={"content": "shared_content"})
        graph.add_edge("n1", "related_to", "n2")
        
        retriever = RAGRetriever(memory, graph)
        results = retriever.retrieve(v1, use_graph=True)
        
        # Should be deduplicated by content - only 1 "shared_content"
        contents = [r.content for r in results]
        assert contents.count("shared_content") == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
