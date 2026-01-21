"""
Test Symbol Channel - 符号通道测试

Phase 6.4
"""

import pytest
import torch

from app.io.symbol_anchor import (
    SymbolAnchor,
    SymbolAnchors,
    CriticalTokenDetector,
    SymbolAnchorEncoder,
    detect_critical_tokens,
    apply_anchors_to_tokens,
)


class TestSymbolAnchor:
    """测试符号锚点数据结构"""
    
    def test_basic_anchor(self):
        """基本锚点测试"""
        anchor = SymbolAnchor(
            position=5,
            token_id=1234,
            token_text="user_name",
            anchor_type="identifier",
        )
        
        assert anchor.position == 5
        assert anchor.token_id == 1234
        assert anchor.is_critical


class TestSymbolAnchors:
    """测试锚点集合"""
    
    def test_collection(self):
        """集合操作测试"""
        anchors = SymbolAnchors()
        anchors.anchors.append(SymbolAnchor(position=0, token_id=100))
        anchors.anchors.append(SymbolAnchor(position=5, token_id=200, slot_id=1))
        
        assert anchors.num_anchors == 2
        assert anchors.get_by_position(0).token_id == 100
        assert anchors.get_by_slot(1).token_id == 200
    
    def test_get_token_ids(self):
        """获取 token ID 列表"""
        anchors = SymbolAnchors()
        anchors.anchors = [
            SymbolAnchor(position=0, token_id=100),
            SymbolAnchor(position=1, token_id=200),
            SymbolAnchor(position=2, token_id=300),
        ]
        
        ids = anchors.get_token_ids()
        
        assert ids == [100, 200, 300]
    
    def test_to_dict(self):
        """转换为字典"""
        anchors = SymbolAnchors()
        anchors.anchors = [
            SymbolAnchor(position=0, token_id=100),
            SymbolAnchor(position=5, token_id=500),
        ]
        
        d = anchors.to_dict()
        
        assert d == {0: 100, 5: 500}


class TestCriticalTokenDetector:
    """测试关键 token 检测器"""
    
    def test_basic_detection(self):
        """基本检测测试"""
        detector = CriticalTokenDetector(d_model=256)
        hidden = torch.randn(2, 16, 256)
        
        scores, mask = detector(hidden)
        
        assert scores.shape == (2, 16)
        assert mask.shape == (2, 16)
    
    def test_with_attention_mask(self):
        """带注意力掩码测试"""
        detector = CriticalTokenDetector(d_model=128)
        hidden = torch.randn(2, 16, 128)
        attention_mask = torch.zeros(2, 16)
        attention_mask[:, :8] = 1
        
        scores, mask = detector(hidden, attention_mask=attention_mask)
        
        # Padding 位置的分数应被掩盖
        assert scores[:, 8:].sum() == 0


class TestSymbolAnchorEncoder:
    """测试符号锚点编码器"""
    
    def test_basic_encoding(self):
        """基本编码测试"""
        encoder = SymbolAnchorEncoder(d_model=256, d_output=128)
        hidden = torch.randn(2, 16, 256)
        token_ids = torch.randint(0, 1000, (2, 16))
        
        anchors, anchor_vector = encoder(hidden, token_ids)
        
        assert isinstance(anchors, SymbolAnchors)
        assert anchor_vector.shape == (2, 128)
    
    def test_with_texts(self):
        """带文本信息测试"""
        encoder = SymbolAnchorEncoder(d_model=128, d_output=64)
        hidden = torch.randn(1, 8, 128)
        token_ids = torch.randint(0, 1000, (1, 8))
        token_texts = [["def", "foo", "(", "x", ")", ":", "return", "x"]]
        
        anchors, anchor_vector = encoder(hidden, token_ids, token_texts)
        
        # 应该检测到一些关键 token
        assert anchors.num_anchors >= 0


class TestDetectCriticalTokens:
    """测试关键 token 检测函数"""
    
    def test_identifier_detection(self):
        """标识符检测"""
        text = "user_name = get_user(id)"
        critical = detect_critical_tokens(text)
        
        # 应该检测到 user_name, get_user, id
        texts = [c[1] for c in critical]
        assert "user_name" in texts
        assert "get_user" in texts
        assert "id" in texts
    
    def test_bracket_detection(self):
        """括号检测"""
        text = "foo(x, y)"
        critical = detect_critical_tokens(text)
        
        types = [c[2] for c in critical]
        assert "bracket" in types


class TestApplyAnchors:
    """测试锚点应用"""
    
    def test_basic_apply(self):
        """基本应用测试"""
        original = torch.tensor([1, 2, 3, 4, 5])
        generated = torch.tensor([10, 20, 30, 40, 50])
        
        anchors = SymbolAnchors()
        anchors.anchors = [
            SymbolAnchor(position=0, token_id=1),
            SymbolAnchor(position=2, token_id=3),
        ]
        
        result = apply_anchors_to_tokens(original, anchors, generated)
        
        assert result[0] == 1  # 被锚点替换
        assert result[1] == 20  # 保持生成值
        assert result[2] == 3  # 被锚点替换


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
