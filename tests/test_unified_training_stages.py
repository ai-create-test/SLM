"""
Test Unified Training Stages - 统一训练阶段测试

Phase 11.6
"""

import pytest
import torch

from app.interfaces.base_module import HierarchicalLatent
from app.interfaces.unified_latent import UnifiedLatent, StructureRef, SymbolAnchors


class TestHierarchicalReconstructionLoss:
    """测试层次化重建损失"""
    
    def test_basic_loss(self):
        """基本损失计算"""
        from app.training.unified_training_stages import HierarchicalReconstructionLoss
        
        loss_fn = HierarchicalReconstructionLoss()
        
        pred = HierarchicalLatent(
            global_=torch.randn(2, 1, 256),
            chunks=torch.randn(2, 4, 256),
        )
        target = HierarchicalLatent(
            global_=torch.randn(2, 1, 256),
            chunks=torch.randn(2, 4, 256),
        )
        
        loss, loss_dict = loss_fn(pred, target)
        
        assert loss.item() >= 0
        assert "global" in loss_dict
        assert "chunks" in loss_dict
        assert "total" in loss_dict


class TestStructureChannelLoss:
    """测试结构通道损失"""
    
    def test_no_structure(self):
        """无结构时损失为零"""
        from app.training.unified_training_stages import StructureChannelLoss
        
        loss_fn = StructureChannelLoss()
        
        pred = UnifiedLatent(
            semantic=HierarchicalLatent(
                global_=torch.randn(1, 1, 256),
                chunks=torch.randn(1, 4, 256),
            ),
            structure=None,
        )
        target = UnifiedLatent(
            semantic=HierarchicalLatent(
                global_=torch.randn(1, 1, 256),
                chunks=torch.randn(1, 4, 256),
            ),
            structure=None,
        )
        
        loss, loss_dict = loss_fn(pred, target)
        
        assert loss.item() == 0.0


class TestSymbolChannelLoss:
    """测试符号通道损失"""
    
    def test_no_symbols(self):
        """无符号时损失为零"""
        from app.training.unified_training_stages import SymbolChannelLoss
        
        loss_fn = SymbolChannelLoss()
        
        pred = UnifiedLatent(
            semantic=HierarchicalLatent(
                global_=torch.randn(1, 1, 256),
                chunks=torch.randn(1, 4, 256),
            ),
            symbols=None,
        )
        target = UnifiedLatent(
            semantic=HierarchicalLatent(
                global_=torch.randn(1, 1, 256),
                chunks=torch.randn(1, 4, 256),
            ),
            symbols=None,
        )
        
        loss, loss_dict = loss_fn(pred, target)
        
        assert loss.item() == 0.0


class TestCurriculumScheduler:
    """测试课程学习调度器"""
    
    def test_weights_progression(self):
        """权重渐进测试"""
        from app.training.unified_training_stages import CurriculumScheduler
        
        scheduler = CurriculumScheduler({
            "curriculum_warmup": 2,
            "structure_start_epoch": 3,
            "symbol_start_epoch": 5,
        })
        
        # Epoch 0: 只有语义
        w0 = scheduler.get_weights(0, 10)
        assert w0["structure"] == 0.0
        assert w0["symbol"] == 0.0
        
        # Epoch 5: 语义 + 结构
        w5 = scheduler.get_weights(5, 10)
        assert w5["structure"] > 0.0
        assert w5["symbol"] >= 0.0
        
        # Epoch 9: 全部
        w9 = scheduler.get_weights(9, 10)
        assert w9["semantic"] > 0.0
    
    def test_difficulty_progression(self):
        """难度渐进测试"""
        from app.training.unified_training_stages import CurriculumScheduler
        
        scheduler = CurriculumScheduler({})
        
        assert scheduler.get_difficulty(0, 10) == "easy"
        assert scheduler.get_difficulty(5, 10) == "hard"
        assert scheduler.get_difficulty(9, 10) == "expert"


class TestTrainingResult:
    """测试训练结果"""
    
    def test_result_creation(self):
        """结果创建测试"""
        from app.training.unified_training_stages import TrainingResult
        
        result = TrainingResult(
            stage="test",
            epochs=10,
            total_steps=100,
            final_loss=0.5,
            best_loss=0.3,
        )
        
        assert result.stage == "test"
        assert result.epochs == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
