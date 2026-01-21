"""
Unified Training Stages - AMHVQ+ 统一训练阶段

Phase 11: 支持三通道的训练流程

Stages:
    - HierarchicalVQVAEStage: 层次化 VQ-VAE 训练
    - StructureChannelLoss: 结构通道损失
    - SymbolChannelLoss: 符号通道损失
    - UnifiedTrainingStage: 三通道联合训练
"""

from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..interfaces.base_module import HierarchicalLatent
from ..interfaces.unified_latent import UnifiedLatent


@dataclass
class TrainingResult:
    """训练结果"""
    stage: str
    epochs: int
    total_steps: int
    final_loss: float
    best_loss: float
    history: List[Dict[str, float]] = field(default_factory=list)
    duration_seconds: float = 0.0


# ============================================================================
# 损失函数
# ============================================================================

class HierarchicalReconstructionLoss(nn.Module):
    """
    层次化重建损失
    
    计算 global + chunks 的重建损失。
    """
    
    def __init__(
        self,
        global_weight: float = 1.0,
        chunks_weight: float = 1.0,
        detail_weight: float = 0.5,
    ):
        super().__init__()
        self.global_weight = global_weight
        self.chunks_weight = chunks_weight
        self.detail_weight = detail_weight
    
    def forward(
        self,
        pred: HierarchicalLatent,
        target: HierarchicalLatent,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算损失
        
        Returns:
            total_loss, loss_dict
        """
        losses = {}
        
        # Global 损失
        global_loss = F.mse_loss(pred.global_, target.global_)
        losses["global"] = global_loss.item()
        
        # Chunks 损失
        chunks_loss = F.mse_loss(pred.chunks, target.chunks)
        losses["chunks"] = chunks_loss.item()
        
        # Total
        total = (
            self.global_weight * global_loss +
            self.chunks_weight * chunks_loss
        )
        
        # Detail 损失 (如果有)
        if pred.detail is not None and target.detail is not None:
            detail_loss = F.mse_loss(pred.detail, target.detail)
            losses["detail"] = detail_loss.item()
            total = total + self.detail_weight * detail_loss
        
        losses["total"] = total.item()
        
        return total, losses


class StructureChannelLoss(nn.Module):
    """
    结构通道损失
    
    确保结构信息正确保留。
    """
    
    def __init__(
        self,
        summary_weight: float = 1.0,
        slot_weight: float = 0.5,
    ):
        super().__init__()
        self.summary_weight = summary_weight
        self.slot_weight = slot_weight
    
    def forward(
        self,
        pred_unified: UnifiedLatent,
        target_unified: UnifiedLatent,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算结构通道损失"""
        losses = {}
        total = torch.tensor(0.0)
        
        if pred_unified.structure is None or target_unified.structure is None:
            losses["structure"] = 0.0
            return total, losses
        
        # 结构摘要向量损失
        if (pred_unified.structure.summary_vector is not None and
            target_unified.structure.summary_vector is not None):
            summary_loss = F.mse_loss(
                pred_unified.structure.summary_vector,
                target_unified.structure.summary_vector,
            )
            losses["structure_summary"] = summary_loss.item()
            total = total + self.summary_weight * summary_loss
        
        # 槽位数量匹配
        pred_slots = len(pred_unified.structure.slots) if pred_unified.structure.slots else 0
        target_slots = len(target_unified.structure.slots) if target_unified.structure.slots else 0
        slot_diff = abs(pred_slots - target_slots)
        losses["slot_count_diff"] = slot_diff
        
        losses["total"] = total.item()
        
        return total, losses


class SymbolChannelLoss(nn.Module):
    """
    符号通道损失
    
    确保关键符号精确保留。
    """
    
    def __init__(
        self,
        anchor_match_weight: float = 1.0,
        position_weight: float = 0.1,
    ):
        super().__init__()
        self.anchor_match_weight = anchor_match_weight
        self.position_weight = position_weight
    
    def forward(
        self,
        pred_unified: UnifiedLatent,
        target_unified: UnifiedLatent,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算符号通道损失"""
        losses = {}
        total = torch.tensor(0.0)
        
        if pred_unified.symbols is None or target_unified.symbols is None:
            losses["symbols"] = 0.0
            return total, losses
        
        # 锚点匹配率
        pred_anchors = pred_unified.symbols.anchors if pred_unified.symbols else []
        target_anchors = target_unified.symbols.anchors if target_unified.symbols else []
        
        if target_anchors:
            # 计算匹配的锚点
            target_texts = {a.token_text for a in target_anchors if a.token_text}
            pred_texts = {a.token_text for a in pred_anchors if a.token_text}
            
            matches = len(target_texts & pred_texts)
            match_rate = matches / len(target_texts) if target_texts else 1.0
            
            # 损失 = 1 - 匹配率
            anchor_loss = 1.0 - match_rate
            losses["anchor_match_rate"] = match_rate
            losses["anchor_loss"] = anchor_loss
            total = total + self.anchor_match_weight * anchor_loss
        
        losses["total"] = total.item()
        
        return total, losses


# ============================================================================
# Phase 11.1: HierarchicalVQVAEStage
# ============================================================================

class HierarchicalVQVAEStage:
    """
    Stage: 层次化 VQ-VAE 训练
    
    训练 HierarchicalParagraphEncoder + Decoder
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        config: Dict[str, Any],
        device: str = "cpu",
    ):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.config = config
        self.device = device
        
        # 损失函数
        self.recon_loss = HierarchicalReconstructionLoss()
        
        # 优化器
        lr = config.get("lr", 1e-4)
        params = list(encoder.parameters()) + list(decoder.parameters())
        self.optimizer = AdamW(params, lr=lr)
        
        # 学习率调度
        self.scheduler = None
    
    def train(
        self,
        train_loader,
        eval_loader=None,
        num_epochs: int = 10,
        log_interval: int = 10,
        callback: Optional[Callable] = None,
    ) -> TrainingResult:
        """训练"""
        start_time = time.time()
        history = []
        best_loss = float("inf")
        total_steps = 0
        
        for epoch in range(num_epochs):
            self.encoder.train()
            self.decoder.train()
            epoch_loss = 0.0
            epoch_steps = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # 前向传播
                text = batch.get("text", batch.get("input", None))
                if text is None:
                    continue
                
                # 编码
                encoder_output = self.encoder(text)
                hierarchical = encoder_output.hierarchical_latent
                
                # 解码
                decoder_output = self.decoder(hierarchical)
                
                # 计算损失
                # 重建损失 + RVQ commitment loss
                recon_loss = decoder_output.loss if decoder_output.loss is not None else torch.tensor(0.0)
                vq_loss = self.encoder.get_loss(encoder_output)
                
                total_loss = recon_loss + self.config.get("vq_weight", 0.25) * vq_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_steps += 1
                total_steps += 1
                
                if batch_idx % log_interval == 0 and callback:
                    callback({
                        "epoch": epoch,
                        "step": batch_idx,
                        "loss": total_loss.item(),
                    })
            
            avg_loss = epoch_loss / max(epoch_steps, 1)
            history.append({"epoch": epoch, "loss": avg_loss})
            
            if avg_loss < best_loss:
                best_loss = avg_loss
        
        return TrainingResult(
            stage="hierarchical_vqvae",
            epochs=num_epochs,
            total_steps=total_steps,
            final_loss=avg_loss,
            best_loss=best_loss,
            history=history,
            duration_seconds=time.time() - start_time,
        )


# ============================================================================
# Phase 11.4: UnifiedTrainingStage
# ============================================================================

class UnifiedTrainingStage:
    """
    三通道联合训练
    
    同时训练语义、结构、符号通道。
    """
    
    def __init__(
        self,
        unified_encoder: nn.Module,
        unified_decoder: nn.Module,
        config: Dict[str, Any],
        device: str = "cpu",
    ):
        self.encoder = unified_encoder.to(device)
        self.decoder = unified_decoder.to(device)
        self.config = config
        self.device = device
        
        # 损失函数
        self.recon_loss = HierarchicalReconstructionLoss()
        self.structure_loss = StructureChannelLoss()
        self.symbol_loss = SymbolChannelLoss()
        
        # 权重
        self.semantic_weight = config.get("semantic_weight", 1.0)
        self.structure_weight = config.get("structure_weight", 0.5)
        self.symbol_weight = config.get("symbol_weight", 0.5)
        
        # 优化器
        lr = config.get("lr", 1e-4)
        params = list(unified_encoder.parameters()) + list(unified_decoder.parameters())
        self.optimizer = AdamW(params, lr=lr)
        
        # 课程学习
        self.curriculum = CurriculumScheduler(config)
    
    def train(
        self,
        train_loader,
        eval_loader=None,
        num_epochs: int = 10,
        log_interval: int = 10,
        callback: Optional[Callable] = None,
    ) -> TrainingResult:
        """三通道联合训练"""
        start_time = time.time()
        history = []
        best_loss = float("inf")
        total_steps = 0
        
        for epoch in range(num_epochs):
            self.encoder.train()
            self.decoder.train()
            epoch_losses = {"semantic": 0.0, "structure": 0.0, "symbol": 0.0, "total": 0.0}
            epoch_steps = 0
            
            # 课程学习权重
            curriculum_weights = self.curriculum.get_weights(epoch, num_epochs)
            
            for batch_idx, batch in enumerate(train_loader):
                text = batch.get("text", batch.get("input", None))
                scene = batch.get("scene", None)
                
                if text is None:
                    continue
                
                # 编码
                encoder_output = self.encoder(text, scene=scene)
                unified_latent = encoder_output.unified_latent
                
                # 解码
                decoder_output = self.decoder(unified_latent)
                
                # 计算各通道损失
                # 1. 语义损失
                semantic_loss = decoder_output.loss if decoder_output.loss is not None else torch.tensor(0.0, device=self.device)
                vq_loss = self.encoder.get_loss(encoder_output)
                semantic_total = semantic_loss + 0.25 * vq_loss
                
                # 2. 结构损失 (目前仅统计，不反向传播)
                # 结构通道用于精确重建，不通过梯度优化
                structure_loss = torch.tensor(0.0, device=self.device)
                
                # 3. 符号损失 (目前仅统计)
                symbol_loss = torch.tensor(0.0, device=self.device)
                
                # 总损失
                total_loss = (
                    curriculum_weights["semantic"] * semantic_total +
                    curriculum_weights["structure"] * structure_loss +
                    curriculum_weights["symbol"] * symbol_loss
                )
                
                # 反向传播
                self.optimizer.zero_grad()
                if total_loss.requires_grad:
                    total_loss.backward()
                self.optimizer.step()
                
                epoch_losses["semantic"] += semantic_total.item()
                epoch_losses["structure"] += structure_loss.item()
                epoch_losses["symbol"] += symbol_loss.item()
                epoch_losses["total"] += total_loss.item()
                epoch_steps += 1
                total_steps += 1
                
                if batch_idx % log_interval == 0 and callback:
                    callback({
                        "epoch": epoch,
                        "step": batch_idx,
                        "losses": {k: v / max(batch_idx + 1, 1) for k, v in epoch_losses.items()},
                    })
            
            # 平均损失
            for k in epoch_losses:
                epoch_losses[k] /= max(epoch_steps, 1)
            
            history.append({"epoch": epoch, **epoch_losses})
            
            if epoch_losses["total"] < best_loss:
                best_loss = epoch_losses["total"]
        
        return TrainingResult(
            stage="unified_training",
            epochs=num_epochs,
            total_steps=total_steps,
            final_loss=epoch_losses["total"],
            best_loss=best_loss,
            history=history,
            duration_seconds=time.time() - start_time,
        )


# ============================================================================
# Phase 11.5: 课程学习
# ============================================================================

class CurriculumScheduler:
    """
    课程学习调度器
    
    逐步增加任务复杂度和通道权重。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 课程阶段
        self.warmup_epochs = config.get("curriculum_warmup", 2)
        self.structure_start = config.get("structure_start_epoch", 3)
        self.symbol_start = config.get("symbol_start_epoch", 5)
        
        # 最终权重
        self.final_semantic = config.get("semantic_weight", 1.0)
        self.final_structure = config.get("structure_weight", 0.5)
        self.final_symbol = config.get("symbol_weight", 0.5)
    
    def get_weights(self, current_epoch: int, total_epochs: int) -> Dict[str, float]:
        """获取当前 epoch 的通道权重"""
        # 语义通道始终激活
        semantic_weight = self.final_semantic
        
        # 结构通道逐步激活
        if current_epoch < self.structure_start:
            structure_weight = 0.0
        else:
            progress = (current_epoch - self.structure_start) / max(total_epochs - self.structure_start, 1)
            structure_weight = self.final_structure * min(progress * 2, 1.0)
        
        # 符号通道最后激活
        if current_epoch < self.symbol_start:
            symbol_weight = 0.0
        else:
            progress = (current_epoch - self.symbol_start) / max(total_epochs - self.symbol_start, 1)
            symbol_weight = self.final_symbol * min(progress * 2, 1.0)
        
        return {
            "semantic": semantic_weight,
            "structure": structure_weight,
            "symbol": symbol_weight,
        }
    
    def get_difficulty(self, current_epoch: int, total_epochs: int) -> str:
        """获取当前难度级别"""
        progress = current_epoch / max(total_epochs, 1)
        
        if progress < 0.25:
            return "easy"  # 短文本，简单结构
        elif progress < 0.5:
            return "medium"  # 中等长度
        elif progress < 0.75:
            return "hard"  # 复杂结构
        else:
            return "expert"  # 完整难度
