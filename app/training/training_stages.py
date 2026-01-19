"""
Training Stages - 各阶段训练实现

Stage 1: VQ-VAE 预训练
Stage 2: Dynamics 预训练
Stage 3: Emotion 联合训练
Stage 4: 端到端微调
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from ..io.paragraph_encoder import ParagraphEncoder
from ..io.paragraph_decoder import ParagraphDecoder
from .data_loader import ParagraphDataset, SequenceDataset, ParagraphDataLoader


@dataclass
class StageResult:
    """训练阶段结果"""
    stage: str
    epochs: int
    total_steps: int
    final_loss: float
    best_loss: float
    history: List[Dict[str, float]]
    duration_seconds: float


class StageTrainer:
    """训练阶段基类"""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = "cpu",
    ):
        self.model = model
        self.config = config
        self.device = device
        self.global_step = 0
        self.history: List[Dict[str, float]] = []
    
    def train(
        self,
        train_loader,
        eval_loader=None,
        num_epochs: int = 10,
        log_interval: int = 10,
        eval_interval: int = 100,
        callback: Optional[Callable] = None,
    ) -> StageResult:
        """执行训练"""
        raise NotImplementedError


# ============================================================================
# Stage 1: VQ-VAE Training
# ============================================================================

class VQVAEStage(StageTrainer):
    """
    Stage 1: VQ-VAE 预训练
    
    训练 Encoder + Codebook + Decoder
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = "cpu",
    ):
        super().__init__(model, config, device)
        
        # 获取组件
        self.encoder = model.encoder
        self.decoder = model.decoder
        
        # 超参数
        self.beta_commit = config.get("beta_commit", 0.25)
        self.gamma_codebook = config.get("gamma_codebook", 1.0)
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        
        # 优化器
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
    
    def train(
        self,
        train_loader,
        eval_loader=None,
        num_epochs: int = 10,
        log_interval: int = 10,
        eval_interval: int = 100,
        callback: Optional[Callable] = None,
    ) -> StageResult:
        """训练 VQ-VAE"""
        start_time = time.time()
        best_loss = float('inf')
        
        self.encoder.train()
        self.decoder.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                loss_dict = self._train_step(batch)
                
                epoch_loss += loss_dict["total"]
                num_batches += 1
                self.global_step += 1
                
                # 日志
                if self.global_step % log_interval == 0:
                    self.history.append({
                        "step": self.global_step,
                        "epoch": epoch,
                        **loss_dict,
                    })
                
                # 回调
                if callback:
                    callback(self.global_step, loss_dict)
            
            avg_loss = epoch_loss / max(num_batches, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            print(f"[Stage 1 VQ-VAE] Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
        
        return StageResult(
            stage="vqvae",
            epochs=num_epochs,
            total_steps=self.global_step,
            final_loss=avg_loss,
            best_loss=best_loss,
            history=self.history,
            duration_seconds=time.time() - start_time,
        )
    
    def _train_step(self, batch) -> Dict[str, float]:
        """单步训练"""
        texts = batch.texts
        
        # 编码
        encoder_output = self.encoder(texts)
        
        # 计算损失
        total_loss = 0.0
        losses = {}
        
        # VQ 损失
        if encoder_output.vq_output is not None:
            vq_out = encoder_output.vq_output
            commit_loss = vq_out.commitment_loss
            codebook_loss = vq_out.codebook_loss
            
            losses["commit"] = commit_loss.item()
            losses["codebook"] = codebook_loss.item()
            losses["perplexity"] = vq_out.perplexity.item()
            
            total_loss = self.beta_commit * commit_loss + self.gamma_codebook * codebook_loss
        
        # Note: Reconstruction loss requires target token IDs, which is handled separately
        # in text generation tasks. VQ-VAE stage focuses on codebook learning.
        
        # 反向传播
        if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
            self.optimizer.zero_grad()
            total_loss.backward()
            
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()),
                    self.max_grad_norm,
                )
            
            self.optimizer.step()
        
        losses["total"] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        return losses


# ============================================================================
# Stage 2: Dynamics Training
# ============================================================================

class DynamicsStage(StageTrainer):
    """
    Stage 2: Dynamics 预训练
    
    训练 Mamba/GRU 预测下一个潜向量
    冻结 Encoder
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = "cpu",
    ):
        super().__init__(model, config, device)
        
        self.encoder = model.encoder
        self.dynamics = model.dynamics
        
        # 冻结 encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        
        # 超参数
        self.learning_rate = config.get("learning_rate", 5e-5)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.ponder_cost_weight = config.get("ponder_cost_weight", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.dynamics.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        self.mse_loss = nn.MSELoss()
    
    def train(
        self,
        train_loader,
        eval_loader=None,
        num_epochs: int = 10,
        log_interval: int = 10,
        eval_interval: int = 100,
        callback: Optional[Callable] = None,
    ) -> StageResult:
        """训练 Dynamics"""
        start_time = time.time()
        best_loss = float('inf')
        
        self.dynamics.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                loss_dict = self._train_step(batch)
                
                epoch_loss += loss_dict["total"]
                num_batches += 1
                self.global_step += 1
                
                if self.global_step % log_interval == 0:
                    self.history.append({
                        "step": self.global_step,
                        "epoch": epoch,
                        **loss_dict,
                    })
                
                if callback:
                    callback(self.global_step, loss_dict)
            
            avg_loss = epoch_loss / max(num_batches, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            print(f"[Stage 2 Dynamics] Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
        
        return StageResult(
            stage="dynamics",
            epochs=num_epochs,
            total_steps=self.global_step,
            final_loss=avg_loss,
            best_loss=best_loss,
            history=self.history,
            duration_seconds=time.time() - start_time,
        )
    
    def _train_step(self, batch) -> Dict[str, float]:
        """单步训练"""
        # Handle different batch types
        if hasattr(batch, 'input_sequences'):
            # SequenceBatch format
            input_sequences = batch.input_sequences
            target_paragraphs = batch.target_paragraphs
            
            # 编码序列
            with torch.no_grad():
                z_sequences = []
                for seq in input_sequences:
                    output = self.encoder(seq)
                    z_sequences.append(output.latent.vector)
                
                z_input = torch.stack(z_sequences, dim=0)
                
                target_output = self.encoder(target_paragraphs)
                z_target = target_output.latent.vector
        else:
            # ParagraphBatch format - use consecutive paragraphs
            texts = batch.texts
            if len(texts) < 2:
                return {"prediction": 0.0, "total": 0.0}
            
            with torch.no_grad():
                # Encode all paragraphs
                all_output = self.encoder(texts)
                z_all = all_output.latent.vector  # [batch, d_latent]
                
                # Input: first n-1 paragraphs, Target: last n-1 paragraphs
                z_input = z_all[:-1].unsqueeze(1)  # [n-1, 1, d_latent]
                z_target = z_all[1:]  # [n-1, d_latent]
        # Clone to avoid memory aliasing issues with optimizer
        z_input = z_input.clone()
        z_target = z_target.clone()
        
        # 预测
        output = self.dynamics(z_input)
        z_pred = output.predicted_latent
        
        # 损失
        prediction_loss = self.mse_loss(z_pred, z_target)
        total_loss = prediction_loss
        
        losses = {"prediction": prediction_loss.item()}
        
        # Ponder cost
        if hasattr(output, 'ponder_cost') and output.ponder_cost is not None:
            ponder_loss = output.ponder_cost
            total_loss = total_loss + self.ponder_cost_weight * ponder_loss
            losses["ponder"] = ponder_loss.item() if isinstance(ponder_loss, torch.Tensor) else ponder_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        losses["total"] = total_loss.item()
        return losses


# ============================================================================
# Stage 3: Emotion Joint Training
# ============================================================================

class EmotionStage(StageTrainer):
    """
    Stage 3: Emotion 联合训练
    
    训练 VADEncoder + ModulatedDynamicsModel
    使用带情感标注的数据
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = "cpu",
    ):
        super().__init__(model, config, device)
        
        self.encoder = model.encoder
        self.emotion_encoder = model.emotion_encoder
        self.modulated_dynamics = model.modulated_dynamics
        
        # 冻结 encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        
        # 超参数
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        
        # 可训练参数: VADEncoder + ModulatedDynamics
        params = list(self.emotion_encoder.vad_encoder.parameters())
        params += list(self.modulated_dynamics.parameters())
        
        self.optimizer = optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        self.mse_loss = nn.MSELoss()
    
    def train(
        self,
        train_loader,
        eval_loader=None,
        num_epochs: int = 10,
        log_interval: int = 10,
        eval_interval: int = 100,
        callback: Optional[Callable] = None,
    ) -> StageResult:
        """训练 Emotion"""
        start_time = time.time()
        best_loss = float('inf')
        
        self.emotion_encoder.train()
        self.modulated_dynamics.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                loss_dict = self._train_step(batch)
                
                epoch_loss += loss_dict["total"]
                num_batches += 1
                self.global_step += 1
                
                if self.global_step % log_interval == 0:
                    self.history.append({
                        "step": self.global_step,
                        "epoch": epoch,
                        **loss_dict,
                    })
                
                if callback:
                    callback(self.global_step, loss_dict)
            
            avg_loss = epoch_loss / max(num_batches, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            print(f"[Stage 3 Emotion] Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
        
        return StageResult(
            stage="emotion",
            epochs=num_epochs,
            total_steps=self.global_step,
            final_loss=avg_loss,
            best_loss=best_loss,
            history=self.history,
            duration_seconds=time.time() - start_time,
        )
    
    def _train_step(self, batch) -> Dict[str, float]:
        """单步训练"""
        input_sequences = batch.input_sequences
        target_paragraphs = batch.target_paragraphs
        
        # 情感标签 (如果有)
        emotions = getattr(batch, 'emotions', None)
        
        # 编码序列
        with torch.no_grad():
            z_sequences = []
            for seq in input_sequences:
                output = self.encoder(seq)
                z_sequences.append(output.latent.vector)
            
            z_input = torch.stack(z_sequences, dim=0)
            
            target_output = self.encoder(target_paragraphs)
            z_target = target_output.latent.vector
        
        # 情感编码
        if emotions is not None:
            emotion_vec = self.emotion_encoder(emotions)
        else:
            # 随机情感 (数据增强)
            batch_size = len(input_sequences)
            random_emotions = ["happy", "sad", "neutral", "angry"]
            import random
            emotions = [random.choice(random_emotions) for _ in range(batch_size)]
            emotion_vec = self.emotion_encoder(emotions)
        
        # 预测
        output = self.modulated_dynamics(z_input, emotion=emotion_vec)
        z_pred = output.predicted_latent
        
        # 损失
        prediction_loss = self.mse_loss(z_pred, z_target)
        total_loss = prediction_loss
        
        losses = {"prediction": prediction_loss.item()}
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        
        if self.max_grad_norm > 0:
            params = list(self.emotion_encoder.vad_encoder.parameters())
            params += list(self.modulated_dynamics.parameters())
            torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        
        self.optimizer.step()
        
        losses["total"] = total_loss.item()
        return losses


# ============================================================================
# Stage 4: End-to-end Finetuning
# ============================================================================

class FinetuneStage(StageTrainer):
    """
    Stage 4: 端到端微调
    
    使用低学习率联合训练所有组件
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = "cpu",
    ):
        super().__init__(model, config, device)
        
        self.learning_rate = config.get("learning_rate", 1e-5)  # 低学习率
        self.weight_decay = config.get("weight_decay", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        
        # 所有参数
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        self.mse_loss = nn.MSELoss()
    
    def train(
        self,
        train_loader,
        eval_loader=None,
        num_epochs: int = 5,
        log_interval: int = 10,
        eval_interval: int = 100,
        callback: Optional[Callable] = None,
    ) -> StageResult:
        """端到端微调"""
        start_time = time.time()
        best_loss = float('inf')
        
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                loss_dict = self._train_step(batch)
                
                epoch_loss += loss_dict["total"]
                num_batches += 1
                self.global_step += 1
                
                if self.global_step % log_interval == 0:
                    self.history.append({
                        "step": self.global_step,
                        "epoch": epoch,
                        **loss_dict,
                    })
                
                if callback:
                    callback(self.global_step, loss_dict)
            
            avg_loss = epoch_loss / max(num_batches, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            print(f"[Stage 4 Finetune] Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
        
        return StageResult(
            stage="finetune",
            epochs=num_epochs,
            total_steps=self.global_step,
            final_loss=avg_loss,
            best_loss=best_loss,
            history=self.history,
            duration_seconds=time.time() - start_time,
        )
    
    def _train_step(self, batch) -> Dict[str, float]:
        """单步训练"""
        # 根据 batch 类型处理
        if hasattr(batch, 'input_sequences'):
            return self._train_step_sequence(batch)
        else:
            return self._train_step_paragraph(batch)
    
    def _train_step_paragraph(self, batch) -> Dict[str, float]:
        """段落级训练"""
        texts = batch.texts
        
        encoder_output = self.model.encoder(texts)
        total_loss = torch.tensor(0.0, device=self.device)
        losses = {}
        
        if encoder_output.vq_output is not None:
            vq_out = encoder_output.vq_output
            total_loss = total_loss + 0.25 * vq_out.commitment_loss
            losses["commit"] = vq_out.commitment_loss.item()
        
        if total_loss.requires_grad:
            self.optimizer.zero_grad()
            total_loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        losses["total"] = total_loss.item()
        return losses
    
    def _train_step_sequence(self, batch) -> Dict[str, float]:
        """序列级训练"""
        input_sequences = batch.input_sequences
        target_paragraphs = batch.target_paragraphs
        
        z_sequences = []
        for seq in input_sequences:
            output = self.model.encoder(seq)
            z_sequences.append(output.latent.vector)
        
        z_input = torch.stack(z_sequences, dim=0)
        
        target_output = self.model.encoder(target_paragraphs)
        z_target = target_output.latent.vector
        
        output = self.model.dynamics(z_input)
        z_pred = output.predicted_latent
        
        prediction_loss = self.mse_loss(z_pred, z_target)
        total_loss = prediction_loss
        
        losses = {"prediction": prediction_loss.item()}
        
        self.optimizer.zero_grad()
        total_loss.backward()
        
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        losses["total"] = total_loss.item()
        return losses
