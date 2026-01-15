"""
Dynamics Trainer - 动力学预测模型训练器

核心功能：
1. 训练 Mamba 模型预测下一个段落的潜向量
2. 使用已训练的 Encoder 将段落编码为潜向量序列
3. 损失: MSE(predicted, target) + λ * ponder_cost (ACT)

训练流程:
    [p1, p2, p3, p4] → Encoder → [z1, z2, z3, z4]
                                    ↓
                              Dynamics Model
                                    ↓
                              z4_pred (预测)
                                    ↓
                           MSE(z4_pred, z4)  (损失)
"""

from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from ..io.paragraph_encoder import ParagraphEncoder
from ..brain.dynamics_model import DynamicsModel
from ..brain.modulated_dynamics import ModulatedDynamicsModel
from .data_loader import ParagraphDataset, SequenceDataset, ParagraphDataLoader, SequenceBatch


@dataclass
class DynamicsTrainingMetrics:
    """动力学训练指标"""
    epoch: int = 0
    step: int = 0
    total_loss: float = 0.0
    prediction_loss: float = 0.0
    ponder_loss: float = 0.0
    avg_reasoning_steps: float = 0.0
    learning_rate: float = 0.0
    samples_per_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "total_loss": self.total_loss,
            "prediction_loss": self.prediction_loss,
            "ponder_loss": self.ponder_loss,
            "avg_reasoning_steps": self.avg_reasoning_steps,
            "learning_rate": self.learning_rate,
            "samples_per_sec": self.samples_per_sec,
        }


class DynamicsTrainer:
    """
    动力学预测模型训练器
    
    训练 Mamba 模型预测下一个段落的潜向量。
    
    使用示例:
        trainer = DynamicsTrainer(
            encoder=encoder,
            dynamics=dynamics_model,
            learning_rate=1e-4,
        )
        
        dataset = ParagraphDataset.synthetic(1000)
        seq_dataset = SequenceDataset(dataset, seq_len=5)
        trainer.train(seq_dataset, num_epochs=10)
    """
    
    def __init__(
        self,
        encoder: ParagraphEncoder,
        dynamics: DynamicsModel,
        modulated_dynamics: Optional[ModulatedDynamicsModel] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        ponder_cost_weight: float = 0.01,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
        log_interval: int = 10,
        freeze_encoder: bool = True,
    ):
        """
        Args:
            encoder: 预训练的段落编码器 (通常冻结)
            dynamics: 动力学模型 (被训练)
            modulated_dynamics: 可选的带情感调制的动力学模型
            learning_rate: 学习率
            weight_decay: 权重衰减
            ponder_cost_weight: ACT 思考代价权重
            warmup_steps: 预热步数
            max_grad_norm: 梯度裁剪
            device: 设备
            log_interval: 日志间隔
            freeze_encoder: 是否冻结编码器
        """
        self.encoder = encoder.to(device)
        self.dynamics = dynamics.to(device)
        self.modulated_dynamics = modulated_dynamics.to(device) if modulated_dynamics else None
        
        self.ponder_cost_weight = ponder_cost_weight
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.log_interval = log_interval
        self.freeze_encoder = freeze_encoder
        
        # 冻结编码器
        if freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # 收集可训练参数
        params = list(dynamics.parameters())
        if modulated_dynamics:
            params += list(modulated_dynamics.parameters())
        
        # 优化器
        self.optimizer = optim.AdamW(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.history: List[DynamicsTrainingMetrics] = []
    
    def train(
        self,
        dataset: SequenceDataset,
        num_epochs: int = 10,
        batch_size: int = 16,
        shuffle: bool = True,
        use_modulated: bool = False,
        emotion_encoder=None,
    ) -> List[DynamicsTrainingMetrics]:
        """
        训练动力学模型
        
        Args:
            dataset: 序列数据集
            num_epochs: 训练轮数
            batch_size: 批次大小
            shuffle: 是否打乱
            use_modulated: 是否使用调制版动力学
            emotion_encoder: 情感编码器 (use_modulated=True 时需要)
            
        Returns:
            训练历史
        """
        dataloader = ParagraphDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        
        # 学习率调度
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            return 1.0
        
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        
        # 选择动力学模型
        dynamics = self.modulated_dynamics if use_modulated and self.modulated_dynamics else self.dynamics
        dynamics.train()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_metrics = self._train_epoch(
                dataloader,
                epoch,
                dynamics,
                emotion_encoder,
            )
            self.history.extend(epoch_metrics)
        
        return self.history
    
    def _train_epoch(
        self,
        dataloader: ParagraphDataLoader,
        epoch: int,
        dynamics,
        emotion_encoder=None,
    ) -> List[DynamicsTrainingMetrics]:
        """训练一个 epoch"""
        epoch_metrics = []
        
        for batch_idx, batch in enumerate(dataloader):
            step_start = time.time()
            
            # 前向传播 + 计算损失
            losses = self._train_step(batch, dynamics, emotion_encoder)
            
            # 反向传播
            total_loss = losses["total_tensor"]
            total_loss.backward()
            
            # 梯度裁剪
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    dynamics.parameters(),
                    self.max_grad_norm,
                )
            
            # 优化器步进
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if self.scheduler:
                self.scheduler.step()
            
            # 记录指标
            step_time = time.time() - step_start
            samples_per_sec = len(batch.input_sequences) / step_time if step_time > 0 else 0
            
            metrics = DynamicsTrainingMetrics(
                epoch=epoch,
                step=self.global_step,
                total_loss=losses["total"],
                prediction_loss=losses.get("prediction", 0.0),
                ponder_loss=losses.get("ponder", 0.0),
                avg_reasoning_steps=losses.get("avg_steps", 1.0),
                learning_rate=self.optimizer.param_groups[0]["lr"],
                samples_per_sec=samples_per_sec,
            )
            
            epoch_metrics.append(metrics)
            
            # 日志
            if self.global_step % self.log_interval == 0:
                self._log_metrics(metrics)
            
            self.global_step += 1
        
        return epoch_metrics
    
    def _train_step(
        self,
        batch: SequenceBatch,
        dynamics,
        emotion_encoder=None,
    ) -> Dict[str, Any]:
        """单步训练"""
        input_sequences = batch.input_sequences  # [[p1, p2, p3], ...]
        target_paragraphs = batch.target_paragraphs  # [p4, ...]
        
        batch_size = len(input_sequences)
        seq_len = len(input_sequences[0])
        
        # 编码输入序列
        # 对每个序列中的每个段落分别编码
        with torch.no_grad():
            z_sequences = []
            for seq in input_sequences:
                # 编码整个序列
                output = self.encoder(seq)
                z_seq = output.latent.vector  # [seq_len, d_latent]
                z_sequences.append(z_seq)
            
            # 堆叠 [batch, seq_len, d_latent]
            z_input = torch.stack(z_sequences, dim=0)
            
            # 编码目标
            target_output = self.encoder(target_paragraphs)
            z_target = target_output.latent.vector  # [batch, d_latent]
        
        # 动力学预测
        if hasattr(dynamics, 'forward') and 'emotion' in str(dynamics.forward.__code__.co_varnames):
            # 调制版本
            emotion = None
            if emotion_encoder is not None:
                # 随机选择情感 (数据增强)
                emotion_ids = torch.randint(0, 8, (batch_size,))
                emotion = emotion_encoder(emotion_ids)
            
            output = dynamics(z_input, emotion=emotion)
        else:
            # 普通版本
            output = dynamics(z_input)
        
        z_pred = output.predicted_latent  # [batch, d_latent]
        
        # 计算预测损失
        prediction_loss = self.mse_loss(z_pred, z_target)
        
        # 可选：余弦相似度损失
        # cosine_target = torch.ones(batch_size, device=z_pred.device)
        # cosine_loss = self.cosine_loss(z_pred, z_target, cosine_target)
        
        # 总损失
        total_loss = prediction_loss
        
        # ACT ponder cost (如果有)
        ponder_loss = 0.0
        avg_steps = 1.0
        if hasattr(output, 'ponder_cost') and output.ponder_cost is not None:
            ponder_loss = output.ponder_cost
            total_loss = total_loss + self.ponder_cost_weight * ponder_loss
            ponder_loss = ponder_loss.item() if isinstance(ponder_loss, torch.Tensor) else ponder_loss
        
        if hasattr(output, 'num_steps') and output.num_steps is not None:
            avg_steps = output.num_steps.float().mean().item()
        
        return {
            "total_tensor": total_loss,
            "total": total_loss.item(),
            "prediction": prediction_loss.item(),
            "ponder": ponder_loss,
            "avg_steps": avg_steps,
        }
    
    def _log_metrics(self, metrics: DynamicsTrainingMetrics) -> None:
        """打印训练指标"""
        print(
            f"[E{metrics.epoch:02d} S{metrics.step:05d}] "
            f"loss={metrics.total_loss:.4f} "
            f"pred={metrics.prediction_loss:.4f} "
            f"ponder={metrics.ponder_loss:.4f} "
            f"steps={metrics.avg_reasoning_steps:.1f} "
            f"lr={metrics.learning_rate:.2e} "
            f"({metrics.samples_per_sec:.1f} samp/s)"
        )
    
    def evaluate(
        self,
        dataset: SequenceDataset,
        batch_size: int = 16,
    ) -> DynamicsTrainingMetrics:
        """评估模型"""
        self.dynamics.eval()
        
        dataloader = ParagraphDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        
        total_loss = 0.0
        total_pred = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                losses = self._eval_step(batch, self.dynamics)
                total_loss += losses.get("total", 0.0)
                total_pred += losses.get("prediction", 0.0)
                num_batches += 1
        
        avg_metrics = DynamicsTrainingMetrics(
            total_loss=total_loss / num_batches,
            prediction_loss=total_pred / num_batches,
        )
        
        self.dynamics.train()
        return avg_metrics
    
    def _eval_step(self, batch: SequenceBatch, dynamics) -> Dict[str, float]:
        """评估步骤"""
        input_sequences = batch.input_sequences
        target_paragraphs = batch.target_paragraphs
        
        # 编码
        z_sequences = []
        for seq in input_sequences:
            output = self.encoder(seq)
            z_sequences.append(output.latent.vector)
        
        z_input = torch.stack(z_sequences, dim=0)
        target_output = self.encoder(target_paragraphs)
        z_target = target_output.latent.vector
        
        # 预测
        output = dynamics(z_input)
        z_pred = output.predicted_latent
        
        # 损失
        prediction_loss = self.mse_loss(z_pred, z_target)
        
        return {
            "total": prediction_loss.item(),
            "prediction": prediction_loss.item(),
        }
    
    def save(self, path: str) -> None:
        """保存模型"""
        state = {
            "dynamics": self.dynamics.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
        }
        if self.modulated_dynamics:
            state["modulated_dynamics"] = self.modulated_dynamics.state_dict()
        
        torch.save(state, path)
    
    def load(self, path: str) -> None:
        """加载模型"""
        state = torch.load(path, map_location=self.device)
        
        self.dynamics.load_state_dict(state["dynamics"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.global_step = state.get("global_step", 0)
        self.current_epoch = state.get("current_epoch", 0)
        
        if self.modulated_dynamics and "modulated_dynamics" in state:
            self.modulated_dynamics.load_state_dict(state["modulated_dynamics"])
