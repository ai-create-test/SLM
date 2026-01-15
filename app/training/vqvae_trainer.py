"""
VQ-VAE Trainer - 向量量化变分自编码器训练器

核心功能：
1. 训练 Encoder 将段落压缩为潜向量
2. 训练 VQ Codebook 学习离散表示
3. 训练 Decoder 从潜向量重建文本

损失函数:
    L = L_recon + β * L_commit + γ * L_codebook
    
    L_recon: 重建损失 (Encoder 输出 ≈ Decoder 输入到 Encoder 的输出)
    L_commit: 承诺损失 (让 Encoder 输出靠近码本)
    L_codebook: 码本损失 (EMA 模式下为 0)
"""

from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from ..io.paragraph_encoder import ParagraphEncoder, EncoderOutput
from ..io.paragraph_decoder import ParagraphDecoder
from ..io.vq_codebook import VQCodebook
from .data_loader import ParagraphDataset, ParagraphDataLoader, ParagraphBatch


@dataclass
class TrainingMetrics:
    """训练指标"""
    epoch: int = 0
    step: int = 0
    total_loss: float = 0.0
    recon_loss: float = 0.0
    commit_loss: float = 0.0
    codebook_loss: float = 0.0
    perplexity: float = 0.0
    codebook_utilization: float = 0.0
    learning_rate: float = 0.0
    samples_per_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "total_loss": self.total_loss,
            "recon_loss": self.recon_loss,
            "commit_loss": self.commit_loss,
            "codebook_loss": self.codebook_loss,
            "perplexity": self.perplexity,
            "codebook_utilization": self.codebook_utilization,
            "learning_rate": self.learning_rate,
            "samples_per_sec": self.samples_per_sec,
        }


class VQVAETrainer:
    """
    VQ-VAE 训练器
    
    训练编码器-码本-解码器的端到端流程。
    
    使用示例:
        trainer = VQVAETrainer(
            encoder=encoder,
            decoder=decoder,
            codebook=codebook,
            learning_rate=1e-4,
        )
        
        # 训练
        dataset = ParagraphDataset.synthetic(1000)
        trainer.train(dataset, num_epochs=10)
    """
    
    def __init__(
        self,
        encoder: ParagraphEncoder,
        decoder: Optional[ParagraphDecoder] = None,
        codebook: Optional[VQCodebook] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        beta_commit: float = 0.25,
        gamma_codebook: float = 1.0,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
        log_interval: int = 10,
    ):
        """
        Args:
            encoder: 段落编码器
            decoder: 段落解码器 (可选，如果为 None 则不计算重建损失)
            codebook: VQ 码本 (可选，如果为 None 则使用 encoder 内置的)
            learning_rate: 学习率
            weight_decay: 权重衰减
            beta_commit: 承诺损失系数
            gamma_codebook: 码本损失系数
            warmup_steps: 预热步数
            max_grad_norm: 梯度裁剪
            device: 设备
            log_interval: 日志间隔
        """
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device) if decoder else None
        self.codebook = codebook or getattr(encoder, 'codebook', None)
        
        self.learning_rate = learning_rate
        self.beta_commit = beta_commit
        self.gamma_codebook = gamma_codebook
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.log_interval = log_interval
        
        # 收集可训练参数
        params = list(encoder.parameters())
        if decoder:
            params += list(decoder.parameters())
        if codebook and codebook not in encoder.modules():
            params += list(codebook.parameters())
        
        # 优化器
        self.optimizer = optim.AdamW(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        
        # 学习率调度器
        self.scheduler = None  # 在 train() 中根据 epoch 数设置
        
        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.history: List[TrainingMetrics] = []
    
    def train(
        self,
        dataset: ParagraphDataset,
        num_epochs: int = 10,
        batch_size: int = 32,
        shuffle: bool = True,
        reset_dead_codes_interval: int = 100,
    ) -> List[TrainingMetrics]:
        """
        训练 VQ-VAE
        
        Args:
            dataset: 段落数据集
            num_epochs: 训练轮数
            batch_size: 批次大小
            shuffle: 是否打乱
            reset_dead_codes_interval: 死码重置间隔
            
        Returns:
            训练历史指标列表
        """
        dataloader = ParagraphDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        
        total_steps = num_epochs * len(dataloader)
        
        # 设置学习率调度器
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            return 1.0
        
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        
        # 训练循环
        self.encoder.train()
        if self.decoder:
            self.decoder.train()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_metrics = self._train_epoch(
                dataloader,
                epoch,
                reset_dead_codes_interval,
            )
            self.history.extend(epoch_metrics)
        
        return self.history
    
    def _train_epoch(
        self,
        dataloader: ParagraphDataLoader,
        epoch: int,
        reset_dead_codes_interval: int,
    ) -> List[TrainingMetrics]:
        """训练一个 epoch"""
        epoch_metrics = []
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            step_start = time.time()
            
            # 前向传播 + 计算损失
            losses = self._train_step(batch)
            
            # 梯度裁剪
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.encoder.parameters(),
                    self.max_grad_norm,
                )
            
            # 优化器步进
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if self.scheduler:
                self.scheduler.step()
            
            # 死码重置
            if self.codebook and self.global_step % reset_dead_codes_interval == 0:
                reset_count = self.codebook.reset_dead_codes()
            
            # 记录指标
            step_time = time.time() - step_start
            samples_per_sec = len(batch.texts) / step_time if step_time > 0 else 0
            
            metrics = TrainingMetrics(
                epoch=epoch,
                step=self.global_step,
                total_loss=losses["total"],
                recon_loss=losses.get("recon", 0.0),
                commit_loss=losses.get("commit", 0.0),
                codebook_loss=losses.get("codebook", 0.0),
                perplexity=losses.get("perplexity", 0.0),
                codebook_utilization=self.codebook.codebook_utilization if self.codebook else 0.0,
                learning_rate=self.optimizer.param_groups[0]["lr"],
                samples_per_sec=samples_per_sec,
            )
            
            epoch_metrics.append(metrics)
            
            # 日志
            if self.global_step % self.log_interval == 0:
                self._log_metrics(metrics)
            
            self.global_step += 1
        
        return epoch_metrics
    
    def _train_step(self, batch: ParagraphBatch) -> Dict[str, float]:
        """单步训练"""
        texts = batch.texts
        
        # 编码
        encoder_output = self.encoder(texts)
        
        # 获取损失
        total_loss = 0.0
        losses = {}
        
        # VQ 损失
        if encoder_output.vq_output is not None:
            vq_out = encoder_output.vq_output
            commit_loss = vq_out.commitment_loss
            codebook_loss = vq_out.codebook_loss
            perplexity = vq_out.perplexity.item()
            
            losses["commit"] = commit_loss.item()
            losses["codebook"] = codebook_loss.item()
            losses["perplexity"] = perplexity
            
            total_loss = total_loss + self.beta_commit * commit_loss
            total_loss = total_loss + self.gamma_codebook * codebook_loss
        
        # 重建损失 (如果有 decoder)
        if self.decoder is not None:
            latent = encoder_output.latent
            
            # 解码
            decoder_output = self.decoder(latent)
            
            # 重建损失：比较解码器输出与原始编码
            # 这里简化为 logits 与某种目标之间的损失
            # 实际应该是 cross-entropy with teacher forcing
            if decoder_output.logits is not None:
                # 使用池化向量作为目标的代理
                target = encoder_output.pooled
                if target is not None:
                    recon_loss = torch.nn.functional.mse_loss(
                        decoder_output.logits.mean(dim=1),  # 简化
                        target,
                    )
                    losses["recon"] = recon_loss.item()
                    total_loss = total_loss + recon_loss
        
        # 如果没有任何损失，使用编码器输出作为自监督
        if total_loss == 0.0 and encoder_output.pooled is not None:
            # 自监督：预测自己
            pooled = encoder_output.pooled
            noise = torch.randn_like(pooled) * 0.01
            noisy = pooled + noise
            recon_loss = torch.nn.functional.mse_loss(noisy, pooled)
            losses["recon"] = recon_loss.item()
            total_loss = recon_loss
        
        # 反向传播
        if isinstance(total_loss, torch.Tensor):
            total_loss.backward()
            losses["total"] = total_loss.item()
        else:
            losses["total"] = total_loss
        
        return losses
    
    def _log_metrics(self, metrics: TrainingMetrics) -> None:
        """打印训练指标"""
        print(
            f"[E{metrics.epoch:02d} S{metrics.step:05d}] "
            f"loss={metrics.total_loss:.4f} "
            f"commit={metrics.commit_loss:.4f} "
            f"ppl={metrics.perplexity:.1f} "
            f"util={metrics.codebook_utilization*100:.1f}% "
            f"lr={metrics.learning_rate:.2e} "
            f"({metrics.samples_per_sec:.1f} samp/s)"
        )
    
    def evaluate(
        self,
        dataset: ParagraphDataset,
        batch_size: int = 32,
    ) -> TrainingMetrics:
        """
        评估模型
        
        Returns:
            平均指标
        """
        self.encoder.eval()
        if self.decoder:
            self.decoder.eval()
        
        dataloader = ParagraphDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        
        total_loss = 0.0
        total_commit = 0.0
        total_ppl = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                losses = self._eval_step(batch)
                total_loss += losses.get("total", 0.0)
                total_commit += losses.get("commit", 0.0)
                total_ppl += losses.get("perplexity", 0.0)
                num_batches += 1
        
        avg_metrics = TrainingMetrics(
            total_loss=total_loss / num_batches,
            commit_loss=total_commit / num_batches,
            perplexity=total_ppl / num_batches,
            codebook_utilization=self.codebook.codebook_utilization if self.codebook else 0.0,
        )
        
        self.encoder.train()
        if self.decoder:
            self.decoder.train()
        
        return avg_metrics
    
    def _eval_step(self, batch: ParagraphBatch) -> Dict[str, float]:
        """评估步骤 (无梯度)"""
        texts = batch.texts
        encoder_output = self.encoder(texts)
        
        losses = {"total": 0.0}
        
        if encoder_output.vq_output is not None:
            vq_out = encoder_output.vq_output
            losses["commit"] = vq_out.commitment_loss.item()
            losses["perplexity"] = vq_out.perplexity.item()
            losses["total"] = vq_out.commitment_loss.item()
        
        return losses
    
    def save(self, path: str) -> None:
        """保存模型和训练状态"""
        state = {
            "encoder": self.encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
        }
        if self.decoder:
            state["decoder"] = self.decoder.state_dict()
        if self.codebook:
            state["codebook"] = self.codebook.state_dict()
        
        torch.save(state, path)
    
    def load(self, path: str) -> None:
        """加载模型和训练状态"""
        state = torch.load(path, map_location=self.device)
        
        self.encoder.load_state_dict(state["encoder"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.global_step = state.get("global_step", 0)
        self.current_epoch = state.get("current_epoch", 0)
        
        if self.decoder and "decoder" in state:
            self.decoder.load_state_dict(state["decoder"])
        if self.codebook and "codebook" in state:
            self.codebook.load_state_dict(state["codebook"])
