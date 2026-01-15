"""
VQ Codebook - 向量量化码本

核心组件：实现离散潜变量的关键

设计原则:
1. EMA (Exponential Moving Average) 码本更新
2. 多头 VQ (Product Quantization 风格)
3. 码本利用率监控与重置
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQOutput:
    """
    向量量化输出
    """
    quantized: torch.Tensor          # 量化后的向量  [batch, d_latent]
    indices: torch.Tensor            # 码本索引      [batch, num_codebooks]
    commitment_loss: torch.Tensor    # 承诺损失
    codebook_loss: torch.Tensor      # 码本损失 (EMA 模式下为 0)
    perplexity: torch.Tensor         # 困惑度 (码本利用率指标)


class VQCodebook(nn.Module):
    """
    向量量化码本
    
    实现 VQ-VAE 中的离散瓶颈层。
    
    核心公式:
        z_q = CodeBook[argmin_j ||z_e - e_j||]
    
    训练时:
        Loss = ||z_e - sg[z_q]||^2 (commitment) + ||sg[z_e] - z_q||^2 (codebook)
        或使用 EMA 更新码本 (推荐)
    
    使用示例:
        codebook = VQCodebook(d_latent=512, codebook_size=8192)
        z_continuous = encoder(paragraph)  # [batch, d_latent]
        vq_out = codebook(z_continuous)
        z_quantized = vq_out.quantized     # 离散化后的向量
    """
    
    def __init__(
        self,
        d_latent: int,
        codebook_size: int = 8192,
        num_codebooks: int = 1,
        commitment_cost: float = 0.25,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_dead_code: int = 2,
    ):
        """
        Args:
            d_latent: 潜空间维度
            codebook_size: 每个码本的条目数
            num_codebooks: 码本数量 (多头 VQ)
            commitment_cost: 承诺损失系数
            use_ema: 是否使用 EMA 更新码本
            ema_decay: EMA 衰减率
            epsilon: 数值稳定性
            threshold_dead_code: 死码重置阈值
        """
        super().__init__()
        
        assert d_latent % num_codebooks == 0, \
            f"d_latent ({d_latent}) must be divisible by num_codebooks ({num_codebooks})"
        
        self.d_latent = d_latent
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.d_subcode = d_latent // num_codebooks
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.threshold_dead_code = threshold_dead_code
        
        # 码本嵌入
        # [num_codebooks, codebook_size, d_subcode]
        self.embedding = nn.Parameter(
            torch.randn(num_codebooks, codebook_size, self.d_subcode)
        )
        nn.init.uniform_(self.embedding, -1.0 / codebook_size, 1.0 / codebook_size)
        
        if use_ema:
            # EMA 相关的缓冲区
            self.register_buffer("ema_cluster_size", torch.zeros(num_codebooks, codebook_size))
            self.register_buffer("ema_embedding_sum", self.embedding.data.clone())
            self.register_buffer("code_usage", torch.zeros(num_codebooks, codebook_size))
        
    def forward(self, z: torch.Tensor) -> VQOutput:
        """
        向量量化前向传播
        
        Args:
            z: 连续潜向量 [batch, d_latent]
            
        Returns:
            VQOutput 包含量化向量、索引和损失
        """
        batch_size = z.shape[0]
        
        # 分割为多个子码
        # [batch, num_codebooks, d_subcode]
        z_split = z.view(batch_size, self.num_codebooks, self.d_subcode)
        
        # 计算与码本的距离
        # [batch, num_codebooks, codebook_size]
        distances = self._compute_distances(z_split)
        
        # 找到最近的码字
        # [batch, num_codebooks]
        indices = distances.argmin(dim=-1)
        
        # 获取量化后的向量
        # [batch, num_codebooks, d_subcode]
        z_quantized = self._get_quantized(indices)
        
        # 计算损失
        commitment_loss = F.mse_loss(z_split, z_quantized.detach())
        
        if self.use_ema:
            codebook_loss = torch.tensor(0.0, device=z.device)
            if self.training:
                self._update_ema(z_split, indices)
        else:
            codebook_loss = F.mse_loss(z_split.detach(), z_quantized)
        
        # Straight-Through Estimator
        z_quantized = z_split + (z_quantized - z_split).detach()
        
        # 恢复形状
        # [batch, d_latent]
        z_quantized = z_quantized.view(batch_size, self.d_latent)
        
        # 计算困惑度 (码本利用率指标)
        perplexity = self._compute_perplexity(indices)
        
        return VQOutput(
            quantized=z_quantized,
            indices=indices,
            commitment_loss=self.commitment_cost * commitment_loss,
            codebook_loss=codebook_loss,
            perplexity=perplexity,
        )
    
    def _compute_distances(self, z: torch.Tensor) -> torch.Tensor:
        """计算与码本的欧氏距离"""
        # z: [batch, num_codebooks, d_subcode]
        # embedding: [num_codebooks, codebook_size, d_subcode]
        
        # 扩展维度进行广播
        z_expanded = z.unsqueeze(2)  # [batch, num_codebooks, 1, d_subcode]
        emb_expanded = self.embedding.unsqueeze(0)  # [1, num_codebooks, codebook_size, d_subcode]
        
        # 计算欧氏距离的平方
        distances = ((z_expanded - emb_expanded) ** 2).sum(dim=-1)
        
        return distances  # [batch, num_codebooks, codebook_size]
    
    def _get_quantized(self, indices: torch.Tensor) -> torch.Tensor:
        """根据索引获取量化向量"""
        batch_size = indices.shape[0]
        
        # indices: [batch, num_codebooks]
        # 需要对每个 codebook 分别索引
        quantized = []
        for i in range(self.num_codebooks):
            # [batch, d_subcode]
            q = self.embedding[i][indices[:, i]]
            quantized.append(q)
        
        # [batch, num_codebooks, d_subcode]
        return torch.stack(quantized, dim=1)
    
    def _update_ema(self, z: torch.Tensor, indices: torch.Tensor) -> None:
        """EMA 更新码本"""
        batch_size = z.shape[0]
        
        for i in range(self.num_codebooks):
            # 计算每个码字被选中的次数
            one_hot = F.one_hot(indices[:, i], self.codebook_size).float()
            
            # 更新 cluster size
            new_cluster_size = one_hot.sum(dim=0)
            self.ema_cluster_size[i] = (
                self.ema_decay * self.ema_cluster_size[i] +
                (1 - self.ema_decay) * new_cluster_size
            )
            
            # 更新 embedding sum
            new_embedding_sum = one_hot.T @ z[:, i, :]
            self.ema_embedding_sum[i] = (
                self.ema_decay * self.ema_embedding_sum[i] +
                (1 - self.ema_decay) * new_embedding_sum
            )
            
            # 更新 embedding
            n = self.ema_cluster_size[i].sum()
            cluster_size = (
                (self.ema_cluster_size[i] + self.epsilon) /
                (n + self.codebook_size * self.epsilon) * n
            )
            
            self.embedding.data[i] = self.ema_embedding_sum[i] / cluster_size.unsqueeze(1)
            
            # 更新使用计数
            self.code_usage[i] += new_cluster_size
    
    def _compute_perplexity(self, indices: torch.Tensor) -> torch.Tensor:
        """计算码本利用率 (困惑度)"""
        # 统计每个码字的使用概率
        avg_probs_list = []
        for i in range(self.num_codebooks):
            one_hot = F.one_hot(indices[:, i], self.codebook_size).float()
            avg_probs = one_hot.mean(dim=0)
            avg_probs_list.append(avg_probs)
        
        avg_probs = torch.stack(avg_probs_list).mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return perplexity
    
    def reset_dead_codes(self) -> int:
        """
        重置未使用的码字
        
        Returns:
            重置的码字数量
        """
        if not self.use_ema:
            return 0
        
        reset_count = 0
        for i in range(self.num_codebooks):
            dead_codes = self.code_usage[i] < self.threshold_dead_code
            num_dead = dead_codes.sum().item()
            
            if num_dead > 0:
                # 用随机向量重置
                new_codes = torch.randn(num_dead, self.d_subcode, device=self.embedding.device)
                new_codes = new_codes * 0.01  # 小随机值
                self.embedding.data[i, dead_codes] = new_codes
                
                # 重置 EMA 统计
                self.ema_cluster_size[i, dead_codes] = 0
                self.ema_embedding_sum[i, dead_codes] = new_codes
                
                reset_count += num_dead
        
        # 清零使用计数
        self.code_usage.zero_()
        
        return reset_count
    
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        从离散索引解码回连续向量
        
        Args:
            indices: 码本索引 [batch, num_codebooks]
            
        Returns:
            连续向量 [batch, d_latent]
        """
        batch_size = indices.shape[0]
        z_quantized = self._get_quantized(indices)
        return z_quantized.view(batch_size, self.d_latent)
    
    @property
    def codebook_utilization(self) -> float:
        """返回码本利用率 (0-1)"""
        if self.use_ema:
            used = (self.code_usage > 0).float().mean().item()
            return used
        return 1.0
