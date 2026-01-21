"""
Residual VQ - 残差向量量化

AMHVQ+ 核心组件：多层残差量化，实现渐进式精度恢复。

原理:
    Layer 0: 粗粒度量化 → 捕捉主要语义
    Layer 1: 残差量化 → 捕捉风格细节
    Layer 2: 残差量化 → 捕捉精确细节
    
    重建: z ≈ q0 + q1 + q2 + ...
    
参考: SoundStream, Encodec, MAGVIT-2
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RVQOutput:
    """
    残差向量量化输出
    """
    quantized: torch.Tensor           # 最终量化结果 [batch, seq_len, d_latent]
    indices: torch.Tensor             # 所有层的码本索引 [batch, seq_len, num_layers]
    commitment_loss: torch.Tensor     # 总承诺损失
    codebook_loss: torch.Tensor       # 总码本损失
    perplexity: torch.Tensor          # 平均困惑度
    per_layer_quantized: List[torch.Tensor]  # 每层的量化结果 (用于渐进解码)
    per_layer_losses: List[torch.Tensor]     # 每层的损失


class ResidualVQLayer(nn.Module):
    """
    单层向量量化
    
    基于欧氏距离的最近邻码本查询。
    """
    
    def __init__(
        self,
        d_latent: int,
        codebook_size: int = 4096,
        commitment_cost: float = 0.25,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        """
        Args:
            d_latent: 潜空间维度
            codebook_size: 码本大小
            commitment_cost: 承诺损失系数
            use_ema: 是否使用 EMA 更新码本
            ema_decay: EMA 衰减率
            epsilon: 数值稳定性
        """
        super().__init__()
        
        self.d_latent = d_latent
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        
        # 码本嵌入 [codebook_size, d_latent]
        self.embedding = nn.Embedding(codebook_size, d_latent)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)
        
        if use_ema:
            self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
            self.register_buffer("ema_embedding_sum", self.embedding.weight.data.clone())
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            z: 输入向量 [batch, seq_len, d_latent] 或 [batch, d_latent]
            
        Returns:
            quantized: 量化后的向量
            indices: 码本索引
            commitment_loss: 承诺损失
            perplexity: 困惑度
        """
        # 处理维度
        input_shape = z.shape
        if z.dim() == 2:
            z = z.unsqueeze(1)  # [batch, 1, d_latent]
        
        batch_size, seq_len, d_latent = z.shape
        z_flat = z.reshape(-1, d_latent)  # [batch * seq_len, d_latent]
        
        # 计算与码本的距离
        # distances[i, j] = ||z_i - e_j||^2
        distances = (
            z_flat.pow(2).sum(dim=-1, keepdim=True)
            - 2 * z_flat @ self.embedding.weight.T
            + self.embedding.weight.pow(2).sum(dim=-1)
        )  # [batch * seq_len, codebook_size]
        
        # 找到最近的码字
        indices = distances.argmin(dim=-1)  # [batch * seq_len]
        
        # 获取量化向量
        z_quantized = self.embedding(indices)  # [batch * seq_len, d_latent]
        
        # 计算损失
        commitment_loss = F.mse_loss(z_flat, z_quantized.detach())
        
        if self.use_ema and self.training:
            self._update_ema(z_flat, indices)
            codebook_loss = torch.tensor(0.0, device=z.device)
        else:
            codebook_loss = F.mse_loss(z_flat.detach(), z_quantized)
        
        # Straight-Through Estimator
        z_quantized = z_flat + (z_quantized - z_flat).detach()
        
        # 计算困惑度
        perplexity = self._compute_perplexity(indices)
        
        # 恢复形状
        z_quantized = z_quantized.reshape(batch_size, seq_len, d_latent)
        indices = indices.reshape(batch_size, seq_len)
        
        # 如果输入是 2D，输出也是 2D
        if len(input_shape) == 2:
            z_quantized = z_quantized.squeeze(1)
            indices = indices.squeeze(1)
        
        return z_quantized, indices, self.commitment_cost * commitment_loss, perplexity
    
    def _update_ema(self, z_flat: torch.Tensor, indices: torch.Tensor) -> None:
        """EMA 更新码本"""
        one_hot = F.one_hot(indices, self.codebook_size).float()  # [N, codebook_size]
        
        # 更新 cluster size
        new_cluster_size = one_hot.sum(dim=0)
        self.ema_cluster_size.data = (
            self.ema_decay * self.ema_cluster_size + 
            (1 - self.ema_decay) * new_cluster_size
        )
        
        # 更新 embedding sum
        new_embedding_sum = one_hot.T @ z_flat  # [codebook_size, d_latent]
        self.ema_embedding_sum.data = (
            self.ema_decay * self.ema_embedding_sum +
            (1 - self.ema_decay) * new_embedding_sum
        )
        
        # 更新 embedding
        n = self.ema_cluster_size.sum()
        cluster_size = (
            (self.ema_cluster_size + self.epsilon) /
            (n + self.codebook_size * self.epsilon) * n
        )
        self.embedding.weight.data = self.ema_embedding_sum / cluster_size.unsqueeze(1)
    
    def _compute_perplexity(self, indices: torch.Tensor) -> torch.Tensor:
        """计算困惑度 (码本利用率)"""
        one_hot = F.one_hot(indices, self.codebook_size).float()
        avg_probs = one_hot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """从索引解码"""
        return self.embedding(indices)


class ResidualVQ(nn.Module):
    """
    残差向量量化 (RVQ)
    
    多层量化，每层编码上一层的残差。
    
    使用示例:
        rvq = ResidualVQ(d_latent=512, codebook_size=4096, num_layers=3)
        
        # 编码
        output = rvq(z)
        z_quantized = output.quantized  # 完整量化
        indices = output.indices        # [batch, seq_len, num_layers]
        
        # 渐进解码 (仅前 2 层)
        z_coarse = output.per_layer_quantized[0] + output.per_layer_quantized[1]
        
        # 从索引解码
        z_decoded = rvq.decode(indices)
    """
    
    def __init__(
        self,
        d_latent: int,
        codebook_size: int = 4096,
        num_layers: int = 3,
        commitment_cost: float = 0.25,
        use_ema: bool = True,
        ema_decay: float = 0.99,
    ):
        """
        Args:
            d_latent: 潜空间维度
            codebook_size: 每层码本大小
            num_layers: 量化层数
            commitment_cost: 承诺损失系数
            use_ema: 是否使用 EMA 更新码本
            ema_decay: EMA 衰减率
        """
        super().__init__()
        
        self.d_latent = d_latent
        self.codebook_size = codebook_size
        self.num_layers = num_layers
        
        # 创建多层 VQ
        self.layers = nn.ModuleList([
            ResidualVQLayer(
                d_latent=d_latent,
                codebook_size=codebook_size,
                commitment_cost=commitment_cost,
                use_ema=use_ema,
                ema_decay=ema_decay,
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, z: torch.Tensor) -> RVQOutput:
        """
        前向传播
        
        Args:
            z: 输入向量 [batch, seq_len, d_latent] 或 [batch, d_latent]
            
        Returns:
            RVQOutput
        """
        # 记录输入维度
        input_shape = z.shape
        if z.dim() == 2:
            z = z.unsqueeze(1)  # [batch, 1, d_latent]
        
        batch_size, seq_len, d_latent = z.shape
        
        residual = z
        quantized_sum = torch.zeros_like(z)
        
        all_indices = []
        all_quantized = []
        all_losses = []
        all_perplexities = []
        
        for layer in self.layers:
            # 量化当前残差
            quantized, indices, commitment_loss, perplexity = layer(residual)
            
            # 累加量化结果
            quantized_sum = quantized_sum + quantized
            
            # 计算新残差
            residual = residual - quantized.detach()
            
            # 记录
            all_indices.append(indices)
            all_quantized.append(quantized)
            all_losses.append(commitment_loss)
            all_perplexities.append(perplexity)
        
        # 堆叠索引
        indices = torch.stack(all_indices, dim=-1)  # [batch, seq_len, num_layers]
        
        # 总损失
        total_commitment = sum(all_losses)
        codebook_loss = torch.tensor(0.0, device=z.device)
        avg_perplexity = sum(all_perplexities) / len(all_perplexities)
        
        # 如果输入是 2D，输出也调整
        if len(input_shape) == 2:
            quantized_sum = quantized_sum.squeeze(1)
            indices = indices.squeeze(1)
            all_quantized = [q.squeeze(1) for q in all_quantized]
        
        return RVQOutput(
            quantized=quantized_sum,
            indices=indices,
            commitment_loss=total_commitment,
            codebook_loss=codebook_loss,
            perplexity=avg_perplexity,
            per_layer_quantized=all_quantized,
            per_layer_losses=all_losses,
        )
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        从索引解码
        
        Args:
            indices: 码本索引 [batch, seq_len, num_layers] 或 [batch, num_layers]
            
        Returns:
            解码后的向量 [batch, seq_len, d_latent] 或 [batch, d_latent]
        """
        input_shape = indices.shape
        if indices.dim() == 2:
            indices = indices.unsqueeze(1)  # [batch, 1, num_layers]
        
        batch_size, seq_len, num_layers = indices.shape
        
        # 累加各层的解码结果
        decoded = torch.zeros(batch_size, seq_len, self.d_latent, device=indices.device)
        
        for i, layer in enumerate(self.layers):
            layer_indices = indices[:, :, i]  # [batch, seq_len]
            decoded = decoded + layer.decode(layer_indices)
        
        # 恢复形状
        if len(input_shape) == 2:
            decoded = decoded.squeeze(1)
        
        return decoded
    
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """
        仅编码，返回索引
        
        Args:
            z: 输入向量
            
        Returns:
            indices: 码本索引
        """
        output = self.forward(z)
        return output.indices
    
    def get_codebook_usage(self) -> List[float]:
        """获取各层码本利用率"""
        usages = []
        for layer in self.layers:
            if hasattr(layer, 'ema_cluster_size'):
                used = (layer.ema_cluster_size > 0).float().mean().item()
                usages.append(used)
            else:
                usages.append(1.0)
        return usages


# ============================================================
# 工具函数
# ============================================================

def progressive_decode(
    rvq_output: RVQOutput,
    num_layers: int,
) -> torch.Tensor:
    """
    渐进式解码 (仅使用前 N 层)
    
    Args:
        rvq_output: RVQ 输出
        num_layers: 使用的层数
        
    Returns:
        部分解码的向量
    """
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    
    num_layers = min(num_layers, len(rvq_output.per_layer_quantized))
    
    result = rvq_output.per_layer_quantized[0]
    for i in range(1, num_layers):
        result = result + rvq_output.per_layer_quantized[i]
    
    return result
