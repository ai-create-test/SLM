"""
Self Critic - 自我评价器

评估生成内容的质量，为回溯提供依据。
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn


class SelfCritic(nn.Module):
    """
    自我评价器
    
    评估生成内容的质量，输出评分。
    
    使用示例:
        critic = SelfCritic(d_latent=512)
        
        # 评估
        score = critic(generated_latent, context)
        
        if score < threshold:
            # 触发回溯
            ...
    """
    
    def __init__(
        self,
        d_latent: int = 512,
        d_context: int = 512,
        num_criteria: int = 4,
    ):
        """
        Args:
            d_latent: 潜向量维度
            d_context: 上下文维度
            num_criteria: 评价维度数量
        """
        super().__init__()
        
        self.d_latent = d_latent
        self.num_criteria = num_criteria
        
        # 多维度评价
        self.criteria_names = [
            "coherence",    # 连贯性
            "relevance",    # 相关性
            "quality",      # 质量
            "confidence",   # 置信度
        ][:num_criteria]
        
        # 评价网络
        self.scorer = nn.Sequential(
            nn.Linear(d_latent + d_context, d_latent),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_latent, d_latent // 2),
            nn.ReLU(),
            nn.Linear(d_latent // 2, num_criteria),
            nn.Sigmoid(),  # 输出 0-1 分数
        )
    
    def forward(
        self,
        generated: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        评估生成内容
        
        Args:
            generated: 生成的潜向量 [batch, d_latent]
            context: 上下文向量 [batch, d_context]
            
        Returns:
            综合分数 [batch, 1]
        """
        if context is None:
            context = torch.zeros_like(generated)
        
        # 拼接输入
        combined = torch.cat([generated, context], dim=-1)
        
        # 多维度评分
        scores = self.scorer(combined)  # [batch, num_criteria]
        
        # 综合分数 (加权平均)
        overall = scores.mean(dim=-1, keepdim=True)
        
        return overall
    
    def evaluate_detailed(
        self,
        generated: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        详细评估
        
        Returns:
            各维度分数的字典
        """
        if context is None:
            context = torch.zeros_like(generated)
        
        combined = torch.cat([generated, context], dim=-1)
        scores = self.scorer(combined)  # [batch, num_criteria]
        
        # 取平均 (batch 维度)
        scores_mean = scores.mean(dim=0)
        
        result = {
            name: scores_mean[i].item()
            for i, name in enumerate(self.criteria_names)
        }
        result["overall"] = scores_mean.mean().item()
        
        return result
    
    def should_reject(
        self,
        generated: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        threshold: float = 0.3,
    ) -> bool:
        """
        判断是否应该拒绝/回溯
        """
        score = self.forward(generated, context)
        return score.mean().item() < threshold
