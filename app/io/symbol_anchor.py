"""
Symbol Anchor - 符号锚点编码器

AMHVQ+ 符号通道：识别并锚定关键 token (变量名、函数名、括号等)。

原理:
    1. 识别文本中的"关键" token (需要精确保留的)
    2. 存储其精确 token ID 和位置
    3. 解码时强制使用锚定的 token
    
关键 token 类型:
    - 标识符 (变量名、函数名)
    - 括号对 (匹配的括号)
    - 运算符
    - 引号内的字符串
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Set
import torch
import torch.nn as nn
import torch.nn.functional as F
import re


@dataclass
class SymbolAnchor:
    """符号锚点"""
    position: int               # token 位置
    token_id: int               # 精确 token ID
    token_text: str = ""        # 原始文本
    slot_id: Optional[int] = None  # 对应结构槽位
    anchor_type: str = "identifier"  # 锚点类型
    is_critical: bool = True    # 是否关键
    confidence: float = 1.0     # 置信度
    
    def __repr__(self) -> str:
        return f"Anchor(pos={self.position}, '{self.token_text}', type={self.anchor_type})"


@dataclass
class SymbolAnchors:
    """符号锚点集合"""
    anchors: List[SymbolAnchor] = field(default_factory=list)
    vector: Optional[torch.Tensor] = None  # 锚点编码 [d_symbol]
    
    @property
    def num_anchors(self) -> int:
        return len(self.anchors)
    
    def get_by_position(self, pos: int) -> Optional[SymbolAnchor]:
        for anchor in self.anchors:
            if anchor.position == pos:
                return anchor
        return None
    
    def get_by_slot(self, slot_id: int) -> Optional[SymbolAnchor]:
        for anchor in self.anchors:
            if anchor.slot_id == slot_id:
                return anchor
        return None
    
    def get_token_ids(self) -> List[int]:
        return [a.token_id for a in self.anchors]
    
    def get_positions(self) -> List[int]:
        return [a.position for a in self.anchors]
    
    def to_dict(self) -> Dict[int, int]:
        """位置 -> token_id 映射"""
        return {a.position: a.token_id for a in self.anchors}
    
    def to(self, device: torch.device) -> "SymbolAnchors":
        return SymbolAnchors(
            anchors=self.anchors,
            vector=self.vector.to(device) if self.vector is not None else None,
        )


class CriticalTokenDetector(nn.Module):
    """
    关键 token 检测器
    
    识别需要精确保留的 token。
    """
    
    def __init__(
        self,
        d_model: int,
        threshold: float = 0.5,
        max_anchors: int = 32,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.threshold = threshold
        self.max_anchors = max_anchors
        
        # 基于注意力的检测器
        self.detector = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        
        # 关键 token 类型 (规则增强)
        self.critical_patterns = {
            "identifier": re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$'),
            "bracket": re.compile(r'^[\(\)\[\]\{\}]$'),
            "operator": re.compile(r'^[+\-*/=<>!&|]+$'),
            "string": re.compile(r'^["\']'),
            "number": re.compile(r'^[0-9]+\.?[0-9]*$'),
        }
        
        # 排除的常见词
        self.exclude_words: Set[str] = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall',
            'and', 'or', 'but', 'if', 'then', 'else', 'for', 'while',
            'def', 'class', 'return', 'import', 'from', 'as', 'in',
        }
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        token_texts: Optional[List[str]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        检测关键 token
        
        Args:
            hidden_states: [batch, seq_len, d_model]
            token_texts: 每个 token 的文本 (用于规则增强)
            attention_mask: [batch, seq_len]
            
        Returns:
            scores: [batch, seq_len] 关键性分数
            critical_mask: [batch, seq_len] 是否为关键 token
        """
        # 神经网络预测
        scores = self.detector(hidden_states).squeeze(-1)  # [batch, seq_len]
        scores = torch.sigmoid(scores)
        
        # 规则增强
        if token_texts is not None:
            rule_scores = self._rule_based_scores(token_texts, scores.device)
            # 规则分数提升神经网络分数
            scores = scores + 0.5 * rule_scores
            scores = scores.clamp(0, 1)
        
        # 应用掩码
        if attention_mask is not None:
            scores = scores * attention_mask
        
        # 阈值判断
        critical_mask = (scores > self.threshold).float()
        
        return scores, critical_mask
    
    def _rule_based_scores(
        self,
        token_texts: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        """基于规则计算分数"""
        # 假设是单个样本的 token 列表
        scores = []
        for text in token_texts:
            score = 0.0
            text_lower = text.lower()
            
            # 排除常见词
            if text_lower in self.exclude_words:
                score = 0.0
            # 检查模式
            elif self.critical_patterns["identifier"].match(text):
                score = 0.8
            elif self.critical_patterns["bracket"].match(text):
                score = 1.0  # 括号必须精确
            elif self.critical_patterns["operator"].match(text):
                score = 0.9
            elif self.critical_patterns["number"].match(text):
                score = 0.7
            elif self.critical_patterns["string"].match(text):
                score = 0.7
            
            scores.append(score)
        
        return torch.tensor(scores, device=device).unsqueeze(0)


class SymbolAnchorEncoder(nn.Module):
    """
    符号锚点编码器
    
    将检测到的锚点编码为向量。
    """
    
    def __init__(
        self,
        d_model: int,
        d_output: int = 128,
        max_anchors: int = 32,
        threshold: float = 0.5,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_output = d_output
        self.max_anchors = max_anchors
        
        # 关键 token 检测器
        self.detector = CriticalTokenDetector(
            d_model=d_model,
            threshold=threshold,
            max_anchors=max_anchors,
        )
        
        # 锚点编码器
        self.anchor_encoder = nn.Sequential(
            nn.Linear(d_model + 1, d_model),  # +1 for position encoding
            nn.GELU(),
            nn.Linear(d_model, d_output),
        )
        
        # 聚合所有锚点
        self.aggregator = nn.Sequential(
            nn.Linear(d_output, d_output),
            nn.Tanh(),
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        token_texts: Optional[List[List[str]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[SymbolAnchors, torch.Tensor]:
        """
        检测并编码符号锚点
        
        Args:
            hidden_states: [batch, seq_len, d_model]
            token_ids: [batch, seq_len]
            token_texts: 每个样本的 token 文本列表
            attention_mask: [batch, seq_len]
            
        Returns:
            anchors: SymbolAnchors (第一个样本的)
            anchor_vector: [batch, d_output] 锚点摘要向量
        """
        batch_size, seq_len, d_model = hidden_states.shape
        device = hidden_states.device
        
        # 检测关键 token
        texts = token_texts[0] if token_texts else None
        scores, critical_mask = self.detector(hidden_states, texts, attention_mask)
        
        # 提取锚点 (仅处理第一个样本，简化)
        anchors = self._extract_anchors(
            scores[0],
            critical_mask[0],
            token_ids[0],
            token_texts[0] if token_texts else None,
        )
        
        # 编码锚点
        anchor_vectors = []
        for b in range(batch_size):
            batch_anchors = []
            positions = torch.where(critical_mask[b] > 0)[0][:self.max_anchors]
            
            for pos in positions:
                pos_encoding = torch.tensor([pos.item() / seq_len], device=device)
                combined = torch.cat([hidden_states[b, pos], pos_encoding])
                anchor_vec = self.anchor_encoder(combined)
                batch_anchors.append(anchor_vec)
            
            if batch_anchors:
                stacked = torch.stack(batch_anchors)
                aggregated = self.aggregator(stacked.mean(dim=0))
            else:
                aggregated = torch.zeros(self.d_output, device=device)
            
            anchor_vectors.append(aggregated)
        
        anchor_vector = torch.stack(anchor_vectors)  # [batch, d_output]
        
        # 设置向量
        anchors.vector = anchor_vector[0]
        
        return anchors, anchor_vector
    
    def _extract_anchors(
        self,
        scores: torch.Tensor,
        critical_mask: torch.Tensor,
        token_ids: torch.Tensor,
        token_texts: Optional[List[str]],
    ) -> SymbolAnchors:
        """提取锚点信息"""
        anchors = SymbolAnchors()
        
        positions = torch.where(critical_mask > 0)[0][:self.max_anchors]
        
        for i, pos in enumerate(positions):
            pos_int = pos.item()
            anchor = SymbolAnchor(
                position=pos_int,
                token_id=token_ids[pos_int].item(),
                token_text=token_texts[pos_int] if token_texts and pos_int < len(token_texts) else "",
                slot_id=i,
                confidence=scores[pos_int].item(),
            )
            anchors.anchors.append(anchor)
        
        return anchors


# ============================================================
# 工具函数
# ============================================================

def detect_critical_tokens(
    text: str,
    tokenizer=None,
) -> List[Tuple[int, str, str]]:
    """
    检测文本中的关键 token
    
    Returns:
        [(position, token_text, type), ...]
    """
    critical = []
    
    # 标识符
    for match in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', text):
        word = match.group(1)
        if word not in {'def', 'class', 'if', 'for', 'while', 'return', 'import',
                        'from', 'as', 'in', 'and', 'or', 'not', 'is', 'True', 'False', 'None'}:
            critical.append((match.start(), word, "identifier"))
    
    # 括号
    for match in re.finditer(r'[\(\)\[\]\{\}]', text):
        critical.append((match.start(), match.group(), "bracket"))
    
    return critical


def apply_anchors_to_tokens(
    token_ids: torch.Tensor,
    anchors: SymbolAnchors,
    generated_ids: torch.Tensor,
) -> torch.Tensor:
    """
    将锚点应用到生成的 token
    
    强制替换锚定位置的 token。
    
    Args:
        token_ids: 原始 token ID
        anchors: 锚点信息
        generated_ids: 生成的 token ID
        
    Returns:
        修正后的 token ID
    """
    result = generated_ids.clone()
    
    for anchor in anchors.anchors:
        pos = anchor.position
        if pos < result.shape[-1]:
            result[..., pos] = anchor.token_id
    
    return result
