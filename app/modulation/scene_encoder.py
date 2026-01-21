"""
Scene Encoder - 场景模式编码器

将离散的场景模式编码为连续向量，用于调制推理风格。
"""

from typing import Dict, Optional, Union
import torch
import torch.nn as nn


# 场景 ID 映射
SCENE_IDS: Dict[str, int] = {
    "chat": 0,          # 日常对话
    "coding": 1,        # 编程辅助
    "debate": 2,        # 辩论/论证
    "creative": 3,      # 创意写作
    "analysis": 4,      # 数据分析
    "teaching": 5,      # 教学解释
    "roleplay": 6,      # 角色扮演
    "formal": 7,        # 正式文档
}

# 反向映射
ID_TO_SCENE = {v: k for k, v in SCENE_IDS.items()}


class SceneEncoder(nn.Module):
    """
    场景模式编码器
    
    将离散的场景 ID 编码为稠密向量。
    
    使用示例:
        encoder = SceneEncoder(d_scene=128)
        
        # 从 ID 编码
        scene_vec = encoder(torch.tensor([1]))  # Coding
        
        # 从名称编码
        scene_vec = encoder.encode_name("creative")
    """
    
    def __init__(
        self,
        d_scene: int = 128,
        num_scenes: int = 8,
        init_std: float = 0.02,
    ):
        """
        Args:
            d_scene: 场景向量维度
            num_scenes: 场景类别数
            init_std: 权重初始化标准差
        """
        super().__init__()
        
        self.d_scene = d_scene
        self.num_scenes = num_scenes
        
        # 场景嵌入层
        self.embedding = nn.Embedding(
            num_embeddings=num_scenes,
            embedding_dim=d_scene,
        )
        
        # 初始化
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)
    
    def forward(
        self,
        scene_id: Union[torch.Tensor, int],
    ) -> torch.Tensor:
        """
        编码场景 ID
        
        Args:
            scene_id: 场景 ID [batch] 或 单个 int
            
        Returns:
            场景向量 [batch, d_scene]
        """
        if isinstance(scene_id, int):
            scene_id = torch.tensor([scene_id], device=self.embedding.weight.device)
        
        if scene_id.dim() > 1:
            scene_id = scene_id.squeeze(-1)
        
        return self.embedding(scene_id)
    
    def encode_name(
        self,
        scene_name: str,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        从场景名称编码
        
        Args:
            scene_name: 场景名称 (如 "coding", "creative")
            device: 目标设备
        """
        scene_id = SCENE_IDS.get(scene_name.lower(), 0)
        device = device or self.embedding.weight.device
        return self.forward(torch.tensor([scene_id], device=device))
    
    def get_scene_name(self, scene_id: int) -> str:
        """获取场景名称"""
        return ID_TO_SCENE.get(scene_id, f"unknown_{scene_id}")
    
    @property
    def scene_names(self) -> list:
        """返回所有场景名称"""
        return list(SCENE_IDS.keys())


class SceneAwareProjection(nn.Module):
    """
    场景感知投影
    
    根据不同场景使用不同的投影权重。
    实现场景特定的处理逻辑。
    """
    
    def __init__(
        self,
        d_input: int,
        d_output: int,
        num_scenes: int = 8,
    ):
        """
        Args:
            d_input: 输入维度
            d_output: 输出维度
            num_scenes: 场景数量
        """
        super().__init__()
        
        # 每个场景一组投影权重
        self.projections = nn.ModuleList([
            nn.Linear(d_input, d_output)
            for _ in range(num_scenes)
        ])
        
        # 场景混合权重
        self.scene_encoder = SceneEncoder(d_scene=64, num_scenes=num_scenes)
        self.mix_proj = nn.Linear(64, num_scenes)
    
    def forward(
        self,
        x: torch.Tensor,
        scene_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        场景感知投影
        
        Args:
            x: 输入特征 [batch, d_input]
            scene_id: 场景 ID [batch]
            
        Returns:
            投影后的特征 [batch, d_output]
        """
        batch_size = x.shape[0]
        
        # 获取场景嵌入
        scene_emb = self.scene_encoder(scene_id)  # [batch, 64]
        mix_weights = torch.softmax(self.mix_proj(scene_emb), dim=-1)  # [batch, num_scenes]
        
        # 计算所有场景的投影
        outputs = torch.stack([
            proj(x) for proj in self.projections
        ], dim=1)  # [batch, num_scenes, d_output]
        
        # 加权混合
        output = torch.sum(
            outputs * mix_weights.unsqueeze(-1),
            dim=1
        )  # [batch, d_output]
        
        return output


# ============================================================
# AMHVQ+ 精度路由
# ============================================================

from dataclasses import dataclass
from typing import List
import re


@dataclass
class PrecisionConfig:
    """精度配置"""
    semantic: bool = True     # 语义通道 (始终激活)
    structure: bool = False   # 结构通道
    symbols: bool = False     # 符号通道
    
    def all_channels(self) -> bool:
        return self.semantic and self.structure and self.symbols
    
    def semantic_only(self) -> bool:
        return self.semantic and not self.structure and not self.symbols


# 场景到精度配置的映射
SCENE_PRECISION_CONFIG: Dict[str, PrecisionConfig] = {
    "chat": PrecisionConfig(semantic=True, structure=False, symbols=False),
    "coding": PrecisionConfig(semantic=True, structure=True, symbols=True),
    "debate": PrecisionConfig(semantic=True, structure=True, symbols=False),
    "creative": PrecisionConfig(semantic=True, structure=False, symbols=False),
    "analysis": PrecisionConfig(semantic=True, structure=True, symbols=False),
    "teaching": PrecisionConfig(semantic=True, structure=False, symbols=False),
    "roleplay": PrecisionConfig(semantic=True, structure=False, symbols=False),
    "formal": PrecisionConfig(semantic=True, structure=True, symbols=True),
}


class SceneAwarePrecisionRouter(nn.Module):
    """
    场景感知精度路由器
    
    根据场景决定激活哪些通道:
    - chat: 仅语义通道 (快速、灵活)
    - coding: 全部三通道 (精确)
    - creative: 仅语义通道 (自由发挥)
    - formal: 全部三通道 (精确)
    
    使用示例:
        router = SceneAwarePrecisionRouter()
        
        # 获取配置
        config = router.get_config("coding")
        print(config.structure)  # True
        
        # 自动检测
        text = "def hello(): pass"
        scene = router.detect_scene(text)
        config = router.get_config(scene)
    """
    
    def __init__(
        self,
        d_model: int = 768,
        use_auto_detection: bool = True,
    ):
        super().__init__()
        
        self.use_auto_detection = use_auto_detection
        
        # 场景检测器 (基于文本特征)
        if use_auto_detection:
            self.scene_classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, len(SCENE_IDS)),
            )
        else:
            self.scene_classifier = None
        
        # 代码检测模式
        self.code_patterns = [
            re.compile(r'def\s+\w+\s*\('),      # Python function
            re.compile(r'class\s+\w+'),          # Python class
            re.compile(r'import\s+\w+'),         # Import
            re.compile(r'function\s+\w+\s*\('),  # JS function
            re.compile(r'const\s+\w+\s*='),      # JS const
            re.compile(r'if\s*\(.*\)\s*\{'),     # JS/C if
            re.compile(r'#include\s*<'),         # C include
        ]
    
    def get_config(self, scene: str) -> PrecisionConfig:
        """获取场景的精度配置"""
        return SCENE_PRECISION_CONFIG.get(
            scene.lower(),
            PrecisionConfig()  # 默认仅语义
        )
    
    def detect_scene(self, text: str) -> str:
        """
        自动检测文本场景
        
        Args:
            text: 输入文本
            
        Returns:
            场景名称
        """
        # 规则检测: 代码
        for pattern in self.code_patterns:
            if pattern.search(text):
                return "coding"
        
        # 规则检测: 正式文档
        formal_indicators = ["Dear", "Sincerely", "To Whom", "Regards", "尊敬的", "此致"]
        for indicator in formal_indicators:
            if indicator in text:
                return "formal"
        
        # 默认: 聊天
        return "chat"
    
    def forward(
        self,
        text_or_hidden: Union[str, torch.Tensor],
        scene: Optional[str] = None,
    ) -> PrecisionConfig:
        """
        获取精度配置
        
        Args:
            text_or_hidden: 文本字符串或隐藏状态 [batch, seq_len, d_model]
            scene: 可选的场景指定
            
        Returns:
            PrecisionConfig
        """
        if scene is not None:
            return self.get_config(scene)
        
        if isinstance(text_or_hidden, str):
            detected = self.detect_scene(text_or_hidden)
            return self.get_config(detected)
        
        if self.scene_classifier is not None:
            # 使用神经网络分类
            hidden = text_or_hidden
            if hidden.dim() == 3:
                hidden = hidden.mean(dim=1)  # [batch, d_model]
            
            logits = self.scene_classifier(hidden)
            scene_id = logits.argmax(dim=-1).item()
            scene_name = ID_TO_SCENE.get(scene_id, "chat")
            return self.get_config(scene_name)
        
        return PrecisionConfig()
    
    def route(
        self,
        text: str,
        scene: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        路由决策
        
        Returns:
            {"semantic": True, "structure": True/False, "symbols": True/False}
        """
        config = self.forward(text, scene)
        return {
            "semantic": config.semantic,
            "structure": config.structure,
            "symbols": config.symbols,
        }
    
    @staticmethod
    def get_all_scenes() -> List[str]:
        """返回所有支持的场景"""
        return list(SCENE_IDS.keys())
    
    @staticmethod
    def get_precision_table() -> str:
        """返回场景精度配置表"""
        lines = ["Scene       | Semantic | Structure | Symbols"]
        lines.append("-" * 50)
        for scene, config in SCENE_PRECISION_CONFIG.items():
            s = "✓" if config.semantic else "✗"
            t = "✓" if config.structure else "✗"
            y = "✓" if config.symbols else "✗"
            lines.append(f"{scene:11} |    {s}     |     {t}     |    {y}")
        return "\n".join(lines)


# ============================================================
# 自动场景检测器
# ============================================================

class AutoSceneDetector(nn.Module):
    """
    自动场景检测器
    
    基于文本内容自动分类场景。
    """
    
    def __init__(
        self,
        d_model: int = 768,
        num_scenes: int = 8,
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_scenes),
        )
    
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        分类场景
        
        Args:
            hidden: [batch, seq_len, d_model] 或 [batch, d_model]
            
        Returns:
            scene_logits: [batch, num_scenes]
        """
        if hidden.dim() == 3:
            hidden = hidden.mean(dim=1)
        return self.classifier(hidden)
    
    def predict(self, hidden: torch.Tensor) -> List[str]:
        """预测场景名称"""
        logits = self.forward(hidden)
        ids = logits.argmax(dim=-1)
        return [ID_TO_SCENE.get(i.item(), "chat") for i in ids]
