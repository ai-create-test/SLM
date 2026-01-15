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
