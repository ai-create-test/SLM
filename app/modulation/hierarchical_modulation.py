"""
Hierarchical Modulation - 层次化情感/场景调制

Phase 14: Emotion/Modulation 适配

支持对 HierarchicalLatent 的层次化调制:
- Global 级别调制
- Chunk 级别调制
- Detail 级别调制
"""

from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces.base_module import BaseModule, ModuleOutput, HierarchicalLatent
from ..interfaces.unified_latent import UnifiedLatent


@dataclass
class ModulationOutput(ModuleOutput):
    """调制输出"""
    modulated_latent: HierarchicalLatent = None
    global_mod: Optional[torch.Tensor] = None
    chunk_mod: Optional[torch.Tensor] = None


class HierarchicalEmotionModulator(nn.Module):
    """
    层次化情感调制器
    
    对 HierarchicalLatent 的每个层次进行情感调制。
    """
    
    def __init__(
        self,
        d_latent: int = 512,
        d_emotion: int = 64,
        num_emotions: int = 8,
    ):
        super().__init__()
        
        self.d_latent = d_latent
        self.d_emotion = d_emotion
        
        # 情感嵌入
        self.emotion_embedding = nn.Embedding(num_emotions, d_emotion)
        
        # Global 调制 (强调整体情感)
        self.global_modulator = nn.Sequential(
            nn.Linear(d_latent + d_emotion, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
        )
        
        # Chunk 调制 (情感分布到各块)
        self.chunk_modulator = nn.Sequential(
            nn.Linear(d_latent + d_emotion, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
        )
        
        # 可学习的调制强度
        self.global_gate = nn.Parameter(torch.zeros(1))
        self.chunk_gate = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        hierarchical: HierarchicalLatent,
        emotion: Union[int, torch.Tensor],
    ) -> HierarchicalLatent:
        """
        应用层次化情感调制
        
        Args:
            hierarchical: HierarchicalLatent
            emotion: 情感索引或嵌入向量
            
        Returns:
            调制后的 HierarchicalLatent
        """
        batch_size = hierarchical.batch_size
        device = hierarchical.device
        
        # 获取情感嵌入
        if isinstance(emotion, int):
            emotion_idx = torch.tensor([emotion], device=device).expand(batch_size)
            emotion_embed = self.emotion_embedding(emotion_idx)  # [batch, d_emotion]
        elif emotion.dim() == 1:
            emotion_embed = self.emotion_embedding(emotion)
        else:
            emotion_embed = emotion  # 假设已经是嵌入
        
        # Global 调制
        global_vec = hierarchical.global_.squeeze(1)  # [batch, d_latent]
        global_input = torch.cat([global_vec, emotion_embed], dim=-1)
        global_mod = self.global_modulator(global_input)
        modulated_global = global_vec + torch.sigmoid(self.global_gate) * global_mod
        
        # Chunk 调制
        chunks = hierarchical.chunks  # [batch, num_chunks, d_latent]
        num_chunks = chunks.shape[1]
        emotion_expanded = emotion_embed.unsqueeze(1).expand(-1, num_chunks, -1)
        chunk_input = torch.cat([chunks, emotion_expanded], dim=-1)
        chunk_mod = self.chunk_modulator(chunk_input.reshape(-1, self.d_latent + self.d_emotion))
        chunk_mod = chunk_mod.reshape(batch_size, num_chunks, self.d_latent)
        modulated_chunks = chunks + torch.sigmoid(self.chunk_gate) * chunk_mod
        
        return HierarchicalLatent(
            global_=modulated_global.unsqueeze(1),
            chunks=modulated_chunks,
            detail=hierarchical.detail,
            indices=hierarchical.indices,
            chunk_mask=hierarchical.chunk_mask,
        )


class HierarchicalSceneModulator(nn.Module):
    """
    层次化场景调制器
    
    根据场景调整 HierarchicalLatent 的表示。
    """
    
    SCENE_TO_ID = {
        "chat": 0,
        "coding": 1,
        "technical": 2,
        "creative": 3,
        "formal": 4,
    }
    
    def __init__(
        self,
        d_latent: int = 512,
        d_scene: int = 64,
        num_scenes: int = 8,
    ):
        super().__init__()
        
        self.d_latent = d_latent
        self.d_scene = d_scene
        
        # 场景嵌入
        self.scene_embedding = nn.Embedding(num_scenes, d_scene)
        
        # 场景调制
        self.scene_modulator = nn.Sequential(
            nn.Linear(d_latent + d_scene, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
        )
        
        # 调制强度
        self.scene_gate = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        hierarchical: HierarchicalLatent,
        scene: Union[str, int, torch.Tensor],
    ) -> HierarchicalLatent:
        """
        应用场景调制
        
        Args:
            hierarchical: HierarchicalLatent
            scene: 场景名称/索引/嵌入
            
        Returns:
            调制后的 HierarchicalLatent
        """
        batch_size = hierarchical.batch_size
        device = hierarchical.device
        
        # 获取场景嵌入
        if isinstance(scene, str):
            scene_id = self.SCENE_TO_ID.get(scene, 0)
            scene_idx = torch.tensor([scene_id], device=device).expand(batch_size)
            scene_embed = self.scene_embedding(scene_idx)
        elif isinstance(scene, int):
            scene_idx = torch.tensor([scene], device=device).expand(batch_size)
            scene_embed = self.scene_embedding(scene_idx)
        else:
            scene_embed = scene
        
        # 调制 global
        global_vec = hierarchical.global_.squeeze(1)
        global_input = torch.cat([global_vec, scene_embed], dim=-1)
        global_mod = self.scene_modulator(global_input)
        modulated_global = global_vec + torch.sigmoid(self.scene_gate) * global_mod
        
        return HierarchicalLatent(
            global_=modulated_global.unsqueeze(1),
            chunks=hierarchical.chunks,
            detail=hierarchical.detail,
            indices=hierarchical.indices,
            chunk_mask=hierarchical.chunk_mask,
        )


class UnifiedModulator(nn.Module):
    """
    统一调制器
    
    组合情感和场景调制。
    """
    
    def __init__(
        self,
        d_latent: int = 512,
        d_emotion: int = 64,
        d_scene: int = 64,
        num_emotions: int = 8,
        num_scenes: int = 8,
    ):
        super().__init__()
        
        self.emotion_modulator = HierarchicalEmotionModulator(
            d_latent=d_latent,
            d_emotion=d_emotion,
            num_emotions=num_emotions,
        )
        
        self.scene_modulator = HierarchicalSceneModulator(
            d_latent=d_latent,
            d_scene=d_scene,
            num_scenes=num_scenes,
        )
    
    def forward(
        self,
        latent: Union[HierarchicalLatent, UnifiedLatent],
        emotion: Optional[Union[int, str, torch.Tensor]] = None,
        scene: Optional[Union[int, str, torch.Tensor]] = None,
    ) -> Union[HierarchicalLatent, UnifiedLatent]:
        """
        应用统一调制
        
        Args:
            latent: HierarchicalLatent 或 UnifiedLatent
            emotion: 情感
            scene: 场景
            
        Returns:
            调制后的潜向量
        """
        # 提取 HierarchicalLatent
        if isinstance(latent, UnifiedLatent):
            hierarchical = latent.semantic
            is_unified = True
        else:
            hierarchical = latent
            is_unified = False
        
        # 应用情感调制
        if emotion is not None:
            hierarchical = self.emotion_modulator(hierarchical, emotion)
        
        # 应用场景调制
        if scene is not None:
            hierarchical = self.scene_modulator(hierarchical, scene)
        
        # 返回
        if is_unified:
            return UnifiedLatent(
                semantic=hierarchical,
                structure=latent.structure,
                symbols=latent.symbols,
                scene=latent.scene,
                precision_config=latent.precision_config,
            )
        else:
            return hierarchical
