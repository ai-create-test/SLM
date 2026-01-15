"""
Pipeline - 端到端推理流水线

整合所有模块的完整推理管道。
"""

from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import torch
import torch.nn as nn

from .interfaces.base_module import BaseModule, ModuleOutput, LatentVector
from .interfaces.config import Config, ModelConfig, PipelineConfig
from .io import ParagraphEncoder, ParagraphDecoder, SemanticSegmenter
from .brain import DynamicsModel, ACTController, ReasoningLoop
from .memory import LatentMemoryBank, QueryRetriever, CrossAttentionFuser
from .modulation import AdaptiveLayerNorm, EmotionEncoder, SceneEncoder
from .reflection import TrajectoryLogger, Backtracker, SelfCritic
from .search import WebSearch, KnowledgeInjector


@dataclass
class PipelineOutput:
    """流水线输出"""
    text: str                              # 生成的文本
    latent: LatentVector                   # 最终潜向量
    reasoning_steps: int                   # 推理步数
    memories_retrieved: int                # 检索的记忆数
    search_injected: int                   # 注入的搜索结果数
    metadata: Dict[str, Any] = None        # 额外元数据


class NeuralFlowPipeline(nn.Module):
    """
    NeuralFlow 端到端推理流水线
    
    完整的推理流程：
    1. 输入分割为段落
    2. 段落编码为潜向量
    3. 从记忆库检索相关历史
    4. 可选：联网搜索并注入
    5. 核心大脑进行推理 (Mamba + ACT)
    6. 情感/场景调制
    7. 解码生成输出
    
    使用示例:
        config = Config.from_preset("base")
        pipeline = NeuralFlowPipeline.from_config(config)
        
        output = pipeline.generate(
            "用户输入文本...",
            emotion="happy",
            scene="chat",
        )
        print(output.text)
    """
    
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()
        
        self.config = config
        
        # ========== IO 层 ==========
        self.segmenter = SemanticSegmenter(
            max_paragraph_len=config.model.max_paragraph_len,
        )
        self.encoder = ParagraphEncoder.from_config(config.model)
        self.decoder = ParagraphDecoder.from_config(config.model)
        
        # ========== 记忆系统 ==========
        if config.pipeline.enable_memory:
            self.memory_bank = LatentMemoryBank(d_latent=config.model.d_latent)
            self.retriever = QueryRetriever(
                d_latent=config.model.d_latent,
                d_query=config.model.d_model,
            )
            self.memory_fuser = CrossAttentionFuser(
                d_latent=config.model.d_latent,
            )
        else:
            self.memory_bank = None
            self.retriever = None
            self.memory_fuser = None
        
        # ========== 核心大脑 ==========
        self.reasoning_loop = ReasoningLoop.from_config(config.model)
        
        # ========== 调制层 ==========
        if config.pipeline.enable_emotion:
            self.emotion_encoder = EmotionEncoder(d_emotion=config.model.d_emotion)
        else:
            self.emotion_encoder = None
        
        if config.pipeline.enable_scene:
            self.scene_encoder = SceneEncoder(d_scene=config.model.d_scene)
        else:
            self.scene_encoder = None
        
        # ========== 自省模块 ==========
        if config.pipeline.enable_reflection:
            self.trajectory_logger = TrajectoryLogger()
            self.backtracker = Backtracker(
                d_model=config.model.d_model,
                max_backtracks=config.pipeline.max_backtracks,
            )
            self.self_critic = SelfCritic(d_latent=config.model.d_latent)
        else:
            self.trajectory_logger = None
            self.backtracker = None
            self.self_critic = None
        
        # ========== 搜索模块 ==========
        if config.pipeline.enable_search:
            self.search_engine = WebSearch(
                provider=config.pipeline.search_provider,
                api_key=config.pipeline.search_api_key,
            )
            self.knowledge_injector = KnowledgeInjector(
                encoder=self.encoder,
                memory_bank=self.memory_bank,
            ) if self.memory_bank else None
        else:
            self.search_engine = None
            self.knowledge_injector = None
    
    def forward(
        self,
        text: str,
        emotion: Optional[Union[str, int]] = None,
        scene: Optional[Union[str, int]] = None,
        **kwargs,
    ) -> PipelineOutput:
        """
        端到端推理
        
        Args:
            text: 输入文本
            emotion: 情感状态
            scene: 场景模式
            **kwargs: 额外参数
            
        Returns:
            PipelineOutput
        """
        return self.generate(text, emotion=emotion, scene=scene, **kwargs)
    
    def generate(
        self,
        text: str,
        emotion: Optional[Union[str, int]] = None,
        scene: Optional[Union[str, int]] = None,
        enable_search: bool = None,
        max_length: int = None,
    ) -> PipelineOutput:
        """
        生成响应
        """
        # 1. 分割为段落
        paragraphs = self.segmenter.segment(text)
        
        # 2. 编码所有段落
        latent_history = []
        for para in paragraphs:
            output = self.encoder(para.content)
            latent_history.append(output.latent)
            
            # 存入记忆
            if self.memory_bank is not None:
                self.memory_bank.add(output.latent.vector, content=para.content)
        
        # 3. 准备情感/场景向量
        emotion_vec = None
        scene_vec = None
        
        if self.emotion_encoder is not None and emotion is not None:
            if isinstance(emotion, str):
                emotion_vec = self.emotion_encoder.encode_name(emotion)
            else:
                emotion_vec = self.emotion_encoder(emotion)
        
        if self.scene_encoder is not None and scene is not None:
            if isinstance(scene, str):
                scene_vec = self.scene_encoder.encode_name(scene)
            else:
                scene_vec = self.scene_encoder(scene)
        
        # 4. 检索记忆上下文
        memory_context = None
        memories_retrieved = 0
        
        if self.memory_bank is not None and self.memory_fuser is not None and latent_history:
            query = latent_history[-1].vector
            retrieved_items = self.memory_bank.retrieve(query, k=self.config.model.memory_top_k)
            memories_retrieved = len(retrieved_items)
            
            if retrieved_items:
                memory_context = self.memory_fuser.forward_from_items(
                    query.unsqueeze(0),
                    retrieved_items,
                ).squeeze(0)
        
        # 5. 可选：联网搜索
        search_injected = 0
        if (enable_search or (enable_search is None and self.config.pipeline.enable_search)):
            if self.search_engine is not None and self.knowledge_injector is not None:
                # 使用最后一个段落内容作为搜索查询
                query_text = paragraphs[-1].content if paragraphs else text
                # 注意：这里简化处理，实际应该是异步
                # search_injected = await self.knowledge_injector.inject_from_query(
                #     self.search_engine, query_text
                # )
        
        # 6. 核心推理
        if not latent_history:
            # 空输入，返回默认
            return PipelineOutput(
                text="",
                latent=LatentVector(vector=torch.zeros(self.config.model.d_latent)),
                reasoning_steps=0,
                memories_retrieved=0,
                search_injected=0,
            )
        
        z_history = torch.stack([l.vector for l in latent_history], dim=0).unsqueeze(0)
        
        reasoning_output = self.reasoning_loop(
            z_history=z_history,
            memory_context=memory_context.unsqueeze(0) if memory_context is not None else None,
            emotion=emotion_vec,
            scene=scene_vec,
        )
        
        # 7. 解码
        decoder_output = self.decoder.generate(
            latent=reasoning_output.predicted_latent.vector,
            max_length=max_length or self.config.model.max_paragraph_len,
            emotion=emotion_vec,
            scene=scene_vec,
        )
        
        output_text = decoder_output.text[0] if decoder_output.text else ""
        
        return PipelineOutput(
            text=output_text,
            latent=reasoning_output.predicted_latent,
            reasoning_steps=reasoning_output.reasoning_steps,
            memories_retrieved=memories_retrieved,
            search_injected=search_injected,
        )
    
    @classmethod
    def from_config(cls, config: Config) -> "NeuralFlowPipeline":
        """从配置创建实例"""
        return cls(config)
    
    @classmethod
    def from_preset(cls, preset: str = "base") -> "NeuralFlowPipeline":
        """从预设创建实例"""
        config = Config.from_preset(preset)
        return cls(config)
    
    def save(self, path: str) -> None:
        """保存模型"""
        import torch
        torch.save({
            "state_dict": self.state_dict(),
            "config": self.config.to_dict(),
        }, path)
        
        # 保存记忆库
        if self.memory_bank is not None:
            self.memory_bank.save(path + "_memory")
    
    @classmethod
    def load(cls, path: str) -> "NeuralFlowPipeline":
        """加载模型"""
        import torch
        checkpoint = torch.load(path)
        
        config = Config.load(checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        
        return model
