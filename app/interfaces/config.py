"""
Config - 全局配置系统

设计原则:
1. 分层配置 (Model / Training / Pipeline)
2. 支持 YAML/JSON 序列化
3. 类型安全
4. 默认值 + 覆盖机制
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Literal
import json
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """
    模型架构配置
    """
    # ============ 维度配置 ============
    d_latent: int = 512              # 潜空间维度
    d_model: int = 768               # 内部隐藏维度
    d_state: int = 64                # SSM 状态维度 (Mamba)
    d_emotion: int = 128             # 情感嵌入维度
    d_scene: int = 128               # 场景嵌入维度
    
    # ============ VQ-VAE 配置 ============
    codebook_size: int = 8192        # VQ 码本大小
    num_codebooks: int = 4           # 多头 VQ 数量
    commitment_cost: float = 0.25    # VQ 承诺损失系数
    
    # ============ 记忆系统配置 ============
    memory_size: int = 10000         # 最大记忆条目数
    memory_top_k: int = 5            # 检索 Top-K
    memory_index_type: str = "IVF"   # FAISS 索引类型
    
    # ============ 核心大脑配置 ============
    brain_type: Literal["mamba", "gru", "transformer"] = "mamba"
    num_layers: int = 6              # 核心层数
    expand: int = 2                  # Mamba 扩展因子
    d_conv: int = 4                  # Mamba 卷积核大小
    
    # ============ ACT 配置 ============
    max_think_steps: int = 10        # 最大思考步数
    halt_threshold: float = 0.99     # 停止阈值
    ponder_cost: float = 0.01        # 思考代价 (正则化)
    
    # ============ 调制配置 ============
    modulation_type: Literal["adaln", "film", "both"] = "adaln"
    num_emotions: int = 6            # 情感类别数
    num_scenes: int = 5              # 场景类别数
    
    # ============ 其他配置 ============
    vocab_size: int = 100000         # 词表大小 (解码器用)
    max_paragraph_len: int = 256     # 最大段落长度
    dropout: float = 0.1             # Dropout 率
    rope_theta: float = 10000.0      # RoPE 基础频率


@dataclass
class TrainingConfig:
    """
    训练配置
    """
    # ============ 基础训练参数 ============
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 100
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    
    # ============ 优化器配置 ============
    optimizer: str = "adamw"
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    
    # ============ 学习率调度 ============
    scheduler: str = "cosine"
    min_lr: float = 1e-6
    
    # ============ 损失权重 ============
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "reconstruction": 1.0,       # VQ-VAE 重建损失
        "commitment": 0.25,          # VQ 承诺损失
        "prediction": 1.0,           # 潜空间预测损失
        "ponder": 0.01,              # 思考代价
        "contrastive": 0.1,          # 对比学习损失
    })
    
    # ============ 保存与日志 ============
    save_every: int = 1000
    log_every: int = 100
    eval_every: int = 500
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # ============ 数据配置 ============
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2


@dataclass
class PipelineConfig:
    """
    端到端流水线配置
    """
    # ============ 模块启用 ============
    enable_memory: bool = True
    enable_search: bool = False
    enable_reflection: bool = False
    enable_emotion: bool = True
    enable_scene: bool = True
    
    # ============ 搜索配置 ============
    search_provider: str = "duckduckgo"
    search_api_key: Optional[str] = None
    search_cache_ttl: int = 3600
    max_search_results: int = 5
    
    # ============ 回溯配置 ============
    enable_backtrack: bool = False
    backtrack_threshold: float = 0.3
    max_backtracks: int = 3
    
    # ============ 推理配置 ============
    inference_mode: Literal["greedy", "sampling", "beam"] = "sampling"
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


@dataclass
class Config:
    """
    顶层配置容器
    
    整合所有子配置。
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    
    # ============ 元信息 ============
    name: str = "neuralflow"
    version: str = "0.1.0"
    description: str = "Next-generation LLM with paragraph-level reasoning"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def save(self, path: str) -> None:
        """保存配置到文件"""
        path = Path(path)
        data = self.to_dict()
        
        if path.suffix == ".yaml" or path.suffix == ".yml":
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> "Config":
        """从文件加载配置"""
        path = Path(path)
        
        if path.suffix == ".yaml" or path.suffix == ".yml":
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        
        return cls(
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            pipeline=PipelineConfig(**data.get("pipeline", {})),
            name=data.get("name", "neuralflow"),
            version=data.get("version", "0.1.0"),
            description=data.get("description", ""),
        )
    
    @classmethod
    def from_preset(cls, preset: str) -> "Config":
        """
        从预设创建配置
        
        Args:
            preset: 预设名称 ('small', 'base', 'large')
        """
        presets = {
            "small": {
                "model": {
                    "d_latent": 256,
                    "d_model": 512,
                    "num_layers": 4,
                    "codebook_size": 4096,
                },
                "training": {
                    "batch_size": 64,
                    "learning_rate": 2e-4,
                },
            },
            "base": {
                "model": {
                    "d_latent": 512,
                    "d_model": 768,
                    "num_layers": 6,
                    "codebook_size": 8192,
                },
                "training": {
                    "batch_size": 32,
                    "learning_rate": 1e-4,
                },
            },
            "large": {
                "model": {
                    "d_latent": 768,
                    "d_model": 1024,
                    "num_layers": 12,
                    "codebook_size": 16384,
                    "max_think_steps": 16,
                },
                "training": {
                    "batch_size": 16,
                    "learning_rate": 5e-5,
                },
            },
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        
        preset_data = presets[preset]
        return cls(
            model=ModelConfig(**preset_data.get("model", {})),
            training=TrainingConfig(**preset_data.get("training", {})),
        )


# 默认配置实例
DEFAULT_CONFIG = Config()
