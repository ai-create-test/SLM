"""
Config Loader - 增强配置加载器

功能:
- 多层配置覆盖 (CLI > 环境变量 > 用户配置 > 基础配置 > 默认值)
- 配置继承 (inherit 关键字)
- 环境变量插值 (${VAR_NAME})
- 配置验证
"""

import os
import re
import json
import copy
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict

import yaml

from .config import Config, ModelConfig, TrainingConfig, PipelineConfig


# ============================================================================
# Cloud Config
# ============================================================================

@dataclass
class CloudConfig:
    """
    云训练配置
    
    支持 RunPod, Modal, Lambda Labs
    """
    # 平台选择
    provider: str = "runpod"                    # runpod | modal | lambda
    
    # 计算资源
    gpu_type: str = "RTX4090"
    gpu_count: int = 1
    cpu_count: int = 4
    memory_gb: int = 32
    disk_gb: int = 50
    region: Optional[str] = None
    
    # 认证 (从环境变量)
    api_key: Optional[str] = None
    
    # 任务设置
    max_hours: float = 8.0
    auto_terminate: bool = True
    
    # 容器环境
    docker_image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel"
    pip_requirements: List[str] = field(default_factory=lambda: [
        "safetensors",
        "pyyaml",
    ])
    
    # 数据同步
    data_source: str = "local"                  # local | s3 | huggingface
    output_destination: str = "local"
    sync_checkpoints: bool = True
    sync_interval_minutes: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CloudConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# Extended Config
# ============================================================================

@dataclass
class ExtendedConfig(Config):
    """
    扩展配置，增加云训练支持
    """
    cloud: CloudConfig = field(default_factory=CloudConfig)
    
    # 实验元信息
    experiment_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["cloud"] = self.cloud.to_dict()
        data["experiment_name"] = self.experiment_name
        data["tags"] = self.tags
        data["notes"] = self.notes
        return data


# ============================================================================
# Config Loader
# ============================================================================

class ConfigLoader:
    """
    增强配置加载器
    
    支持:
    - 配置继承
    - 环境变量插值 ${VAR}
    - CLI 参数覆盖
    - 多文件合并
    
    使用示例:
        # 基本加载
        config = ConfigLoader.load("configs/base.yaml")
        
        # 带 CLI 覆盖
        config = ConfigLoader.load(
            "configs/base.yaml",
            cli_overrides={"model.d_latent": 256, "training.batch_size": 16}
        )
        
        # 带环境变量
        config = ConfigLoader.load("configs/cloud.yaml")  # ${API_KEY} 自动替换
    """
    
    ENV_PREFIX = "NF_"  # 环境变量前缀
    CONFIG_BASE_DIR = Path("configs")
    
    @classmethod
    def load(
        cls,
        config_path: str,
        cli_overrides: Optional[Dict[str, Any]] = None,
        use_env: bool = True,
    ) -> ExtendedConfig:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            cli_overrides: CLI 覆盖参数 (格式: {"model.d_latent": 256})
            use_env: 是否使用环境变量覆盖
            
        Returns:
            ExtendedConfig
        """
        # 1. 加载原始配置
        raw_config = cls._load_yaml(config_path)
        
        # 2. 处理继承
        if "inherit" in raw_config:
            parent_name = raw_config.pop("inherit")
            parent_path = cls._resolve_inherit_path(parent_name, config_path)
            parent_config = cls._load_yaml(parent_path)
            raw_config = cls._deep_merge(parent_config, raw_config)
        
        # 3. 环境变量插值
        raw_config = cls._interpolate_env(raw_config)
        
        # 4. 应用环境变量覆盖
        if use_env:
            raw_config = cls._apply_env_overrides(raw_config)
        
        # 5. 应用 CLI 覆盖
        if cli_overrides:
            raw_config = cls._apply_cli_overrides(raw_config, cli_overrides)
        
        # 6. 构建 Config 对象
        config = cls._build_config(raw_config)
        
        return config
    
    @classmethod
    def load_with_args(
        cls,
        config_path: str,
        args,  # argparse.Namespace
    ) -> ExtendedConfig:
        """
        从 argparse 参数加载配置
        
        Args:
            config_path: 配置文件路径
            args: argparse.Namespace
            
        Returns:
            ExtendedConfig
        """
        # 提取 CLI 覆盖
        cli_overrides = {}
        
        for key, value in vars(args).items():
            if value is not None and "." in key:
                cli_overrides[key] = value
        
        return cls.load(config_path, cli_overrides=cli_overrides)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExtendedConfig:
        """从字典创建配置"""
        return cls._build_config(data)
    
    # =========================================================================
    # 内部方法
    # =========================================================================
    
    @classmethod
    def _load_yaml(cls, path: str) -> Dict[str, Any]:
        """加载 YAML 文件"""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        
        return data
    
    @classmethod
    def _resolve_inherit_path(cls, parent_name: str, child_path: str) -> str:
        """解析继承路径"""
        child_path = Path(child_path)
        
        # 相对于 configs 目录
        if not parent_name.endswith((".yaml", ".yml")):
            parent_name = f"{parent_name}.yaml"
        
        # 先检查同目录
        same_dir = child_path.parent / parent_name
        if same_dir.exists():
            return str(same_dir)
        
        # 检查 configs 目录
        configs_dir = cls.CONFIG_BASE_DIR / parent_name
        if configs_dir.exists():
            return str(configs_dir)
        
        # 检查 presets 目录
        presets_dir = cls.CONFIG_BASE_DIR / "presets" / parent_name
        if presets_dir.exists():
            return str(presets_dir)
        
        raise FileNotFoundError(f"Parent config not found: {parent_name}")
    
    @classmethod
    def _deep_merge(cls, base: Dict, override: Dict) -> Dict:
        """深度合并字典"""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = cls._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    @classmethod
    def _interpolate_env(cls, data: Any) -> Any:
        """
        环境变量插值
        
        ${VAR_NAME} -> os.environ.get("VAR_NAME")
        ${VAR_NAME:default} -> os.environ.get("VAR_NAME", "default")
        """
        if isinstance(data, str):
            pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
            
            def replacer(match):
                var_name = match.group(1)
                default = match.group(2)
                value = os.environ.get(var_name)
                if value is None:
                    return default if default is not None else ""
                return value
            
            return re.sub(pattern, replacer, data)
        
        elif isinstance(data, dict):
            return {k: cls._interpolate_env(v) for k, v in data.items()}
        
        elif isinstance(data, list):
            return [cls._interpolate_env(v) for v in data]
        
        return data
    
    @classmethod
    def _apply_env_overrides(cls, data: Dict) -> Dict:
        """
        应用环境变量覆盖
        
        NF_MODEL_D_LATENT=256 -> data["model"]["d_latent"] = 256
        """
        result = copy.deepcopy(data)
        prefix = cls.ENV_PREFIX
        
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            
            # NF_MODEL_D_LATENT -> model.d_latent
            path = key[len(prefix):].lower().replace("_", ".")
            parts = path.split(".")
            
            # 处理嵌套 key
            if len(parts) >= 2:
                section = parts[0]
                field = "_".join(parts[1:])
                
                if section not in result:
                    result[section] = {}
                
                # 类型转换
                result[section][field] = cls._parse_value(value)
        
        return result
    
    @classmethod
    def _apply_cli_overrides(cls, data: Dict, overrides: Dict[str, Any]) -> Dict:
        """
        应用 CLI 覆盖
        
        {"model.d_latent": 256} -> data["model"]["d_latent"] = 256
        """
        result = copy.deepcopy(data)
        
        for key, value in overrides.items():
            parts = key.split(".")
            
            # 导航到目标位置
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            current[parts[-1]] = value
        
        return result
    
    @staticmethod
    def _parse_value(value: str) -> Any:
        """解析字符串值为适当类型"""
        # 布尔
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
        
        # 整数
        try:
            return int(value)
        except ValueError:
            pass
        
        # 浮点数
        try:
            return float(value)
        except ValueError:
            pass
        
        # JSON (列表/字典)
        if value.startswith(("[", "{")):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        return value
    
    @classmethod
    def _build_config(cls, data: Dict) -> ExtendedConfig:
        """从字典构建 ExtendedConfig"""
        model_data = data.get("model", {})
        training_data = data.get("training", {})
        pipeline_data = data.get("pipeline", {})
        cloud_data = data.get("cloud", {})
        
        # 处理 loss_weights 特殊字段
        if "loss_weights" in training_data and isinstance(training_data["loss_weights"], dict):
            pass  # 保持原样
        
        return ExtendedConfig(
            model=ModelConfig(**{k: v for k, v in model_data.items() 
                                if k in ModelConfig.__dataclass_fields__}),
            training=TrainingConfig(**{k: v for k, v in training_data.items() 
                                      if k in TrainingConfig.__dataclass_fields__}),
            pipeline=PipelineConfig(**{k: v for k, v in pipeline_data.items() 
                                      if k in PipelineConfig.__dataclass_fields__}),
            cloud=CloudConfig.from_dict(cloud_data),
            name=data.get("name", "neuralflow"),
            version=data.get("version", "0.1.0"),
            description=data.get("description", ""),
            experiment_name=data.get("experiment_name"),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
        )


# ============================================================================
# CLI 参数解析帮助
# ============================================================================

def add_config_args(parser):
    """
    添加配置相关的 CLI 参数
    
    使用:
        parser = argparse.ArgumentParser()
        add_config_args(parser)
        args = parser.parse_args()
    """
    # 配置文件
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Config file path",
    )
    parser.add_argument(
        "--preset", "-p",
        type=str,
        choices=["tiny", "small", "base", "large"],
        default=None,
        help="Config preset",
    )
    
    # 模型配置覆盖
    parser.add_argument("--model.d_latent", type=int, help="Latent dimension")
    parser.add_argument("--model.d_model", type=int, help="Model dimension")
    parser.add_argument("--model.num_layers", type=int, help="Number of layers")
    parser.add_argument("--model.brain_type", type=str, help="Brain type (mamba/gru)")
    
    # 训练配置覆盖
    parser.add_argument("--training.batch_size", type=int, help="Batch size")
    parser.add_argument("--training.learning_rate", type=float, help="Learning rate")
    parser.add_argument("--training.max_epochs", type=int, help="Max epochs")
    
    # 云配置覆盖
    parser.add_argument("--cloud.provider", type=str, help="Cloud provider")
    parser.add_argument("--cloud.gpu_type", type=str, help="GPU type")
    parser.add_argument("--cloud.gpu_count", type=int, help="GPU count")
    
    return parser


def parse_config_from_args(args) -> ExtendedConfig:
    """
    从 argparse 结果解析配置
    
    Args:
        args: argparse.Namespace
        
    Returns:
        ExtendedConfig
    """
    # 收集 CLI 覆盖
    cli_overrides = {}
    for key in ["model.d_latent", "model.d_model", "model.num_layers", "model.brain_type",
                "training.batch_size", "training.learning_rate", "training.max_epochs",
                "cloud.provider", "cloud.gpu_type", "cloud.gpu_count"]:
        # argparse 用下划线替换点
        arg_key = key.replace(".", "_")
        if hasattr(args, arg_key):
            value = getattr(args, arg_key)
            if value is not None:
                cli_overrides[key] = value
    
    # 加载配置
    if args.config:
        return ConfigLoader.load(args.config, cli_overrides=cli_overrides)
    elif args.preset:
        preset_path = f"configs/presets/{args.preset}.yaml"
        if Path(preset_path).exists():
            return ConfigLoader.load(preset_path, cli_overrides=cli_overrides)
        else:
            # 使用内置预设
            base_config = Config.from_preset(args.preset)
            return ExtendedConfig(
                model=base_config.model,
                training=base_config.training,
                pipeline=base_config.pipeline,
            )
    else:
        # 默认配置
        return ExtendedConfig()
