"""
Unified Trainer - 统一训练管理器

管理完整的四阶段训练流程:
- Stage 1: VQ-VAE 预训练
- Stage 2: Dynamics 预训练
- Stage 3: Emotion 联合训练
- Stage 4: 端到端微调
"""

from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ..interfaces.config import Config, TrainingConfig
from ..model.neuralflow_model import NeuralFlowModel
from ..model.model_utils import ModelDirectory, save_training_state, load_training_state
from .training_stages import VQVAEStage, DynamicsStage, EmotionStage, FinetuneStage, StageResult
from .data_loader import ParagraphDataset, SequenceDataset, ParagraphDataLoader


@dataclass
class TrainingResult:
    """训练结果"""
    completed_stages: List[str]
    stage_results: Dict[str, StageResult]
    total_steps: int
    total_time: float
    final_model_path: Optional[str] = None


class UnifiedTrainer:
    """
    统一训练管理器
    
    管理完整的多阶段训练流程。
    
    使用示例:
        model = NeuralFlowModel.from_preset("base")
        trainer = UnifiedTrainer(model, config)
        
        # 完整训练
        result = trainer.train(
            train_data=dataset,
            stages=["vqvae", "dynamics", "emotion"],
        )
        
        # 只训练某个阶段
        result = trainer.train(
            train_data=dataset,
            stages=["dynamics"],
        )
        
        # 恢复训练
        result = trainer.train(
            train_data=dataset,
            resume_from="checkpoints/",
        )
    """
    
    # 阶段训练器映射
    STAGE_TRAINERS = {
        "vqvae": VQVAEStage,
        "dynamics": DynamicsStage,
        "emotion": EmotionStage,
        "finetune": FinetuneStage,
    }
    
    # 阶段默认配置
    STAGE_DEFAULTS = {
        "vqvae": {
            "epochs": 20,
            "batch_size": 32,
            "learning_rate": 1e-4,
            "beta_commit": 0.25,
        },
        "dynamics": {
            "epochs": 50,
            "batch_size": 16,
            "learning_rate": 5e-5,
            "seq_len": 5,
        },
        "emotion": {
            "epochs": 30,
            "batch_size": 16,
            "learning_rate": 1e-4,
        },
        "finetune": {
            "epochs": 10,
            "batch_size": 8,
            "learning_rate": 1e-5,
        },
    }
    
    def __init__(
        self,
        model: NeuralFlowModel,
        config: Optional[Config] = None,
        device: str = "auto",
        output_dir: str = "./outputs",
        log_interval: int = 10,
        save_interval: int = 1000,
    ):
        """
        Args:
            model: NeuralFlowModel 实例
            config: 配置 (如果为 None 则使用 model.config)
            device: 设备 ("auto", "cpu", "cuda", "cuda:0")
            output_dir: 输出目录
            log_interval: 日志间隔
            save_interval: 保存间隔
        """
        self.model = model
        self.config = config or model.config
        self.output_dir = Path(output_dir)
        self.log_interval = log_interval
        self.save_interval = save_interval
        
        # 设备
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        
        # 训练状态
        self.global_step = 0
        self.completed_stages: List[str] = list(model.completed_stages)
    
    def train(
        self,
        train_data: Union[ParagraphDataset, Dataset],
        eval_data: Optional[Dataset] = None,
        stages: List[str] = ["vqvae", "dynamics"],
        stage_configs: Optional[Dict[str, Dict]] = None,
        resume_from: Optional[str] = None,
        callback: Optional[Callable] = None,
    ) -> TrainingResult:
        """
        执行训练
        
        Args:
            train_data: 训练数据集
            eval_data: 验证数据集
            stages: 要执行的阶段列表
            stage_configs: 阶段特定配置覆盖
            resume_from: 恢复训练的检查点路径
            callback: 训练回调函数
            
        Returns:
            TrainingResult
        """
        start_time = time.time()
        stage_configs = stage_configs or {}
        stage_results: Dict[str, StageResult] = {}
        
        # 恢复训练
        if resume_from:
            self._load_checkpoint(resume_from)
        
        print(f"Starting training on device: {self.device}")
        print(f"Stages to train: {stages}")
        print(f"Already completed: {self.completed_stages}")
        
        for stage_name in stages:
            if stage_name in self.completed_stages:
                print(f"Skipping {stage_name} (already completed)")
                continue
            
            print(f"\n{'='*60}")
            print(f"Stage: {stage_name.upper()}")
            print(f"{'='*60}")
            
            result = self._run_stage(
                stage_name=stage_name,
                train_data=train_data,
                eval_data=eval_data,
                config=stage_configs.get(stage_name, {}),
                callback=callback,
            )
            
            stage_results[stage_name] = result
            self.completed_stages.append(stage_name)
            
            # 更新模型元数据
            self.model.update_training_metadata(
                stage=stage_name,
                steps=result.total_steps,
                loss=result.final_loss,
            )
            
            # 保存检查点
            self._save_checkpoint(f"stage_{stage_name}")
        
        # 保存最终模型
        final_path = str(self.output_dir / "final")
        self.model.save_pretrained(final_path)
        
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"Model saved to {final_path}")
        
        return TrainingResult(
            completed_stages=self.completed_stages,
            stage_results=stage_results,
            total_steps=self.global_step,
            total_time=total_time,
            final_model_path=final_path,
        )
    
    def _run_stage(
        self,
        stage_name: str,
        train_data: Dataset,
        eval_data: Optional[Dataset],
        config: Dict[str, Any],
        callback: Optional[Callable],
    ) -> StageResult:
        """运行单个训练阶段"""
        if stage_name not in self.STAGE_TRAINERS:
            raise ValueError(f"Unknown stage: {stage_name}")
        
        # 合并配置
        stage_config = {**self.STAGE_DEFAULTS.get(stage_name, {}), **config}
        
        # 准备数据加载器
        train_loader = self._prepare_dataloader(
            dataset=train_data,
            stage=stage_name,
            batch_size=stage_config.get("batch_size", 16),
            shuffle=True,
        )
        
        eval_loader = None
        if eval_data:
            eval_loader = self._prepare_dataloader(
                dataset=eval_data,
                stage=stage_name,
                batch_size=stage_config.get("batch_size", 16),
                shuffle=False,
            )
        
        # 创建阶段训练器
        TrainerClass = self.STAGE_TRAINERS[stage_name]
        trainer = TrainerClass(
            model=self.model,
            config=stage_config,
            device=self.device,
        )
        
        # 训练
        result = trainer.train(
            train_loader=train_loader,
            eval_loader=eval_loader,
            num_epochs=stage_config.get("epochs", 10),
            log_interval=self.log_interval,
            callback=callback,
        )
        
        self.global_step += result.total_steps
        
        return result
    
    def _prepare_dataloader(
        self,
        dataset: Dataset,
        stage: str,
        batch_size: int,
        shuffle: bool,
    ):
        """准备数据加载器"""
        # VQ-VAE 使用段落数据集
        if stage == "vqvae":
            if not isinstance(dataset, ParagraphDataset):
                # 尝试转换
                if hasattr(dataset, 'to_paragraph_dataset'):
                    dataset = dataset.to_paragraph_dataset()
                elif hasattr(dataset, 'paragraphs'):
                    pass  # 已经是正确格式
                else:
                    # 假设是段落列表
                    dataset = ParagraphDataset(list(dataset))
            
            return ParagraphDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        # Dynamics/Emotion/Finetune 使用序列数据集
        else:
            seq_len = 5  # 默认序列长度
            
            if isinstance(dataset, SequenceDataset):
                seq_dataset = dataset
            elif isinstance(dataset, ParagraphDataset):
                seq_dataset = SequenceDataset(dataset, seq_len=seq_len)
            else:
                # 尝试转换
                para_dataset = ParagraphDataset(list(dataset))
                seq_dataset = SequenceDataset(para_dataset, seq_len=seq_len)
            
            return ParagraphDataLoader(seq_dataset, batch_size=batch_size, shuffle=shuffle)
    
    def _save_checkpoint(self, name: str) -> None:
        """保存检查点"""
        checkpoint_dir = self.output_dir / "checkpoints" / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型权重
        self.model.save_pretrained(str(checkpoint_dir))
        
        # 保存训练状态
        save_training_state(
            path=str(checkpoint_dir / "training_state.json"),
            global_step=self.global_step,
            epoch=0,
            completed_stages=self.completed_stages,
            losses={},
        )
        
        print(f"Checkpoint saved to {checkpoint_dir}")
    
    def _load_checkpoint(self, path: str) -> None:
        """加载检查点"""
        checkpoint_path = Path(path)
        
        # 加载训练状态
        state_path = checkpoint_path / "training_state.json"
        if state_path.exists():
            state = load_training_state(str(state_path))
            self.global_step = state.get("global_step", 0)
            self.completed_stages = state.get("completed_stages", [])
        
        # 加载模型权重
        model_dir = ModelDirectory(str(checkpoint_path))
        if model_dir.exists():
            from ..model.model_utils import load_safetensors, safe_load_state_dict
            state_dict = load_safetensors(str(model_dir.weights_path), device=self.device)
            safe_load_state_dict(self.model, state_dict, strict=False)
        
        print(f"Resumed from {path} (step {self.global_step})")


# ============================================================================
# 便捷函数
# ============================================================================

def train_model(
    model: NeuralFlowModel,
    train_data: Dataset,
    stages: List[str] = ["vqvae", "dynamics"],
    output_dir: str = "./outputs",
    **kwargs,
) -> TrainingResult:
    """
    便捷训练函数
    
    Args:
        model: NeuralFlowModel
        train_data: 训练数据
        stages: 训练阶段
        output_dir: 输出目录
        **kwargs: 传递给 UnifiedTrainer
        
    Returns:
        TrainingResult
    """
    trainer = UnifiedTrainer(model, output_dir=output_dir, **kwargs)
    return trainer.train(train_data=train_data, stages=stages)
