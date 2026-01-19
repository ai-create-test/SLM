#!/usr/bin/env python
"""
train.py - 标准化训练入口

使用示例:
    # 完整训练
    python scripts/train.py --config configs/base.yaml --data data/train.jsonl
    
    # 指定阶段
    python scripts/train.py --config configs/base.yaml --stages vqvae dynamics
    
    # 恢复训练
    python scripts/train.py --config configs/base.yaml --resume outputs/checkpoint/
    
    # 分布式训练
    torchrun --nproc_per_node=4 scripts/train.py --config configs/base.yaml
"""

import argparse
import sys
import os
from pathlib import Path

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from app.interfaces.config import Config
from app.model import NeuralFlowModel
from app.training import UnifiedTrainer, ParagraphDataset
from app.training.data_pipeline import NeuralFlowDataset
from app.training.distributed import setup_distributed, is_main_process, sync_print


def parse_args():
    parser = argparse.ArgumentParser(
        description="NeuralFlow Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # 配置
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config file path (YAML or JSON)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["small", "base", "large"],
        default="base",
        help="Config preset",
    )
    
    # 数据
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Training data file (JSONL, Parquet, or text)",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help="Evaluation data file (optional)",
    )
    
    # 训练
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["vqvae", "dynamics"],
        choices=["vqvae", "dynamics", "emotion", "finetune"],
        help="Training stages to run",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs for all stages",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    
    # 输出
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    
    # 设备
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use",
    )
    
    # 日志
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log interval (steps)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="Save interval (steps)",
    )
    
    return parser.parse_args()


def load_data(path: str, max_samples: int = None) -> NeuralFlowDataset:
    """加载数据"""
    path = Path(path)
    
    if path.suffix == ".jsonl":
        return NeuralFlowDataset.from_jsonl(str(path), max_samples=max_samples)
    elif path.suffix == ".parquet":
        return NeuralFlowDataset.from_parquet(str(path), max_samples=max_samples)
    elif path.suffix in [".txt", ".text"]:
        return NeuralFlowDataset.from_text_file(str(path))
    else:
        # 尝试按行读取
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        return NeuralFlowDataset.from_list(lines)


def main():
    args = parse_args()
    
    # 分布式初始化
    setup_distributed()
    
    sync_print("="*60)
    sync_print("NeuralFlow Training")
    sync_print("="*60)
    
    # 加载配置
    if args.config:
        config = Config.load(args.config)
        sync_print(f"Loaded config from {args.config}")
    else:
        config = Config.from_preset(args.preset)
        sync_print(f"Using preset: {args.preset}")
    
    # 创建模型
    sync_print("Creating model...")
    model = NeuralFlowModel(config)
    sync_print(model)
    
    # 加载数据
    sync_print(f"Loading data from {args.data}...")
    train_data = load_data(args.data)
    sync_print(f"  Train samples: {len(train_data)}")
    
    eval_data = None
    if args.eval_data:
        eval_data = load_data(args.eval_data)
        sync_print(f"  Eval samples: {len(eval_data)}")
    
    # 构建阶段配置
    stage_configs = {}
    for stage in args.stages:
        stage_config = {}
        if args.epochs:
            stage_config["epochs"] = args.epochs
        if args.batch_size:
            stage_config["batch_size"] = args.batch_size
        if args.lr:
            stage_config["learning_rate"] = args.lr
        if stage_config:
            stage_configs[stage] = stage_config
    
    # 创建训练器
    trainer = UnifiedTrainer(
        model=model,
        config=config,
        device=args.device,
        output_dir=args.output,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )
    
    # 训练
    sync_print(f"\nStarting training...")
    sync_print(f"  Stages: {args.stages}")
    sync_print(f"  Output: {args.output}")
    
    result = trainer.train(
        train_data=train_data.to_paragraph_dataset(),
        eval_data=eval_data.to_paragraph_dataset() if eval_data else None,
        stages=args.stages,
        stage_configs=stage_configs,
        resume_from=args.resume,
    )
    
    # 结果
    sync_print("\n" + "="*60)
    sync_print("Training Complete!")
    sync_print("="*60)
    sync_print(f"Total steps: {result.total_steps}")
    sync_print(f"Total time: {result.total_time:.1f}s")
    sync_print(f"Model saved: {result.final_model_path}")
    
    for stage, stage_result in result.stage_results.items():
        sync_print(f"  {stage}: loss={stage_result.final_loss:.4f}")


if __name__ == "__main__":
    main()
