#!/usr/bin/env python
"""
evaluate.py - 模型评估脚本

使用示例:
    python scripts/evaluate.py --model models/my-model/ --data data/test.jsonl
    
    python scripts/evaluate.py --model models/my-model/ --data data/test.jsonl --metrics vqvae dynamics
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from typing import Dict, List, Any

from app.interfaces.config import Config
from app.model import NeuralFlowModel
from app.training import ParagraphDataset, SequenceDataset
from app.training.data_pipeline import NeuralFlowDataset


def parse_args():
    parser = argparse.ArgumentParser(description="NeuralFlow Evaluation Script")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model directory path",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Evaluation data file",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["vqvae", "dynamics"],
        choices=["vqvae", "dynamics", "emotion", "generation"],
        help="Metrics to evaluate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate",
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
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        return NeuralFlowDataset.from_list(lines[:max_samples] if max_samples else lines)


def evaluate_vqvae(
    model: NeuralFlowModel,
    data: NeuralFlowDataset,
    batch_size: int = 8,
    device: str = "cpu",
) -> Dict[str, float]:
    """评估 VQ-VAE 指标"""
    model.eval()
    
    dataset = data.to_paragraph_dataset()
    total_loss = 0.0
    total_perplexity = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch_texts = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            
            output = model.encoder(batch_texts)
            
            if output.vq_output is not None:
                total_loss += output.vq_output.commitment_loss.item()
                total_perplexity += output.vq_output.perplexity.item()
            
            num_batches += 1
    
    return {
        "vqvae_commitment_loss": total_loss / max(num_batches, 1),
        "vqvae_perplexity": total_perplexity / max(num_batches, 1),
    }


def evaluate_dynamics(
    model: NeuralFlowModel,
    data: NeuralFlowDataset,
    batch_size: int = 8,
    seq_len: int = 5,
    device: str = "cpu",
) -> Dict[str, float]:
    """评估 Dynamics 指标"""
    model.eval()
    
    para_dataset = data.to_paragraph_dataset()
    seq_dataset = SequenceDataset(para_dataset, seq_len=seq_len)
    
    total_mse = 0.0
    total_cosine_sim = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for i in range(min(len(seq_dataset), 100)):  # 限制评估样本
            inputs, target = seq_dataset[i]
            
            # 编码序列
            z_inputs = []
            for text in inputs:
                output = model.encoder([text])
                z_inputs.append(output.latent.vector)
            
            z_input = torch.stack(z_inputs, dim=1)  # [1, seq_len, d_latent]
            
            # 编码目标
            target_output = model.encoder([target])
            z_target = target_output.latent.vector
            
            # 预测
            output = model.dynamics(z_input)
            z_pred = output.predicted_latent
            
            # 计算指标
            mse = F.mse_loss(z_pred, z_target).item()
            cosine_sim = F.cosine_similarity(z_pred, z_target, dim=-1).mean().item()
            
            total_mse += mse
            total_cosine_sim += cosine_sim
            num_samples += 1
    
    return {
        "dynamics_mse": total_mse / max(num_samples, 1),
        "dynamics_cosine_sim": total_cosine_sim / max(num_samples, 1),
    }


def evaluate_emotion(
    model: NeuralFlowModel,
    data: NeuralFlowDataset,
    device: str = "cpu",
) -> Dict[str, float]:
    """评估 Emotion 编码指标"""
    model.eval()
    
    # 测试情感编码器
    test_emotions = ["happy", "sad", "angry", "neutral", "excited", "afraid"]
    
    embeddings = []
    with torch.no_grad():
        for emotion in test_emotions:
            emb = model.emotion_encoder(emotion)
            embeddings.append(emb)
    
    # 计算情感向量之间的距离
    embeddings = torch.cat(embeddings, dim=0)
    
    # 平均成对距离
    dists = torch.cdist(embeddings, embeddings)
    avg_dist = dists[dists > 0].mean().item()
    
    return {
        "emotion_avg_distance": avg_dist,
        "emotion_vocab_size": model.emotion_encoder.emotion_count,
    }


def evaluate_generation(
    model: NeuralFlowModel,
    data: NeuralFlowDataset,
    num_samples: int = 5,
    device: str = "cpu",
) -> Dict[str, Any]:
    """评估生成质量 (定性)"""
    model.eval()
    
    dataset = data.to_paragraph_dataset()
    results = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            input_text = dataset[i]
            
            # 编码
            z = model.encode([input_text])
            
            # 预测下一个潜向量
            z_seq = z.unsqueeze(1)
            z_next = model.predict_next(z_seq)
            
            # 解码 (如果解码器可用)
            try:
                output_text = model.decode(z_next, max_length=50)
                generated = output_text[0] if output_text else "[生成失败]"
            except Exception as e:
                generated = f"[Error: {e}]"
            
            results.append({
                "input": input_text[:100] + "..." if len(input_text) > 100 else input_text,
                "generated": generated[:100] + "..." if len(generated) > 100 else generated,
            })
    
    return {"generation_samples": results}


def main():
    args = parse_args()
    
    # 设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("NeuralFlow Evaluation")
    print("=" * 60)
    
    # 加载模型
    print(f"Loading model from {args.model}...")
    try:
        model = NeuralFlowModel.from_pretrained(args.model, device=device)
    except FileNotFoundError:
        print("Model directory not found, trying preset...")
        model = NeuralFlowModel.from_preset("small")
        model.to(device)
    
    print(model)
    
    # 加载数据
    print(f"Loading data from {args.data}...")
    data = load_data(args.data, max_samples=args.max_samples)
    print(f"  Samples: {len(data)}")
    
    # 评估
    results = {}
    
    for metric in args.metrics:
        print(f"\nEvaluating {metric}...")
        
        if metric == "vqvae":
            results.update(evaluate_vqvae(model, data, args.batch_size, device))
        elif metric == "dynamics":
            results.update(evaluate_dynamics(model, data, args.batch_size, device=device))
        elif metric == "emotion":
            results.update(evaluate_emotion(model, data, device))
        elif metric == "generation":
            results.update(evaluate_generation(model, data, device=device))
    
    # 输出结果
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, list):
            print(f"  {key}:")
            for item in value:
                print(f"    - {item}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
