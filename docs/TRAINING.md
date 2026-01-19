# NeuralFlow 训练指南

本文档详细说明如何准备数据、配置训练参数、执行训练流程。

---

## 目录

1. [数据格式](#数据格式)
2. [配置文件](#配置文件)
3. [训练阶段](#训练阶段)
4. [命令行使用](#命令行使用)
5. [云端训练](#云端训练)
6. [常见问题](#常见问题)

---

## 数据格式

### 1. 基础格式：JSONL

每行一个 JSON 对象，包含一个段落：

```jsonl
{"text": "这是第一个段落的内容。它可以包含多个句子。", "id": "doc1_p1"}
{"text": "这是第二个段落，继续前面的内容。", "id": "doc1_p2"}
{"text": "新文档的开始段落。", "id": "doc2_p1"}
```

**字段说明：**

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `text` | string | ✅ | 段落文本内容 |
| `id` | string | ❌ | 段落唯一标识 |
| `emotion` | string | ❌ | 情感标签 (happy/sad/angry/neutral 等) |
| `scene` | string | ❌ | 场景标签 (casual/formal/technical 等) |

### 2. 扩展格式：带情感/场景

```jsonl
{"text": "今天天气真好！", "emotion": "happy", "scene": "casual"}
{"text": "系统报告显示异常。", "emotion": "neutral", "scene": "technical"}
```

### 3. 序列格式 (用于 Dynamics 训练)

包含连续段落的序列，用于预测下一段落：

```jsonl
{"paragraphs": ["第一段", "第二段", "第三段"], "next": "第四段"}
{"paragraphs": ["段落A", "段落B"], "next": "段落C"}
```

### 4. 文本文件格式

纯文本文件，段落以空行分隔：

```text
这是第一个段落。
它可以有多行。

这是第二个段落。
空行用于分隔。

这是第三个段落。
```

---

## 配置文件

### 完整配置结构

```yaml
# configs/my_experiment.yaml

# 可选：继承基础配置
inherit: base

# 模型架构
model:
  d_latent: 512          # 潜空间维度
  d_model: 768           # 隐藏层维度
  num_layers: 6          # Transformer/Mamba 层数
  brain_type: mamba      # 动力学模型类型 (mamba/gru)
  codebook_size: 8192    # VQ-VAE 码本大小
  max_paragraph_len: 256 # 最大段落长度 (tokens)

# 训练参数
training:
  batch_size: 32
  learning_rate: 0.0001
  weight_decay: 0.01
  max_epochs: 100
  warmup_steps: 1000
  gradient_clip: 1.0
  
  # 各阶段特定参数
  vqvae:
    beta_commit: 0.25      # Commitment loss 权重
    gamma_codebook: 1.0    # Codebook loss 权重
  
  dynamics:
    ponder_cost_weight: 0.01  # ACT 思考代价权重

# 数据路径
data:
  train_path: data/train.jsonl
  eval_path: data/eval.jsonl

# 输出设置
output:
  dir: outputs/my_experiment
  save_every_n_epochs: 5
  log_interval: 100
```

### 预设配置

| 预设 | 参数量 | 用途 |
|------|--------|------|
| `tiny` | ~10M | 调试、单元测试 |
| `small` | ~228M | 本地训练、快速验证 |
| `base` | ~500M | 标准训练 |

```bash
# 使用预设
python scripts/train.py --preset small --data data/train.jsonl
```

---

## 训练阶段

NeuralFlow 采用 **4 阶段渐进训练**：

### Stage 1: VQ-VAE (码本学习)

**目标：** 学习将段落压缩为离散码本索引

**损失函数：**
- Commitment Loss: 鼓励编码器输出接近码本向量
- Codebook Loss: 更新码本向量

```bash
python scripts/train.py \
    --config configs/base.yaml \
    --data data/train.jsonl \
    --stages vqvae \
    --epochs 50
```

**监控指标：**
- `perplexity`: 码本利用率 (理想 > codebook_size * 0.1)
- `commit_loss`: 应逐渐下降

### Stage 2: Dynamics (动力学预测)

**目标：** 学习预测下一个段落的潜向量

**损失函数：**
- Prediction Loss: MSE(predicted_z, target_z)
- Ponder Cost: ACT 思考步数惩罚

```bash
python scripts/train.py \
    --config configs/base.yaml \
    --data data/train.jsonl \
    --stages dynamics \
    --epochs 30
```

**前置条件：** Stage 1 完成后 encoder 权重冻结

### Stage 3: Emotion (情感联合训练)

**目标：** 学习情感向量与生成的关联

```bash
python scripts/train.py \
    --config configs/base.yaml \
    --data data/train_with_emotion.jsonl \
    --stages emotion \
    --epochs 20
```

**数据要求：** 必须包含 `emotion` 字段

### Stage 4: Finetune (端到端微调)

**目标：** 全模型低学习率微调

```bash
python scripts/train.py \
    --config configs/base.yaml \
    --data data/train.jsonl \
    --stages finetune \
    --epochs 10 \
    --training.learning_rate 0.00001
```

### 完整流程

```bash
# 一次性执行所有阶段
python scripts/train.py \
    --config configs/base.yaml \
    --data data/train.jsonl \
    --stages vqvae dynamics emotion finetune \
    --output outputs/full_run
```

---

## 命令行使用

### 基本训练

```bash
python scripts/train.py \
    --config configs/base.yaml \
    --data data/train.jsonl \
    --output outputs/run1
```

### CLI 参数覆盖

```bash
python scripts/train.py \
    --config configs/base.yaml \
    --model.d_latent 256 \
    --training.batch_size 64 \
    --training.learning_rate 0.0002
```

### 从检查点恢复

```bash
python scripts/train.py \
    --config configs/base.yaml \
    --resume outputs/run1/checkpoints/epoch_20.pt
```

### 多 GPU 训练

```bash
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/base.yaml \
    --data data/train.jsonl
```

---

## 云端训练

### RunPod

```bash
# 提交任务
python scripts/cloud_train.py submit \
    --provider runpod \
    --gpu RTX4090 \
    --config configs/cloud.yaml \
    --data data/train.jsonl

# 查看状态
python scripts/cloud_train.py status --job-id <job_id>

# 下载结果
python scripts/cloud_train.py download --job-id <job_id> --output ./results
```

### Modal

```bash
python scripts/cloud_train.py submit \
    --provider modal \
    --gpu A100 \
    --config configs/cloud.yaml
```

### Lambda Labs

```bash
python scripts/cloud_train.py submit \
    --provider lambda \
    --gpu H100 \
    --config configs/cloud.yaml
```

### 成本估算

```bash
python scripts/cloud_train.py estimate --gpu RTX4090 --hours 8
# Output: Estimated cost: $2.72 (RunPod RTX4090 @ $0.34/hr)
```

---

## 数据准备脚本

### 从纯文本生成 JSONL

```python
# scripts/prepare_data.py
import json

def text_to_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 按空行分割段落
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, p in enumerate(paragraphs):
            json.dump({'text': p, 'id': f'p{i}'}, f, ensure_ascii=False)
            f.write('\n')

text_to_jsonl('raw_text.txt', 'data/train.jsonl')
```

### 数据验证

```bash
python -c "
from app.training.data_loader import ParagraphDataset
dataset = ParagraphDataset.from_jsonl('data/train.jsonl')
print(f'Total paragraphs: {len(dataset)}')
print(f'Sample: {dataset[0][:100]}...')
"
```

---

## 常见问题

### Q: 训练时出现 NaN loss

**原因：** 学习率过高或梯度爆炸

**解决：**
```yaml
training:
  learning_rate: 0.00005  # 降低学习率
  gradient_clip: 0.5      # 更严格的梯度裁剪
```

### Q: VQ-VAE perplexity 很低 (< 10)

**原因：** 码本利用率低，模式坍塌

**解决：**
- 增大 `codebook_size`
- 降低 `beta_commit`
- 使用 EMA 码本更新

### Q: Dynamics 预测 loss 不下降

**原因：** 编码器产生的潜向量区分度不够

**解决：**
- 确保 VQ-VAE 训练充分
- 增大 `d_latent`
- 检查数据多样性

### Q: 内存不足 (OOM)

**解决：**
```bash
# 减小 batch size
--training.batch_size 8

# 使用梯度累积
--training.gradient_accumulation_steps 4

# 使用混合精度
--training.mixed_precision true
```

---

## 输出文件结构

```
outputs/my_experiment/
├── checkpoints/
│   ├── epoch_5.pt
│   ├── epoch_10.pt
│   └── best.pt
├── logs/
│   ├── train.log
│   └── tensorboard/
├── config.yaml          # 保存的配置
└── metrics.json         # 训练指标
```

---

## 下一步

1. 准备你的数据 → `data/train.jsonl`
2. 选择预设配置 → `--preset small`
3. 启动训练 → `python scripts/train.py ...`
4. 监控指标 → `tensorboard --logdir outputs/*/logs`
5. 评估模型 → `python scripts/evaluate.py`

**问题反馈：** 请提交 Issue 或查看 `docs/FAQ.md`
