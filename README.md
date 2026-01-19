# NeuralFlow - 段落级语义推理 LLM 架构框架

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **实验性项目** - 探索段落级语义推理的下一代 LLM 架构

重要提醒：本项目完全使用Google Antigravity完成(包括该README文件)。

## 🌟 核心理念

NeuralFlow 实现了一种与传统 Token-by-Token 预测不同的 LLM 设计范式：

| 传统 LLM | NeuralFlow |
|----------|------------|
| Token → Token → Token (离散序列) | Paragraph → Latent → Dynamics → Paragraph |
| 线性自回归生成 | 连续语义空间推理 |
| 固定计算量 | 自适应思考时间 (ACT) |

### 设计哲学

1. **段落级思考** - 模型在抽象语义空间规划，而非逐字生成
2. **VQ-VAE 离散压缩** - 将段落压缩为离散码本索引
3. **Mamba SSM 动力学** - 线性复杂度的状态空间模型预测下一步
4. **自适应计算时间 (ACT)** - 简单问题快答，复杂问题深思
5. **深度情感调制 (AdaLN)** - 情感/场景深度影响每层计算

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────────────┐
│                      NeuralFlow Model                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   Encoder    │ => │   Dynamics   │ => │   Decoder    │   │
│  │   (VQ-VAE)   │    │ (Mamba+ACT)  │    │  (生成文本)   │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│         ↑                   ↑                    ↑           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  Modulation  │    │    Memory    │    │   Emotion    │   │
│  │ (AdaLN/FiLM) │    │ (FAISS+Graph)│    │   Encoder    │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 📦 安装

```bash
# 克隆项目
git clone https://github.com/your-username/neuralflow.git
cd neuralflow

# 安装依赖
pip install -r requirements.txt

# 可选: GPU 加速
pip install faiss-gpu mamba-ssm
```
## 📚 文档

| 文档 | 说明 |
|------|------|
| [**新手入门指南**](docs/QUICKSTART.md) | 🌟 第一次使用？从这里开始！ |
| [训练指南](docs/TRAINING.md) | 数据格式、配置、训练阶段详解 |

## 🚀 快速开始

### 使用预设配置

```python
from app.model import NeuralFlowModel
from app.interfaces import Config

# 创建模型 (small: 228M 参数)
model = NeuralFlowModel.from_preset("small")

# 或从配置文件加载
config = Config.load("configs/base.yaml")
model = NeuralFlowModel(config)
```

### 训练模型

```bash
# 本地训练
python scripts/train.py \
    --config configs/base.yaml \
    --data data/train.jsonl \
    --stages vqvae dynamics \
    --output outputs/my_model
```

### 云端训练 (RunPod/Modal/Lambda)

```bash
# 提交云训练任务
python scripts/cloud_train.py submit \
    --provider runpod \
    --gpu RTX4090 \
    --config configs/cloud.yaml

# 查看成本估算
python scripts/cloud_train.py estimate --gpu RTX4090 --hours 8
```

## 📁 项目结构

```
app/
├── interfaces/              # 统一接口和配置
│   ├── config.py           # 分层配置 (Model/Training/Pipeline)
│   ├── config_loader.py    # 增强配置加载器 (继承/环境变量/CLI覆盖)
│   └── base_module.py      # BaseModule, LatentVector
│
├── io/                      # 输入/输出层
│   ├── paragraph_encoder.py    # VQ-VAE 编码器
│   ├── paragraph_decoder.py    # 自回归解码器
│   └── vq_codebook.py          # 向量量化码本
│
├── brain/                   # 核心推理
│   ├── dynamics_model.py       # Mamba/GRU 动力学
│   ├── act_controller.py       # 自适应计算时间
│   └── reasoning_loop.py       # 推理循环
│
├── memory/                  # 记忆系统
│   ├── latent_memory_bank.py   # FAISS 向量存储
│   └── graph_memory.py         # 知识图谱
│
├── modulation/              # 控制调制
│   ├── adaln.py                # AdaLN 层
│   ├── emotion_encoder.py      # VAD 情感编码
│   └── scene_encoder.py        # 场景编码
│
├── training/                # 训练系统
│   ├── unified_trainer.py      # 统一训练器
│   ├── training_stages.py      # 4阶段训练 (VQ-VAE/Dynamics/Emotion/Finetune)
│   └── data_pipeline.py        # 数据加载
│
└── model/                   # 模型封装
    └── neuralflow_model.py     # NeuralFlowModel 主类

cloud/                       # 云训练支持
├── providers/               # RunPod/Modal/Lambda 适配器
├── job_manager.py           # 任务管理
└── sync_utils.py            # 数据同步

configs/                     # 配置文件
├── base.yaml                # 基础配置
├── cloud.yaml               # 云训练配置
└── presets/                 # 快速预设 (tiny/small)

scripts/                     # CLI 脚本
├── train.py                 # 本地训练
├── cloud_train.py           # 云训练
├── evaluate.py              # 模型评估
└── config_gen.py            # 配置生成器
```

## 🔧 配置系统

### 分层配置
```yaml
# configs/base.yaml
model:
  d_latent: 512
  d_model: 768
  brain_type: mamba
  codebook_size: 8192

training:
  batch_size: 32
  learning_rate: 0.0001
  max_epochs: 100

pipeline:
  enable_memory: true
  enable_emotion: true
```

### 配置继承
```yaml
# configs/my_experiment.yaml
inherit: base

model:
  d_latent: 256  # 覆盖父配置
```

### CLI 覆盖
```bash
python scripts/train.py \
    --config configs/base.yaml \
    --model.d_latent 256 \
    --training.batch_size 64
```

## 📊 训练阶段

| 阶段 | 说明 | 训练目标 |
|------|------|----------|
| **1. VQ-VAE** | 码本学习 | commitment + codebook loss |
| **2. Dynamics** | 动力学预测 | 下一段落预测 + ponder cost |
| **3. Emotion** | 情感调制 | 联合情感编码器训练 |
| **4. Finetune** | 端到端微调 | 全模型低学习率微调 |

## ☁️ 云训练支持

| 平台 | GPU | 价格参考 | 推荐场景 |
|------|-----|----------|----------|
| **RunPod** | RTX4090, A100, H100 | $0.34-1.99/hr | 长时间训练 |
| **Modal** | T4, A100, H100 | 按秒计费 | 短任务/调试 |
| **Lambda Labs** | A6000, A100, H100 | $0.80-2.99/hr | 正式训练 |

## � 开发进度

| 模块 | 状态 | 说明 |
|------|------|------|
| interfaces/ | ✅ 完成 | 基类、配置、注册表 |
| io/ | ✅ 完成 | VQ-VAE 编解码器 |
| brain/ | ✅ 完成 | Mamba/GRU + ACT |
| memory/ | ✅ 完成 | FAISS + GraphRAG |
| modulation/ | ✅ 完成 | AdaLN + 情感编码 |
| training/ | ✅ 完成 | 4阶段训练 + 分布式 |
| cloud/ | ✅ 完成 | 3平台云训练支持 |
| 预训练权重 | 🔲 待开发 | 需大规模训练 |

## 📚 参考文献

- [VQ-VAE](https://arxiv.org/abs/1711.00937) - 离散潜变量
- [Mamba](https://arxiv.org/abs/2312.00752) - 选择性状态空间模型
- [ACT](https://arxiv.org/abs/1603.08983) - 自适应计算时间
- [AdaLN](https://arxiv.org/abs/2212.09748) - 自适应层归一化

## 📄 License

MIT License

---

**注意**: 本项目完全使用 Google Antigravity AI 完成开发。
