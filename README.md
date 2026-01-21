<p align="center">
  <h1 align="center">🧠 NeuralFlow AMHVQ+</h1>
  <p align="center">
    <strong>Adaptive Matryoshka Hierarchical VQ + 语义-结构-符号三通道融合框架</strong>
  </p>
  <p align="center">
    <em>段落级语义推理 · 深度情感调制 · 场景自适应精度</em>
  </p>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=for-the-badge&logo=pytorch" alt="PyTorch"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License"/></a>
</p>

<p align="center">
  <sub>🤖 本项目完全使用 Google Antigravity AI 开发（包括本 README）</sub>
</p>

---

## 🎯 项目定位

NeuralFlow AMHVQ+ 是一个**实验性 LLM 架构框架**，探索与传统 Token-by-Token 不同的语言模型设计范式：

| 传统 LLM | NeuralFlow AMHVQ+ |
|:--------:|:-----------------:|
| Token → Token → Token | **Paragraph → UnifiedLatent → Paragraph** |
| 单一向量表示 | **三通道并行：语义 + 结构 + 符号** |
| 固定计算量 | **ACT 自适应思考时间** |
| 情感作为输入 | **深度情感调制 (AdaLN)** |

---

## 🌟 三大核心创新

### 1️⃣ 三通道并行编码架构

AMHVQ+ 的核心是将文本信息分解到三个正交通道，根据场景智能激活：

```
                           输入文本
                              ↓
                    ┌─────────────────────┐
                    │  Scene-Aware Router │  ← 自动场景检测
                    └─────────────────────┘
                              ↓
        ┌─────────────────────┼─────────────────────┐
        ↓                     ↓                     ↓
┌───────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   🧠 语义通道   │   │   📐 结构通道    │   │   🔣 符号通道    │
│   (始终激活)    │   │   (条件激活)     │   │   (条件激活)     │
├───────────────┤   ├─────────────────┤   ├─────────────────┤
│ • Matryoshka  │   │ • AST解析       │   │ • 关键Token检测  │
│ • Residual VQ │   │ • GraphMemory   │   │ • 位置锚定       │
│ • 层次化表示   │   │ • 骨架提取       │   │ • Slot映射       │
└───────────────┘   └─────────────────┘   └─────────────────┘
        ↓                     ↓                     ↓
        └─────────────────────┼─────────────────────┘
                              ↓
                    ┌─────────────────────┐
                    │    UnifiedLatent    │
                    └─────────────────────┘
```

**场景自适应精度：**

| 场景 | 语义维度 | 结构通道 | 符号通道 | 适用场景 |
|:----:|:--------:|:--------:|:--------:|:--------:|
| 💬 chat | 256 | ❌ | ❌ | 日常对话、创意写作 |
| 💻 coding | 512 | ✅ | ✅ | 代码生成、API调用 |
| 📄 technical | 512 | ✅ | ❌ | 技术文档、公式 |
| 🎨 creative | 256 | ❌ | ❌ | 故事创作 |

---

### 2️⃣ 深度情感调制系统 (AdaLN + VAD)

**传统方法**将情感作为简单的向量加法，而 AMHVQ+ 采用 **AdaLN (Adaptive Layer Normalization)** 机制，让情感信号深度影响每一层计算：

```
传统方法:  output = x + emotion_vector     ← 仅影响值

AMHVQ+:   output = γ(emotion) * RMSNorm(x) + β(emotion)
                   ↑                         ↑
              调制分布缩放              调制偏移
```

**VAD 情感空间 (Valence-Arousal-Dominance)：**

基于心理学研究的三维连续情感表示：
- **Valence (效价)**: -1.0 (不愉快) → +1.0 (愉快)
- **Arousal (唤醒度)**: -1.0 (平静) → +1.0 (激动)
- **Dominance (支配度)**: -1.0 (顺从) → +1.0 (掌控)

内置 **25+ 种情感预设**：

| 基本情绪 | 复杂情感 | 文学情感 |
|:---------|:---------|:---------|
| happy, sad, angry | anxious, content | bittersweet (苦乐参半) |
| afraid, surprised | frustrated, proud | wistful (惆怅) |
| disgusted, neutral | curious, nostalgic | serene (宁静) |

**情感混合：**
```python
from app.modulation import blend_emotions

# 70% happy + 30% sad = 淡淡的满足感
mixed = blend_emotions("happy", "sad", weight=0.3)
```

**层次化情感调制：** 对 HierarchicalLatent 的每个层级独立调制：
- 🌐 Global 级别：整体语义风格
- 📦 Chunk 级别：局部段落情感
- 📍 Detail 级别：细粒度细节

---

### 3️⃣ Matryoshka + Residual VQ 嵌套量化

**Matryoshka 表示学习：**

渐进式维度嵌套，一次训练，按需选择精度：
```
[64] ⊂ [128] ⊂ [256] ⊂ [512]
 粗       ↓       ↓      细
```
- 低延迟任务：使用前 64 维
- 高保真任务：使用全部 512 维
- 无需重新训练

**Residual VQ 残差量化：**

多层渐进式重建，每层捕获更细粒度信息：
```
x → Q₁(x) → Q₂(残差₁) → Q₃(残差₂) → Q₄(残差₃)
      ↓          ↓          ↓          ↓
    码本1      码本2      码本3      码本4
```

---

## 📦 安装

```bash
# 克隆项目
git clone https://github.com/your-username/neuralflow-amhvq.git
cd neuralflow-amhvq

# 安装依赖
pip install -r requirements.txt

# 验证安装
python -m pytest tests/ -x
```

---

## 🚀 快速开始

### 创建模型

```python
from app.model.unified_model import UnifiedNeuralFlowModel

# 从预设创建 (small/base/coding)
model = UnifiedNeuralFlowModel.from_preset("base")

# 或从配置文件
from app.interfaces.config import Config
config = Config.load("configs/amhvq_base.yaml")
model = UnifiedNeuralFlowModel(config)
```

### 编码与解码

```python
# 编码 (自动场景检测)
latent = model.encode("def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)")

print(f"检测场景: {latent.scene}")              # coding
print(f"结构通道: {latent.has_structure}")      # True
print(f"符号锚点数: {latent.symbols.num_anchors if latent.symbols else 0}")

# 解码
output = model.decode(latent)
```

### 推理 API

```python
from app.inference import AMHVQInference

inference = AMHVQInference.from_pretrained("./model_dir")

# 精度自适应生成
result = inference.generate(
    "实现一个二分查找算法",
    precision="adaptive",   # low/medium/high/adaptive
    temperature=0.7,
)
print(result.text)
```

---

## 🎓 训练

### 训练命令

```bash
python scripts/train.py \
    --config configs/amhvq_base.yaml \
    --data data/train.jsonl \
    --stages vqvae dynamics \
    --output outputs/my_model
```

### 四阶段训练流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Stage 1    │ ──▶ │  Stage 2    │ ──▶ │  Stage 3    │ ──▶ │  Stage 4    │
│  VQ-VAE     │     │  Dynamics   │     │  Emotion    │     │  Finetune   │
│  码本+重建   │     │  动力学预测  │     │  情感调制    │     │  端到端微调  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

| 阶段 | 目标 | 损失函数 |
|:----:|:----:|:--------:|
| VQ-VAE | 学习语义压缩 | reconstruction + commitment |
| Dynamics | 学习段落预测 | next-latent + ponder cost |
| Emotion | 学习情感调制 | emotion + scene loss |
| Finetune | 端到端优化 | all (低学习率) |

### 课程学习

渐进式激活三个通道：
```yaml
curriculum_warmup: 2      # 前 2 epoch 仅语义
structure_start_epoch: 3  # 第 3 epoch 激活结构
symbol_start_epoch: 5     # 第 5 epoch 激活符号
```

---

## 📚 文档

| 文档 | 说明 |
|:-----|:-----|
| [**🌟 新手入门指南**](docs/QUICKSTART.md) | 30 分钟完成第一次训练 |
| [**📖 训练详解**](docs/TRAINING.md) | 数据格式、配置、阶段详解 |
| [**🏗️ 架构文档**](docs/AMHVQ_ARCHITECTURE.md) | AMHVQ+ 技术细节 |

---

## 📁 项目架构

```
app/
├── 🔌 interfaces/           # 核心接口
│   ├── unified_latent.py       # UnifiedLatent 三通道结构
│   ├── base_module.py          # HierarchicalLatent
│   └── config.py               # 分层配置系统
│
├── 📥 io/                   # 编解码器 (14 模块)
│   ├── unified_encoder.py      # 三通道统一编码器
│   ├── unified_decoder.py      # 结构引导解码器
│   ├── residual_vq.py          # 残差向量量化
│   ├── matryoshka.py           # Matryoshka 嵌套投影
│   ├── semantic_chunker.py     # 语义边界分块
│   ├── structure_parser.py     # AST/句法解析
│   └── symbol_anchor.py        # 符号锚点编码
│
├── 🧠 brain/                # 动力学推理
│   ├── hierarchical_dynamics.py  # 层次化动力学
│   ├── dynamics_model.py         # Mamba/GRU
│   └── act_controller.py         # 自适应计算时间
│
├── 💾 memory/               # 记忆系统
│   ├── hierarchical_memory.py    # 层次化检索
│   ├── graph_memory.py           # 结构图存储
│   └── latent_memory_bank.py     # FAISS 向量库
│
├── 🎭 modulation/           # 情感与场景调制
│   ├── adaln.py                  # AdaLN 自适应层归一化
│   ├── hierarchical_modulation.py # 层次化情感调制
│   ├── vad_utils.py              # VAD 情感空间
│   ├── scene_encoder.py          # 场景路由器
│   └── emotion/                  # 情感子模块
│       ├── vad_encoder.py        # VAD → 向量
│       └── trainable_encoder.py  # 可训练编码器
│
├── 📦 model/                # 模型封装
│   ├── unified_model.py        # UnifiedNeuralFlowModel
│   └── neuralflow_model.py     # 旧版兼容
│
├── 🏃 inference/            # 推理接口
│   └── amhvq_inference.py      # AMHVQInference
│
└── 🎓 training/             # 训练系统
    ├── unified_training_stages.py  # 三通道训练阶段
    ├── unified_trainer.py          # 统一训练器
    ├── data_pipeline.py            # 数据管道
    └── distributed.py              # 分布式训练

configs/
├── amhvq_base.yaml      # 基础配置
├── amhvq_coding.yaml    # 代码专用配置
└── amhvq_small.yaml     # 轻量版配置

scripts/
├── train.py             # 本地训练
├── cloud_train.py       # 云端训练 (RunPod/Modal/Lambda)
└── evaluate.py          # 模型评估

demo/
└── run_pipeline.py      # 端到端演示
```

---

## 📊 技术特点

| 特性 | 描述 |
|:----:|:-----|
| **语义压缩** | 段落 → 512 维向量，高压缩比语义保留 |
| **结构保真** | 代码骨架提取 + 槽位填充，提升变量名/API 精确度 |
| **符号锚定** | 关键 Token 位置锚定，减少生成偏差 |
| **场景自适应** | 自动检测场景，动态调整通道激活 |
| **深度情感调制** | AdaLN 机制调制每层计算，非简单加法 |
| **渐进精度** | Matryoshka 表示支持按需选择维度 |
| **层次记忆** | 两级检索 (Global + Chunk) 增强上下文 |

---

## 🤝 参与贡献

欢迎各种形式的贡献！

### 当前优先事项

- [ ] 🧪 预训练权重（需要大规模训练）
- [ ] 📊 标准化基准测试
- [ ] 🔧 更多结构解析器 (JSON/Markdown)
- [ ] 📱 模型量化导出

### 本地开发

```bash
# 运行测试
python -m pytest tests/ -v

# 运行演示
python -m demo.run_pipeline
```

---

## 📚 学术参考

- [VQ-VAE](https://arxiv.org/abs/1711.00937) - 离散潜变量表示学习
- [Matryoshka Representations](https://arxiv.org/abs/2205.13147) - 嵌套表示学习
- [Mamba](https://arxiv.org/abs/2312.00752) - 选择性状态空间模型
- [ACT](https://arxiv.org/abs/1603.08983) - 自适应计算时间
- [AdaLN (DiT)](https://arxiv.org/abs/2212.09748) - 扩散模型中的自适应层归一化
- [Russell VAD](https://doi.org/10.1037/h0077714) - 情感环形模型

---

## 📄 License

MIT License © 2026

---

<p align="center">
  <strong>🚀 NeuralFlow AMHVQ+ - 探索段落级语义推理的下一代范式</strong>
</p>

<p align="center">
  <sub>Developed with 💜 using Google Antigravity AI</sub>
</p>
