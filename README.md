# NeuralFlow - 前沿 LLM 架构实验框架

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个实现段落级语义推理的下一代 LLM 架构框架。

重要提醒：本项目完全使用Google Antigravity完成，没有任何人工介入。(包括该README文件)

## 🌟 核心理念

本框架探索了一种与传统 Token-by-Token 预测不同的 LLM 设计范式：

```
传统 LLM:   Token → Token → Token → ... (离散序列)
NeuralFlow: Paragraph → LatentSpace → Dynamics → Paragraph (连续语义空间)
```

### 设计哲学

1. **段落级思考** - 模型在抽象语义空间规划，而非逐字生成
2. **离散压缩** - 使用 VQ-VAE 将段落压缩为离散码本索引
3. **动态推理** - Mamba SSM 作为核心动力学系统预测下一步
4. **自适应思考** - ACT 机制实现简单问题快答、复杂问题深思
5. **深度调制** - AdaLN 让情感/场景深度影响每层计算

## 🏗️ 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        NeuralFlow Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │    IO       │ => │    Brain    │ => │  Decoder    │          │
│  │  (VQ-VAE)   │    │(Mamba+ACT)  │    │ (生成文本)  │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│        ↑                  ↑ ↓                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │  Modulation │    │   Memory    │    │ Reflection  │          │
│  │(AdaLN/FiLM) │    │(FAISS/Graph)│    │ (回溯/评价) │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│                           ↑                                       │
│                    ┌─────────────┐                               │
│                    │   Search    │                               │
│                    │  (联网检索)  │                               │
│                    └─────────────┘                               │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
app/
├── interfaces/           # 统一接口定义
│   ├── base_module.py   # BaseModule, LatentVector, MemoryItem
│   ├── config.py        # ModelConfig, TrainingConfig, PipelineConfig
│   └── registry.py      # 模块注册表
│
├── io/                   # 输入/输出层
│   ├── vq_codebook.py   # VQ-VAE 码本 (离散瓶颈层)
│   ├── paragraph_encoder.py  # 段落 → 潜向量
│   ├── paragraph_decoder.py  # 潜向量 → 段落
│   └── semantic_segmenter.py # 语义分割
│
├── brain/               # 核心大脑
│   ├── dynamics_model.py    # Mamba SSM 动力学预测
│   ├── act_controller.py    # 自适应计算时间
│   ├── halt_unit.py         # 停止决策神经元
│   └── reasoning_loop.py    # 推理循环协调器
│
├── memory/              # 记忆系统
│   ├── latent_memory_bank.py    # FAISS 潜向量库
│   ├── query_retriever.py       # 问题导向检索
│   ├── cross_attention_fuser.py # 记忆融合层
│   └── graph_memory.py          # 类图结构存储
│
├── modulation/          # 控制与调制
│   ├── adaln.py         # 自适应层归一化 (深度情感注入)
│   ├── film.py          # FiLM 调制层
│   ├── emotion_encoder.py   # 情感编码器
│   └── scene_encoder.py     # 场景编码器
│
├── reflection/          # 自我回溯
│   ├── trajectory_logger.py # 推理轨迹记录
│   ├── backtracker.py       # 回溯执行器
│   └── self_critic.py       # 自我评价
│
├── search/              # 联网搜索
│   ├── search_interface.py  # 搜索抽象接口
│   ├── web_search.py        # Web 搜索实现
│   ├── knowledge_injector.py # 知识注入器
│   └── cache.py             # 搜索缓存
│
├── core/                # 保留的基础组件
│   ├── bpe_tokenizer.py     # BPE Tokenizer
│   └── tokenizer_factory.py # Tokenizer 工厂
│
└── pipeline.py          # 端到端流水线
```

## 🚀 快速开始

### 安装依赖

```bash
pip install torch numpy faiss-cpu pyyaml tiktoken
# 可选: GPU 加速
pip install faiss-gpu mamba-ssm
```

### 基础使用

```python
from app import NeuralFlowPipeline, Config

# 从预设创建
pipeline = NeuralFlowPipeline.from_preset("base")

# 生成
output = pipeline.generate(
    "请解释量子计算的基本原理",
    emotion="curious",
    scene="teaching",
)
print(output.text)
print(f"推理步数: {output.reasoning_steps}")
```

### 自定义配置

```python
from app import Config, ModelConfig

config = Config(
    model=ModelConfig(
        d_latent=512,
        d_model=768,
        brain_type="mamba",
        max_think_steps=10,
        codebook_size=8192,
    )
)

pipeline = NeuralFlowPipeline.from_config(config)
```

## 🔬 核心技术

### 1. VQ-VAE 语义压缩

```python
from app import ParagraphEncoder, VQCodebook

encoder = ParagraphEncoder(d_model=768, d_latent=512, use_vq=True)
output = encoder("这是一个测试段落...")

z_quantized = output.latent.vector    # 量化后的潜向量
indices = output.latent.codebook_indices  # 离散码本索引
```

### 2. Mamba 动力学模型

```python
from app import DynamicsModel

model = DynamicsModel(
    d_latent=512,
    d_model=768,
    brain_type="mamba",  # 或 "gru"
)

# 预测下一个潜向量
z_history = torch.randn(batch, seq_len, 512)
output = model(z_history)
z_next = output.predicted_latent
```

### 3. 自适应计算时间 (ACT)

```python
from app import ACTController

act = ACTController(d_model=768, max_steps=10)

output = act(
    initial_state=state,
    step_fn=thinking_step,
)
print(f"实际思考步数: {output.num_steps}")
```

### 4. 深度情感调制 (AdaLN)

```python
from app import AdaptiveLayerNorm, EmotionEncoder

emotion_enc = EmotionEncoder(d_emotion=128)
adaln = AdaptiveLayerNorm(d_model=768, d_condition=128)

emotion_vec = emotion_enc.encode_name("happy")
x_modulated = adaln(x, emotion_vec)  # 情感深度影响计算
```

## 📊 开发进度

| 模块 | 状态 | 说明 |
|------|------|------|
| interfaces/ | ✅ 完成 | 基类、配置、注册表 |
| io/ | ✅ 骨架完成 | VQ-VAE, 编解码器 |
| brain/ | ✅ 骨架完成 | Mamba, ACT |
| memory/ | ✅ 骨架完成 | FAISS, Cross-Attention |
| modulation/ | ✅ 骨架完成 | AdaLN, FiLM |
| reflection/ | ✅ 骨架完成 | 轨迹记录, 回溯 |
| search/ | ✅ 骨架完成 | Web 搜索, 缓存 |
| pipeline.py | ✅ 骨架完成 | 端到端流水线 |
| 训练代码 | 🔲 待开发 | DataLoader, Trainer |
| 预训练权重 | 🔲 待开发 | 需要大规模训练 |

## 🛣️ 后续计划

### Phase 1: 核心实现 (当前)
- [x] 模块骨架搭建
- [ ] 单元测试覆盖
- [ ] 集成测试

### Phase 2: 功能完善
- [ ] 真实 Tokenizer 集成
- [ ] 训练循环实现
- [ ] 损失函数设计

### Phase 3: 训练验证
- [ ] 小规模数据集验证
- [ ] 消融实验
- [ ] 性能调优

### Phase 4: 扩展
- [ ] 分布式训练支持
- [ ] 多模态扩展
- [ ] 推理优化

## 📚 参考文献

- [VQ-VAE](https://arxiv.org/abs/1711.00937) - 离散潜变量
- [Mamba](https://arxiv.org/abs/2312.00752) - 选择性状态空间模型
- [ACT](https://arxiv.org/abs/1603.08983) - 自适应计算时间
- [AdaLN](https://arxiv.org/abs/2212.09748) - 自适应层归一化

## 📄 License

MIT License
