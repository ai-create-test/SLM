# 🧠 LLM Experimental Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个实验性的大语言模型（LLM）开发框架，专注于模块化架构设计和前沿技术实现。
备注：本项目完全使用Google Antigravity完成，没有任何人工介入。

## 📖 项目背景

### 设计动机

本项目旨在构建一个**高度模块化、可扩展**的 LLM 实验平台，用于探索和验证以下核心技术：

1. **问题导向检索（Query-Focused Retrieval）** - 以用户问题为主导的增强检索阅览方式
2. **情感与场景注入（Context Injection）** - 通过可学习门控机制将情感/场景信息融入语义表示
3. **长文本处理（Long-Text Processing）** - 语义完整的分块策略与压缩机制

### 技术选型

| 组件 | 技术方案 | 说明 |
|------|----------|------|
| 分词器 | tiktoken (BPE) | OpenAI 生产级实现，支持 GPT-4 编码 |
| 位置编码 | RoPE | 旋转位置编码，更好的长度外推能力 |
| 归一化 | RMSNorm | 比 LayerNorm 更高效，被 LLaMA 采用 |
| 注意力权重 | BM25 / TF-IDF | 工业级检索算法 |

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        Main Entry                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         app/                                │
├─────────────┬─────────────┬─────────────┬─────────────┬────┤
│   core/     │   memory/   │ retrieval/  │ reflection/ │... │
│ ─────────── │ ─────────── │ ─────────── │ ─────────── │    │
│ • Tokenizer │ • Embedding │ • RAG       │ • Self-     │    │
│ • BPE       │ • RoPE      │ • Query-    │   Reflection│    │
│ • Attention │ • Chunking  │   Focused   │             │    │
│   Weights   │ • Fusion    │             │             │    │
└─────────────┴─────────────┴─────────────┴─────────────┴────┘
```

## ✨ 核心特性

### 1. 核心分词模块 (`app/core/`)

- **BPETokenizer**: 基于 tiktoken 的工业级 BPE 分词器
- **问题导向注意力**: 支持 BM25、TF-IDF、关键词匹配等多种策略
- **工厂模式**: 统一的分词器创建接口，支持动态注册

```python
from app.core import get_tokenizer

# 使用 GPT-4 编码
tokenizer = get_tokenizer("gpt4")
result = tokenizer.encode("Hello, world!", query="greeting")
print(result.attention_weights)  # 问题导向注意力权重
```

### 2. 记忆模块 (`app/memory/`)

- **TokenEmbedding**: Token → 稠密向量转换
- **RoPE**: 旋转位置编码，支持长度外推
- **TextChunker**: 语义感知的文本分块
- **ContextFusion**: 情感/场景可学习门控注入

```python
from app.memory import create_context_aware_embedding, EMOTION_IDS

# 创建上下文感知嵌入层
embedding = create_context_aware_embedding(preset='base')

# 注入情感
import torch
token_ids = torch.tensor([[100, 200, 300]])
output = embedding(token_ids, emotion_id=EMOTION_IDS["happy"])
```

### 3. 长文本编码器 (`app/memory/MemoryEncoder`)

```python
from app.memory import create_memory_encoder

encoder = create_memory_encoder(preset='base')
result = encoder.encode_with_chunks("非常长的文本...")

print(f"分块数: {result.num_chunks}")
print(f"嵌入形状: {result.embeddings.shape}")  # [chunks, seq_len, d_model]
```

## 📁 项目结构

```
gemini_test/
├── main.py                 # 入口文件
├── requirements.txt        # 依赖清单
├── README.md              # 项目文档
├── .gitignore             # Git 忽略规则
│
├── app/                   # 核心业务逻辑
│   ├── core/              # 分词器与注意力模块
│   │   ├── tokenizer_base.py      # 抽象基类
│   │   ├── bpe_tokenizer.py       # BPE 实现
│   │   ├── tokenizer_attention.py # 问题导向注意力
│   │   └── tokenizer_factory.py   # 工厂模式
│   │
│   ├── memory/            # 嵌入与记忆模块
│   │   ├── embeddings.py     # Token嵌入 + RoPE
│   │   ├── text_chunker.py   # 文本分块
│   │   ├── fusion.py         # 情感/场景融合
│   │   └── memory_encoder.py # 整合编码器
│   │
│   ├── retrieval/         # 问题导向检索 (待实现)
│   ├── reflection/        # 自我回溯 (待实现)
│   ├── search/            # 联网搜索 (待实现)
│   └── utils/             # 工具函数 (待实现)
│
├── tests/                 # 单元测试
│   └── test_tokenizer.py  # 分词器测试
│
├── test_memory.py         # Memory 模块验收测试
└── test_fusion.py         # Context Fusion 验收测试
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+

### 安装

```bash
# 克隆项目
git clone https://github.com/your-username/gemini_test.git
cd gemini_test

# 安装依赖
pip install -r requirements.txt
```

### 运行测试

```bash
# 运行所有 pytest 测试
pytest tests/ -v

# 运行验收测试
python test_memory.py
python test_fusion.py
```

## 📊 开发状态

| 模块 | 状态 | 说明 |
|------|------|------|
| `core/tokenizer` | ✅ 完成 | BPE 分词器 + 问题导向注意力 |
| `memory/embeddings` | ✅ 完成 | Token 嵌入 + RoPE 位置编码 |
| `memory/chunker` | ✅ 完成 | 语义感知文本分块 |
| `memory/fusion` | ✅ 完成 | 情感/场景门控注入 |
| `retrieval/` | 🚧 规划中 | 问题导向增强检索 |
| `reflection/` | 🚧 规划中 | 模型自我回溯设计 |
| `search/` | 🚧 规划中 | 实时网络搜索 |

## 🔧 配置选项

### 预设配置

| 预设 | vocab_size | d_model | max_seq_len |
|------|------------|---------|-------------|
| small | 110,000 | 256 | 256 |
| base | 110,000 | 768 | 512 |
| large | 150,000 | 1024 | 1024 |

### 自定义配置

```python
from app.memory import MemoryEncoder

encoder = MemoryEncoder(
    vocab_size=100000,
    d_model=512,
    max_seq_len=1024,
    position_encoding='rope',
    dropout=0.1,
)
```

## 📝 API 参考

### 主要类

| 类名 | 模块 | 说明 |
|------|------|------|
| `BPETokenizer` | `app.core` | BPE 分词器 |
| `TokenizerFactory` | `app.core` | 分词器工厂 |
| `CombinedEmbedding` | `app.memory` | 组合嵌入层 |
| `ContextAwareEmbedding` | `app.memory` | 上下文感知嵌入 |
| `MemoryEncoder` | `app.memory` | 整合编码器 |
| `TextChunker` | `app.memory` | 文本分块器 |

### 便捷函数

```python
from app.core import get_tokenizer
from app.memory import create_memory_encoder, create_context_aware_embedding
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 License

MIT License - 详见 [LICENSE](LICENSE) 文件
