# AMHVQ+ Architecture Documentation

> Adaptive Matryoshka Hierarchical VQ + 语义-结构-符号三通道融合

## 概述

AMHVQ+ 是一种先进的段落级语义编码架构，通过三通道并行处理实现高压缩比和精确保真的统一。

## 核心架构

```
输入文本
    ↓
┌─────────────────────────────────────────────────────┐
│              Scene-Aware Router (场景路由)           │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────┬─────────────────┬─────────────────────┐
│  通道 1     │  通道 2          │  通道 3             │
│  SEMANTIC   │  STRUCTURAL      │  SYMBOLIC           │
├─────────────┼─────────────────┼─────────────────────┤
│ z_global    │ AST/结构解析     │ 关键token锚定       │
│ z_chunks    │ GraphMemory存储  │ 精确位置保留        │
│ Matryoshka  │ 骨架提取         │ slot_id映射         │
└─────────────┴─────────────────┴─────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│                  UnifiedLatent                       │
│  { semantic, structure, symbols, scene, precision } │
└─────────────────────────────────────────────────────┘
```

## 模块清单

### IO 层
| 模块 | 文件 | 描述 |
|------|------|------|
| ResidualVQ | `app/io/residual_vq.py` | 多层残差向量量化 |
| SemanticChunker | `app/io/semantic_chunker.py` | 语义边界分块 |
| MatryoshkaProjection | `app/io/matryoshka.py` | 嵌套维度投影 |
| StructureParser | `app/io/structure_parser.py` | AST/句法解析 |
| SymbolAnchorEncoder | `app/io/symbol_anchor.py` | 符号锚点编码 |
| UnifiedEncoder | `app/io/unified_encoder.py` | 三通道统一编码 |
| UnifiedDecoder | `app/io/unified_decoder.py` | 结构引导解码 |

### 接口层
| 模块 | 文件 | 描述 |
|------|------|------|
| HierarchicalLatent | `app/interfaces/base_module.py` | 层次化潜向量 |
| UnifiedLatent | `app/interfaces/unified_latent.py` | 三通道统一潜向量 |
| latent_utils | `app/interfaces/latent_utils.py` | 潜向量转换工具 |

### Brain 层
| 模块 | 文件 | 描述 |
|------|------|------|
| HierarchicalDynamics | `app/brain/hierarchical_dynamics.py` | 层次化动力学 |
| SetEncoder | 同上 | 集合编码器 |

### 训练
| 模块 | 文件 | 描述 |
|------|------|------|
| UnifiedTrainingStage | `app/training/unified_training_stages.py` | 三通道联合训练 |
| CurriculumScheduler | 同上 | 课程学习调度 |

### 模型
| 模块 | 文件 | 描述 |
|------|------|------|
| UnifiedNeuralFlowModel | `app/model/unified_model.py` | 统一模型封装 |

### 推理
| 模块 | 文件 | 描述 |
|------|------|------|
| AMHVQInference | `app/inference/amhvq_inference.py` | 推理接口 |

## 场景配置

| 场景 | 语义位 | 结构 | 符号 |
|------|--------|------|------|
| chat | 256 | ❌ | ❌ |
| coding | 512 | ✅ | ✅ |
| technical | 512 | ✅ | ❌ |
| creative | 256 | ❌ | ❌ |

## 使用示例

```python
from app.model.unified_model import UnifiedNeuralFlowModel

# 创建模型
model = UnifiedNeuralFlowModel.from_preset("coding")

# 编码
latent = model.encode("def hello(): pass", scene="coding")

# 解码
text = model.decode(latent)

# 端到端
output = model.generate("def add(a, b):", max_length=100)
```

## 配置预设

- `configs/amhvq_base.yaml` - 基础配置
- `configs/amhvq_coding.yaml` - 代码专用
- `configs/amhvq_small.yaml` - 小型高效
