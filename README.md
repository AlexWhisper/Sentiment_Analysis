# 深度学习进阶项目：情感分析

这是一个专为研0师弟师妹设计的深度学习进阶项目，通过IMDB电影评论情感分析任务，深入学习自然语言处理中的各种深度学习方法。项目包含多个版本的实现，从传统的LSTM到现代的BERT和Transformer，帮助学习者全面理解NLP中的深度学习技术。

## 项目概述

本项目使用经典的IMDB电影评论数据集，实现了多种不同的神经网络架构进行情感分析（正面/负面二分类）：

- **LSTM**: 使用长短期记忆网络处理序列文本数据
- **BERT**: 基于预训练BERT模型进行微调的情感分析
- **Transformer**: 从零实现Transformer架构进行情感分析
- **文本预处理教程**: 详细的文本预处理基础教程

## 学习路径建议

### 第一阶段：理解文本预处理
1. 学习 `text_preprocessing_tutorial`，掌握文本清洗、分词、词向量等基础概念
2. 理解文本数据如何转换为模型可处理的数值格式
3. 掌握词汇表构建和序列填充技术

### 第二阶段：序列模型实践
1. 从 `LSTM` 开始，理解循环神经网络处理序列数据的原理
2. 学习LSTM的门控机制和长期依赖建模能力
3. 掌握序列到标签的分类任务

### 第三阶段：注意力机制
1. 学习 `Transformer`，理解自注意力机制的工作原理
2. 掌握位置编码和多头注意力的实现
3. 理解Transformer在NLP任务中的优势

### 第四阶段：预训练模型微调
1. 学习 `BERT`，了解预训练语言模型的强大能力
2. 掌握模型微调（Fine-tuning）的技术
3. 理解迁移学习在NLP中的应用

## 项目结构

```
Sentiment_Analysis/
├── README.md                           # 项目总体说明
├── requirements.txt                     # 统一依赖文件
├── .gitignore                          # Git忽略文件
│
├── text_preprocessing_tutorial/         # 文本预处理教程
│   └── text_preprocessing_basics.py    # 文本预处理基础教程
│
├── LSTM/                               # LSTM实现
│   ├── README.md                       # 详细说明文档
│   ├── model.py                        # LSTM模型定义
│   ├── data_loader.py                  # 数据加载和预处理
│   ├── train_and_save.py               # 训练和保存脚本
│   ├── inference.py                    # 推理脚本
│   └── requirements.txt                # 版本特定依赖
│
├── Transformer/                        # Transformer实现
│   ├── README.md                       # 详细说明文档
│   ├── model.py                        # Transformer模型定义
│   ├── data_loader.py                  # 数据加载和预处理
│   ├── train_and_save.py               # 训练和保存脚本
│   ├── inference.py                    # 推理脚本
│   └── requirements.txt                # 版本特定依赖
│
├── BERT/                               # BERT实现
│   ├── README.md                       # 详细说明文档
│   ├── model.py                        # BERT模型定义
│   ├── data_loader.py                  # 数据加载和预处理
│   ├── train_and_save.py               # 训练和保存脚本
│   ├── inference.py                    # 推理脚本
│   └── requirements.txt                # 版本特定依赖
│
├── data/                               # 数据存储目录
└── saved_models/                       # 保存的模型文件
    ├── lstm/                           # LSTM模型文件
    ├── transformer/                    # Transformer模型文件
    └── bert/                           # BERT模型文件
```

## 快速开始

### 环境准备

确保系统已安装Python 3.10，然后安装项目依赖：

```bash
# 安装基础依赖（适用于所有版本）
pip install -r requirements.txt

# 安装spaCy英语模型（用于文本预处理）
python -m spacy download en_core_web_sm

# 或者根据需要安装特定版本的依赖
cd LSTM && pip install -r requirements.txt
cd BERT && pip install -r requirements.txt
```

### 运行示例

每个版本都包含完整的训练和推理流程：

```bash
# 以LSTM版本为例
cd LSTM

# 1. 数据加载测试
python data_loader.py

# 2. 模型训练
python train_and_save.py

# 3. 模型推理
python inference.py
```

### 文本预处理教程

```bash
# 学习文本预处理基础
cd text_preprocessing_tutorial
python text_preprocessing_basics.py
```

## 各版本特点对比

| 版本 | 实现方式 | 难度 | 学习重点 | 预期准确率 | 训练时间 |
|------|----------|------|----------|------------|----------|
| LSTM | PyTorch框架 | ⭐⭐⭐ | 序列建模、门控机制 | 85%+ | 中等 |
| Transformer | PyTorch框架 | ⭐⭐⭐⭐ | 注意力机制、位置编码 | 88%+ | 较长 |
| BERT | 预训练模型微调 | ⭐⭐⭐⭐⭐ | 预训练模型、迁移学习 | 92%+ | 最长 |
| 文本预处理 | 基础教程 | ⭐ | 文本处理基础概念 | - | - |

## 技术栈

- **Python 3.10**: 编程语言
- **PyTorch**: 深度学习框架
- **Transformers**: Hugging Face预训练模型库
- **Datasets**: 数据集加载库
- **spaCy**: 自然语言处理库
- **Matplotlib**: 数据可视化
- **NumPy**: 数值计算
- **tqdm**: 进度条显示

## 数据集介绍

本项目使用IMDB电影评论数据集：
- **训练集**: 25,000条电影评论
- **测试集**: 25,000条电影评论
- **标签**: 正面情感(1) / 负面情感(0)
- **语言**: 英文
- **领域**: 电影评论

## 学习资源

- [IMDB数据集介绍](https://ai.stanford.edu/~amaas/data/sentiment/)
- [PyTorch官方教程](https://pytorch.org/tutorials/)
- [Hugging Face Transformers文档](https://huggingface.co/docs/transformers/)
- [BERT论文](https://arxiv.org/abs/1810.04805)
- [Transformer论文](https://arxiv.org/abs/1706.03762)
- [自然语言处理基础](https://web.stanford.edu/~jurafsky/slp3/)

## 常见问题

### Q: 应该从哪个版本开始学习？
A: 建议按照学习路径循序渐进：文本预处理教程 → LSTM → Transformer → BERT

### Q: 训练时间太长怎么办？
A: 可以减少训练轮数、减小批次大小或使用更小的数据集子集来加快训练速度

### Q: 如何提高模型准确率？
A: 可以尝试调整学习率、增加训练轮数、修改网络结构、使用数据增强或更好的文本预处理技术

### Q: BERT模型下载失败怎么办？
A: 可以手动下载模型文件，或使用国内镜像源，也可以使用较小的BERT模型如`bert-base-uncased`

### Q: 遇到内存不足问题怎么办？
A: 可以减小批次大小、使用梯度累积、或者使用较小的模型和序列长度

## 性能优化建议

1. **GPU加速**: 使用CUDA加速训练过程
2. **批处理**: 合理设置batch_size平衡内存和训练效率
3. **学习率调度**: 使用学习率衰减策略
4. **早停机制**: 防止过拟合，节省训练时间
5. **模型剪枝**: 对于部署场景，可以考虑模型压缩技术

## 贡献指南

欢迎提交Issue和Pull Request来改进项目！请确保：
1. 代码风格一致
2. 添加必要的注释
3. 更新相关文档
4. 测试代码功能
5. 遵循项目的模块化设计原则

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

---

**注意**: 
- 首次运行时会自动下载IMDB数据集和预训练模型，请确保网络连接正常
- BERT模型较大，建议在有足够存储空间和内存的环境下运行
- 建议使用GPU进行训练以获得更好的性能