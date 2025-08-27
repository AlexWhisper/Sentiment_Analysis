# IMDB 情感分析 Transformer 项目

这是一个使用 PyTorch 实现的完整 IMDB 电影评论情感分析项目，采用 Transformer 神经网络进行二分类任务。项目包含从数据加载、模型定义、训练到推理的完整流程。

## 项目概述

本项目实现了一个基于 Transformer 的情感分析模型，用于判断电影评论的情感倾向（正面/负面）。使用经典的 IMDB 数据集进行训练和测试，项目采用模块化设计，代码结构清晰，适合学习和理解 Transformer 神经网络在自然语言处理中的应用。

## 项目结构

```
Transformer/
├── model.py                # Transformer 模型定义和相关功能
├── data_loader.py          # 数据加载、词汇表构建和预处理
├── train_and_save.py       # 模型训练和保存
├── inference.py            # 模型加载和推理
├── requirements.txt        # 项目依赖
├── README.md               # 项目说明文档
└── ../saved_models/        # 保存的模型文件（训练后生成，位于项目根目录）
    └── transformer/        # Transformer模型专用目录
        ├── model.pth           # PyTorch 模型权重
        ├── model_structure.json # 模型结构信息
        └── vocab.json          # 词汇表文件

```

## 快速开始

### 1. 环境准备

确保你的系统已安装 Python 3.7+ 和 pip，然后安装项目依赖：

```bash
pip install -r requirements.txt
```

### 2. 安装 spaCy 英语模型

```bash
python -m spacy download en_core_web_sm
```

### 3. 运行顺序

项目文件需要按以下顺序运行：

#### 步骤 1: 数据加载测试（可选）
```bash
python data_loader.py
```
- 测试数据加载功能
- 下载 IMDB 数据集
- 构建词汇表

#### 步骤 2: 模型训练
```bash
python train_and_save.py
```
- 加载 IMDB 数据集
- 创建并训练 Transformer 模型
- 评估模型性能
- 保存训练好的模型


#### 步骤 3: 模型推理
```bash
python inference.py
```
- 加载保存的模型
- 进行情感分析预测
- 示例测试和准确率统计

## 模型架构

### Transformer 网络结构
- **词嵌入层**: 将词汇转换为密集向量表示
- **位置编码**: 为序列中的每个位置添加位置信息
- **Transformer编码器**: 多层自注意力机制，捕获序列中的长距离依赖关系
- **全局平均池化**: 对序列进行池化操作
- **全连接层 1**: embedding_dim -> hidden_dim//2 神经元，ReLU 激活
- **全连接层 2**: hidden_dim//2 -> 1 神经元（二分类输出）
- **输出**: Sigmoid 激活函数，输出 0-1 之间的概率

### 模型参数
- **词汇表大小**: 25,000（可配置）
- **词嵌入维度**: 128（必须能被注意力头数整除）
- **Transformer 隐藏层维度**: 256
- **注意力头数**: 8
- **Transformer 层数**: 2
- **最大序列长度**: 512
- **优化器**: Adam
- **学习率**: 0.001
- **损失函数**: BCEWithLogitsLoss
- **批次大小**: 32
- **训练轮数**: 100（可配置）

## 功能特性

### 数据处理 (`data_loader.py`)
- 自动下载 IMDB 数据集
- 使用 spaCy 进行文本分词
- 构建词汇表和词汇映射
- 数据预处理和批处理
- 序列填充和截断

### 模型定义 (`model.py`)
- Transformer 网络架构定义
- 位置编码实现
- 模型保存和加载功能
- 准确率计算函数
- 模型结构信息管理

### 训练流程 (`train_and_save.py`)
- 完整的训练和验证流程
- 训练过程可视化（进度条）
- 最佳模型自动保存
- 训练历史图表生成
- 性能评估和分析

### 推理应用 (`inference.py`)
- 模型加载和恢复
- 单句情感预测
- 预测结果展示
- 示例测试和准确率统计

## 技术实现

### 核心技术
- **PyTorch**: 深度学习框架
- **Hugging Face Datasets**: 数据集加载
- **spaCy**: 自然语言处理和分词
- **NumPy**: 数值计算
- **Matplotlib**: 数据可视化
- **tqdm**: 进度条显示

### 关键特性
- 模块化设计，代码复用性高
- 完整的错误处理和异常管理
- 详细的训练过程监控
- 灵活的模型配置选项
- 示例预测测试

## 性能表现

在标准 IMDB 测试集上，模型通常能达到：
- **训练准确率**: 85-90%
- **验证准确率**: 82-87%
- **测试准确率**: 82-87%
- **训练时间**: 约 10-30 分钟（取决于硬件配置和数据大小）

## 自定义配置

### 调整训练参数
在 `train_and_save.py` 的 `main()` 函数中修改：

```python
EPOCHS = 100               # 训练轮数
LEARNING_RATE = 0.001      # 学习率
BATCH_SIZE = 32            # 批次大小
EMBEDDING_DIM = 128        # 词嵌入维度（必须能被注意力头数整除）
HIDDEN_DIM = 256           # Transformer隐藏层维度
MAX_VOCAB_SIZE = 25000     # 词汇表大小
```

### 更改数据集配置
在 `data_loader.py` 的 `prepare_data()` 函数中调整：

```python
data_components = prepare_data(
    max_vocab_size=25000,   # 词汇表大小
    batch_size=32,          # 批次大小
    test_size=0.2,          # 验证集比例
    seed=22                 # 随机种子
)
```

## 常见问题

### Q: 运行时提示 "No module named 'torch'"
**A**: 请确保已正确安装 PyTorch：
```bash
pip install torch torchvision
```

### Q: 提示 "Can't find model 'en_core_web_sm'"
**A**: 请安装 spaCy 英语模型：
```bash
python -m spacy download en_core_web_sm
```

### Q: 训练过程中内存不足
**A**: 可以减少批次大小或词汇表大小：
```python
BATCH_SIZE = 16           # 减小批次大小
MAX_VOCAB_SIZE = 10000    # 减少词汇表大小
```

### Q: 模型保存失败
**A**: 确保项目根目录下的 `saved_models/transformer/` 目录存在写入权限，或手动创建该目录。

### Q: 推理时提示找不到模型文件
**A**: 请先运行 `train_and_save.py` 训练并保存模型。

## 学习资源

- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [IMDB 数据集介绍](https://ai.stanford.edu/~amaas/data/sentiment/)
- [Transformer 网络详解](https://arxiv.org/abs/1706.03762)
- [Hugging Face Datasets 文档](https://huggingface.co/docs/datasets/)
- [spaCy 官方文档](https://spacy.io/)

## 使用示例

### 训练模型
```bash
# 训练模型（大约需要10-30分钟）
python train_and_save.py
```

### 示例预测测试
```bash
# 运行推理脚本
python inference.py

# 会自动测试多个示例句子，输出如下：
句子: 'This movie is absolutely fantastic!'
预测情感: positive
预测概率: 0.8234
真实情感: positive
预测正确: ✓
```

### 批量测试
运行 `inference.py` 会自动进行示例测试，展示模型在各种句子上的表现。

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目！

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

---

**注意**: 
- 首次运行时会自动下载 IMDB 数据集（约 80MB），请确保网络连接正常
- 训练过程需要一定时间，建议在有 GPU 的环境下运行以加速训练
- 模型文件会保存在项目根目录的 `saved_models/transformer/` 目录下，请确保有足够的磁盘空间