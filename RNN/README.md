# IMDb 电影评论情感分析循环神经网络项目

这是一个使用 PyTorch 实现的完整 IMDb 电影评论情感分析项目，包含从数据加载、模型定义、训练到推理的完整流程。

## 🎯 项目概述

本项目实现了一个循环神经网络（RNN）来对电影评论进行情感分类（正面/负面），使用经典的 IMDb 数据集进行训练和测试。项目采用模块化设计，代码结构清晰，适合学习和理解循环神经网络在自然语言处理任务中的应用。

## 📁 项目结构

```
sentiment_analysis/
├── model.py                # 循环神经网络模型定义
├── imdb_loader.py          # IMDb 数据加载和预处理
├── train_and_save.py       # 模型训练和保存
├── inference.py            # 模型加载和推理
├── requirements.txt        # 项目依赖
├── README.md               # 项目说明文档
└── saved_models/           # 保存的模型文件
    ├── rnn_model.pth       # PyTorch 模型权重
    └── model_info.json     # 模型信息（包括词汇表）
```

## 🚀 快速开始

### 1. 环境准备

确保你的系统已安装 Python 3.7+ 和 pip，然后安装项目依赖：

```bash
pip install -r requirements.txt
```

### 2. 运行顺序

项目文件需要按以下顺序运行：

#### 步骤 1: 数据加载测试
```bash
python imdb_loader.py
```
- 下载 IMDb 数据集
- 测试数据加载功能
- 可视化样本数据

#### 步骤 2: 模型训练
```bash
python train_and_save.py
```
- 加载 IMDb 数据
- 创建并训练循环神经网络
- 评估模型性能
- 保存训练好的模型和词汇表

#### 步骤 3: 模型推理
```bash
python inference.py
```
- 加载保存的模型和词汇表
- 进行情感预测
- 提供交互式预测模式

## 🧠 模型架构

### 循环神经网络 (RNN) 结构
- **输入**: 文本序列，通过嵌入层转换为向量序列
- **嵌入层**: 将词汇索引映射到密集向量表示
- **RNN 层**: 2层 RNN, 128个隐藏单元
- **全连接层**: 128个神经元 -> 2个神经元 (对应正面/负面)
- **输出层**: 2 个神经元（对应情感类别）

### 训练参数
- **优化器**: Adam
- **学习率**: 0.001
- **损失函数**: CrossEntropyLoss
- **批次大小**: 32
- **训练轮数**: 10

## 📊 功能特性

### 数据加载 (`imdb_loader.py`)
- 自动下载 IMDb 数据集
- 数据预处理和分析
- 样本可视化

### 模型定义 (`model.py`)
- 包含嵌入层的RNN架构
- 内置训练方法
- 预测和评估功能
- 模型保存和加载

### 训练流程 (`train_and_save.py`)
- 完整的训练流程
- 构建词汇表
- 性能评估和分析
- 模型和词汇表自动保存

### 推理应用 (`inference.py`)
- 模型和词汇表加载
- 对新文本进行情感预测
- 交互式测试模式

## 🔧 技术实现

### 核心技术
- **PyTorch**: 深度学习框架
- **NumPy**: 数值计算
- **Matplotlib**: 数据可视化
- **datasets**: Hugging Face 数据集加载库
- **tqdm**: 进度条库

### 关键特性
- 模块化设计，代码复用性高
- 详细的训练过程监控
- 灵活的模型配置选项

## 📈 性能表现

在 IMDb 测试集上，模型通常能达到：
- **训练准确率**: 95%+
- **测试准确率**: 85%+
- **训练时间**: 约 10-20 分钟（取决于硬件配置）

## 🛠️ 自定义配置

### 调整训练参数
在 `train_and_save.py` 中修改训练相关的超参数：

```python
learning_rate = 0.001
epochs = 10
batch_size = 32
```

## 🐛 常见问题

### Q: 运行时提示 "No module named 'torch'"
**A**: 请确保已正确安装 PyTorch：
```bash
pip install torch
```

### Q: 训练过程中内存不足
**A**: 可以减少批次大小或词汇表大小。

### Q: 模型保存失败
**A**: 确保 `saved_models/` 目录存在写入权限，或手动创建该目录。

## 📚 学习资源

- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [Hugging Face Datasets 文档](https://huggingface.co/docs/datasets/)
- [循环神经网络 (RNN) 详解](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目！

## 📄 许可证

本项目采用 MIT 许可证。

**注意**: 首次运行时会自动下载 IMDb 数据集，请确保网络连接正常。
