# BERT情感分析模型

本项目实现了基于BERT的情感分析模型，通过微调预训练的BERT模型来进行二分类情感分析任务。

## 项目特点

- 🤖 **预训练模型微调**: 基于`bert-base-uncased`预训练模型进行微调
- 📊 **完整训练流程**: 包含数据加载、模型训练、验证和测试
- 🎯 **高性能**: 利用BERT的强大语言理解能力
- 🔧 **易于使用**: 提供交互式推理功能
- 💾 **模型保存**: 自动保存最佳模型和训练信息

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 主要依赖包

- `torch`: PyTorch深度学习框架
- `transformers`: Hugging Face Transformers库
- `datasets`: 数据集加载库
- `numpy`: 数值计算
- `tqdm`: 进度条

## 项目结构

```
BERT/
├── model.py              # BERT模型定义
├── data_loader.py         # 数据加载和预处理
├── train_and_save.py      # 训练脚本
├── inference.py           # 推理脚本
├── requirements.txt       # 依赖文件
├── README.md             # 项目说明
└── saved_models/         # 保存的模型文件（训练后生成）
    ├── bert_sentiment_model_*.pth
    └── bert_model_info_*.json
```

## 使用方法

### 1. 训练模型

```bash
python train_and_save.py
```

训练参数说明：
- **EPOCHS**: 3（BERT微调通常只需要少量epoch）
- **LEARNING_RATE**: 2e-5（BERT推荐学习率）
- **BATCH_SIZE**: 32（训练脚本中设置，可根据GPU内存调整）
- **MAX_LENGTH**: 256（序列最大长度）
- **MODEL_NAME**: 'bert-base-uncased'

### 2. 模型推理

```bash
python inference.py
```

这将自动查找最新的训练模型并启动交互式预测模式。在交互式模式下，您可以：
- 输入文本进行情感分析
- 查看预测结果和置信度
- 输入 'quit' 或 'exit' 退出程序

## 模型架构

### BERT微调架构

```
输入文本
    ↓
BERT Tokenizer (分词)
    ↓
BERT Encoder (预训练)
    ↓
[CLS] Token 表示
    ↓
Dropout Layer
    ↓
Linear Classification Layer
    ↓
情感分类结果 (正面/负面)
```

### 关键特性

1. **预训练权重**: 使用BERT预训练权重初始化
2. **微调策略**: 对整个模型进行端到端微调
3. **学习率调度**: 使用线性预热和衰减
4. **梯度裁剪**: 防止梯度爆炸
5. **权重衰减**: L2正则化防止过拟合

## 数据处理

### 数据集
- **数据源**: IMDB电影评论数据集
- **训练集**: 22,500条评论（从25,000条中分出验证集）
- **验证集**: 2,500条评论
- **测试集**: 25,000条评论
- **标签**: 0（负面）、1（正面）

### 预处理步骤
1. **分词**: 使用BERT tokenizer
2. **特殊标记**: 添加[CLS]和[SEP]标记
3. **填充/截断**: 统一序列长度
4. **注意力掩码**: 标识有效token位置

## 训练过程

### 训练配置
- **优化器**: AdamW（适合Transformer模型）
- **损失函数**: CrossEntropyLoss
- **学习率调度**: 线性预热 + 线性衰减
- **预热比例**: 10%的训练步数
- **权重衰减**: 0.01
- **梯度裁剪**: 最大范数1.0

### 训练监控
- 实时显示训练/验证损失和准确率
- 自动保存最佳验证准确率模型
- 保存详细的模型信息和配置

## 性能表现

### 预期性能
- **验证准确率**: ~92-95%
- **测试准确率**: ~90-93%
- **训练时间**: 约30-60分钟（取决于硬件）

### 硬件要求
- **推荐**: NVIDIA GPU（8GB+ 显存）
- **最低**: CPU（训练较慢）
- **内存**: 8GB+ RAM

## 文件说明

### 核心文件

#### `model.py`
- `BERTSentimentAnalyzer`: 主模型类
- `binary_accuracy`: 准确率计算函数
- `save_model_info`: 模型信息保存

#### `data_loader.py`
- `IMDBDataset`: 自定义数据集类
- `prepare_data`: 数据准备函数

#### `train_and_save.py`
- `train_epoch`: 训练一个epoch
- `evaluate_epoch`: 评估一个epoch
- `main`: 主训练流程

#### `inference.py`
- `BERTInference`: 推理类
- `predict_single`: 单文本预测
- `interactive_predict`: 交互式预测

## 使用示例

### Python代码示例

```python
from model import BERTSentimentAnalyzer
from inference import BERTInference

# 创建推理器
inferencer = BERTInference(model_path='saved_models/bert_sentiment_model_latest.pth')
inferencer.load_model()

# 单文本预测
result = inferencer.predict_single("This movie is fantastic!")
print(f"情感: {result['sentiment']}, 概率: {result['probability']:.4f}")

# 多个文本预测
texts = ["I love this movie!", "This is terrible."]
for i, text in enumerate(texts):
    result = inferencer.predict_single(text)
    print(f"文本{i+1}: {result['sentiment']} ({result['probability']:.4f})")
```

## 常见问题

### Q1: 首次运行时下载模型很慢怎么办？
A: BERT模型较大（约400MB），首次下载需要时间。可以：
- 使用国内镜像源
- 手动下载模型文件
- 使用较小的模型如`distilbert-base-uncased`

### Q2: GPU内存不足怎么办？
A: 可以：
- 减小`BATCH_SIZE`（如改为8或4）
- 减小`MAX_LENGTH`（如改为128）
- 使用梯度累积
- 使用CPU训练（较慢）

### Q3: 如何提高模型性能？
A: 可以尝试：
- 增加训练epoch数
- 调整学习率
- 使用更大的BERT模型（如`bert-large`）
- 数据增强技术
- 集成多个模型

### Q4: 如何处理中文文本？
A: 需要：
- 使用中文BERT模型（如`bert-base-chinese`）
- 相应调整tokenizer
- 准备中文情感分析数据集

## 扩展功能

### 1. 多分类情感分析
修改`num_classes`参数支持多类别分类（如正面、负面、中性）

### 2. 其他预训练模型
可以替换为其他模型：
- RoBERTa: `roberta-base`
- DistilBERT: `distilbert-base-uncased`
- ALBERT: `albert-base-v2`

### 3. 领域适应
在特定领域数据上进一步微调模型

## 参考资料

- [BERT论文](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers文档](https://huggingface.co/docs/transformers/)
- [PyTorch官方文档](https://pytorch.org/docs/)
- [IMDB数据集](https://huggingface.co/datasets/imdb)

## 许可证

本项目仅供学习和研究使用。