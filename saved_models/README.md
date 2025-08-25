# 模型保存目录说明

本目录用于统一保存所有版本的情感分析模型文件。每个模型版本都有独立的子文件夹：

## 目录结构

```
saved_models/
├── bert/           # BERT模型文件
├── lstm/           # LSTM模型文件
├── rnn/            # RNN模型文件
├── transformer/    # Transformer模型文件
└── README.md       # 本说明文件
```

## 模型文件说明

### BERT模型
- 位置：`saved_models/bert/`
- 文件：`bert_sentiment_model_*.pth`
- 信息：`bert_model_info_*.json`

### LSTM模型
- 位置：`saved_models/lstm/`
- 文件：`model.pth`
- 结构：`model_structure.json`
- 词汇表：`vocab_stoi.json`, `vocab_itos.json`

### RNN模型
- 位置：`saved_models/rnn/`
- 文件：`rnn_model.pth`
- 信息：`model_info.json`
- 词汇表：`vocab.json`

### Transformer模型
- 位置：`saved_models/transformer/`
- 文件：`model.pth`
- 结构：`model_structure.json`
- 词汇表：`vocab_stoi.json`, `vocab_itos.json`

## 数据集说明

所有版本的数据集都统一保存在根目录的 `data/` 文件夹中，避免重复下载。

## 使用方式

各个版本的训练和推理脚本已经更新为使用统一的路径：
- 数据集：从 `../data/` 加载
- 模型：保存到对应的 `../saved_models/{version}/` 目录

训练完成后，模型文件会自动保存到对应的子目录中。