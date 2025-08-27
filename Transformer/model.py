# Copyright (c) 2025 NIS Lab, Hebei Normal University
# Author: Wang Minghu <wangminghu41@163.com>
# Internal Use Only. Unauthorized distribution is strictly prohibited.
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List
import json
import os
import math

class PositionalEncoding(nn.Module):
    """
    位置编码模块
    """
    def __init__(self, d_model: int, max_seq_len: int = 512):
        super().__init__()
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerSentimentAnalyzer(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int, pad_idx: int, learning_rate: float = 0.001, num_heads: int = 8, num_layers: int = 2, max_seq_len: int = 512):
        """
        初始化Transformer情感分析模型
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            hidden_dim: Transformer隐藏层维度
            output_dim: 输出维度（通常为1，用于二分类）
            pad_idx: 填充符号的索引
            learning_rate: 学习率
            num_heads: 多头注意力的头数
            num_layers: Transformer层数
            max_seq_len: 最大序列长度
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pad_idx = pad_idx
        self.learning_rate = learning_rate
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # 模型层定义
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_len)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.fc1 = nn.Linear(embedding_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        
        # 优化器和损失函数
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 初始化填充符号的嵌入为零向量
        self.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            text: 输入文本张量，形状为 (seq_len, batch_size) 或 (batch_size, seq_len)
            
        Returns:
            预测输出，形状为 (batch_size, output_dim)
        """
        # 如果输入是 (seq_len, batch_size)，转换为 (batch_size, seq_len)
        if text.dim() == 2 and text.size(0) != text.size(1):
            if text.size(0) > text.size(1):  # 假设seq_len > batch_size
                text = text.transpose(0, 1)
        
        # 创建padding mask
        padding_mask = (text == self.pad_idx)
        
        # 词嵌入和位置编码
        embedded = self.embedding(text) * math.sqrt(self.embedding_dim)
        embedded = self.pos_encoding(embedded.transpose(0, 1)).transpose(0, 1)
        
        # Transformer编码
        transformer_output = self.transformer(embedded, src_key_padding_mask=padding_mask)
        
        # 全局平均池化（忽略padding位置）
        mask = (~padding_mask).float().unsqueeze(-1)
        pooled = (transformer_output * mask).sum(dim=1) / mask.sum(dim=1)
        
        # 分类
        predictions = self.dropout(self.relu(self.fc1(pooled)))
        predictions = self.fc2(predictions)
        return predictions
    
    def get_model_info(self) -> Dict:
        """
        获取模型结构信息
        
        Returns:
            包含模型参数的字典
        """
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'pad_idx': self.pad_idx,
            'learning_rate': self.learning_rate,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'max_seq_len': self.max_seq_len
        }

def binary_accuracy(preds: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    计算二分类准确率
    
    Args:
        preds: 模型预测值
        y: 真实标签
        
    Returns:
        准确率
    """
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    return correct.sum() / len(correct)

def save_model_info(model: TransformerSentimentAnalyzer, vocab_stoi: Dict[str, int], vocab_itos: List[str], model_dir: str = None):
    """
    保存模型结构信息和词汇表
    
    Args:
        model: Transformer模型实例
        vocab_stoi: 词汇表（字符串到索引的映射）
        vocab_itos: 词汇表（索引到字符串的映射）
        model_dir: 模型保存目录
    """
    # 设置默认保存目录到根目录的saved_models/transformer文件夹
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'transformer')
    
    # 创建模型目录
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 保存模型结构信息
    model_info = model.get_model_info()
    with open(os.path.join(model_dir, 'model_structure.json'), 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    # 保存词汇表
    vocab_info = {
        'vocab_stoi': vocab_stoi,
        'vocab_itos': vocab_itos
    }
    with open(os.path.join(model_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(vocab_info, f, indent=2, ensure_ascii=False)
    
    # 保存PyTorch模型状态
    model_path = os.path.join(model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    
    print(f"模型已保存到 {model_dir}/")
    print(f"   - 模型结构: model_structure.json")
    print(f"   - 词汇表: vocab.json")
    print(f"   - PyTorch模型: model.pth")

def load_model_info(model_dir: str = None) -> tuple:
    """
    从保存的文件中加载模型和词汇表
    
    Args:
        model_dir: 模型保存目录
        
    Returns:
        (model, vocab_stoi, vocab_itos): 加载的模型和词汇表
    """
    # 设置默认加载目录到根目录的saved_models/transformer文件夹
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'transformer')
    
    # 加载模型结构信息
    with open(os.path.join(model_dir, 'model_structure.json'), 'r', encoding='utf-8') as f:
        model_info = json.load(f)
    
    # 加载词汇表
    with open(os.path.join(model_dir, 'vocab.json'), 'r', encoding='utf-8') as f:
        vocab_info = json.load(f)
    
    vocab_stoi = vocab_info['vocab_stoi']
    vocab_itos = vocab_info['vocab_itos']
    
    # 创建模型实例
    model = TransformerSentimentAnalyzer(
        vocab_size=model_info['vocab_size'],
        embedding_dim=model_info['embedding_dim'],
        hidden_dim=model_info['hidden_dim'],
        output_dim=model_info['output_dim'],
        pad_idx=model_info['pad_idx'],
        learning_rate=model_info['learning_rate'],
        num_heads=model_info.get('num_heads', 8),
        num_layers=model_info.get('num_layers', 2),
        max_seq_len=model_info.get('max_seq_len', 512)
    )
    
    # 加载模型权重
    model_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    print(f"模型已从 {model_dir}/ 加载")
    return model, vocab_stoi, vocab_itos