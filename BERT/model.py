import os
# 设置Hugging Face镜像站，必须在导入transformers之前设置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
import numpy as np
import json


class BERTSentimentAnalyzer(nn.Module):
    """
    基于BERT的情感分析模型
    使用预训练的BERT模型进行微调
    """
    
    def __init__(self, model_name='bert-base-uncased', num_classes=2, dropout_rate=0.1):
        """
        初始化BERT情感分析模型
        
        Args:
            model_name (str): 预训练BERT模型名称
            num_classes (int): 分类类别数量
            dropout_rate (float): Dropout比率
        """
        super(BERTSentimentAnalyzer, self).__init__()
        
        # 模型参数
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # 加载预训练的BERT模型
        self.bert = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            output_attentions=False,
            output_hidden_states=False
        )
        
        # 添加dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 初始化tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # 优化器和损失函数将在训练时设置
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        前向传播
        
        Args:
            input_ids (torch.Tensor): 输入的token IDs
            attention_mask (torch.Tensor): 注意力掩码
            token_type_ids (torch.Tensor): token类型IDs
            labels (torch.Tensor): 标签（训练时使用）
            
        Returns:
            outputs: BERT模型输出
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        
        return outputs
    
    def setup_optimizer(self, learning_rate=2e-5, weight_decay=0.01, num_training_steps=None, num_warmup_steps=None):
        """
        设置优化器和学习率调度器
        
        Args:
            learning_rate (float): 学习率
            weight_decay (float): 权重衰减
            num_training_steps (int): 总训练步数
            num_warmup_steps (int): 预热步数
        """
        # 设置不同的学习率给不同的参数组
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
        
        # 设置学习率调度器
        if num_training_steps and num_warmup_steps:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
    
    def predict(self, text, max_length=512):
        """
        对单个文本进行预测
        
        Args:
            text (str): 输入文本
            max_length (int): 最大序列长度
            
        Returns:
            prediction (int): 预测类别
            probability (float): 预测概率
        """
        self.eval()
        
        # 编码文本
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        # 移动到设备
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        with torch.no_grad():
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()
            probability = probabilities[0][prediction].item()
            
        return prediction, probability
    
    def get_model_info(self):
        """
        获取模型信息
        
        Returns:
            dict: 模型信息字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # 假设float32
        }

def binary_accuracy(preds, y):
    """
    计算二分类准确率
    
    Args:
        preds (torch.Tensor): 预测logits
        y (torch.Tensor): 真实标签
        
    Returns:
        float: 准确率
    """
    # 获取预测类别
    predicted = torch.argmax(preds, dim=1)
    correct = (predicted == y).float()
    acc = correct.sum() / len(correct)
    return acc.item()

def save_model_info(model, filepath, additional_info=None):
    """
    保存模型信息到JSON文件
    
    Args:
        model: 模型实例
        filepath (str): 保存路径
        additional_info (dict): 额外信息
    """
    model_info = model.get_model_info()
    
    if additional_info:
        model_info.update(additional_info)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print(f"模型信息已保存到: {filepath}")

def load_model_info(filepath):
    """
    从JSON文件加载模型信息
    
    Args:
        filepath (str): 文件路径
        
    Returns:
        dict: 模型信息
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        model_info = json.load(f)
    
    return model_info

if __name__ == "__main__":
    # 测试模型创建
    print("创建BERT情感分析模型...")
    model = BERTSentimentAnalyzer()
    
    # 打印模型信息
    info = model.get_model_info()
    print("\n模型信息:")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # 测试预测功能
    test_text = "This movie is really great! I love it."
    print(f"\n测试文本: {test_text}")
    
    try:
        prediction, probability = model.predict(test_text)
        print(f"预测结果: {prediction} (概率: {probability:.4f})")
    except Exception as e:
        print(f"预测时出现错误: {e}")
        print("注意: 首次运行需要下载预训练模型，请确保网络连接正常")