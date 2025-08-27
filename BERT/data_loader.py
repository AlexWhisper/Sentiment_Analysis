# Copyright (c) 2025 NIS Lab, Hebei Normal University
# Author: Wang Minghu <wangminghu41@163.com>
# Internal Use Only. Unauthorized distribution is strictly prohibited.

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
import numpy as np
from typing import List

class IMDBDataset(Dataset):
    """
    IMDB数据集类，用于BERT模型训练
    """
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizer, max_length: int = 512):
        """
        初始化数据集
        
        Args:
            texts (List[str]): 文本列表
            labels (List[int]): 标签列表
            tokenizer (BertTokenizer): BERT分词器
            max_length (int): 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            dict: 包含input_ids, attention_mask, labels的字典
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 使用BERT tokenizer编码文本
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # 添加[CLS]和[SEP]标记
            max_length=self.max_length,
            padding='max_length',  # 填充到最大长度
            truncation=True,  # 截断超长文本
            return_attention_mask=True,  # 返回注意力掩码
            return_tensors='pt'  # 返回PyTorch张量
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def prepare_data(batch_size=16, max_length=512, model_name='bert-base-uncased', 
                 cache_dir=None, test_size=0.1, random_seed=42):
    """
    准备IMDB数据集用于BERT训练
    
    Args:
        batch_size (int): 批次大小
        max_length (int): 最大序列长度
        model_name (str): BERT模型名称
        cache_dir (str): 缓存目录
        test_size (float): 测试集比例
        random_seed (int): 随机种子
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, tokenizer)
    """
    print("正在加载IMDB数据集...")

    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # 加载IMDB数据集
    try:
        dataset = load_dataset('imdb', cache_dir=cache_dir)
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        print("尝试使用本地数据...")
        # 如果无法下载，尝试使用本地数据
        dataset = load_dataset('imdb', cache_dir=cache_dir, download_mode='reuse_cache_if_exists')
    
    # 初始化tokenizer
    print(f"初始化tokenizer: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # 准备训练数据
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    
    # 准备测试数据
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']
    
    # 从训练集中分出验证集
    np.random.seed(random_seed)
    train_size = len(train_texts)
    val_size = int(train_size * test_size)
    
    # 随机打乱训练数据
    indices = np.random.permutation(train_size)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    # 分割数据（将numpy索引转换为Python int类型）
    val_texts = [train_texts[int(i)] for i in val_indices]
    val_labels = [train_labels[int(i)] for i in val_indices]
    
    train_texts = [train_texts[int(i)] for i in train_indices]
    train_labels = [train_labels[int(i)] for i in train_indices]
    
    print(f"数据集大小:")
    print(f"  训练集: {len(train_texts)}")
    print(f"  验证集: {len(val_texts)}")
    print(f"  测试集: {len(test_texts)}")
    
    # 创建数据集对象
    train_dataset = IMDBDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = IMDBDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = IMDBDataset(test_texts, test_labels, tokenizer, max_length)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows上设置为0避免多进程问题
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"数据加载器创建完成，批次大小: {batch_size}")
    
    return train_loader, val_loader, test_loader, tokenizer