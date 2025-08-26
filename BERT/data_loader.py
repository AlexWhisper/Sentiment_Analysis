# Copyright (c) 2025 NIS Lab, Hebei Normal University
# Author: Wang Minghu <wangminghu41@163.com>
# Internal Use Only. Unauthorized distribution is strictly prohibited.

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
import numpy as np
from typing import List, Tuple, Dict

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

def analyze_text_lengths(texts: List[str], tokenizer: BertTokenizer, sample_size: int = 1000):
    """
    分析文本长度分布，帮助确定合适的max_length
    
    Args:
        texts (List[str]): 文本列表
        tokenizer (BertTokenizer): 分词器
        sample_size (int): 采样大小
        
    Returns:
        dict: 长度统计信息
    """
    print(f"分析文本长度分布（采样 {sample_size} 个文本）...")
    
    # 随机采样
    if len(texts) > sample_size:
        indices = np.random.choice(len(texts), sample_size, replace=False)
        sample_texts = [texts[i] for i in indices]
    else:
        sample_texts = texts
    
    lengths = []
    for text in sample_texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        lengths.append(len(tokens))
    
    lengths = np.array(lengths)
    
    stats = {
        'mean': np.mean(lengths),
        'median': np.median(lengths),
        'std': np.std(lengths),
        'min': np.min(lengths),
        'max': np.max(lengths),
        'percentile_90': np.percentile(lengths, 90),
        'percentile_95': np.percentile(lengths, 95),
        'percentile_99': np.percentile(lengths, 99)
    }
    
    print("文本长度统计:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    
    # 建议max_length
    suggested_max_length = int(stats['percentile_95'])
    print(f"\n建议的max_length: {suggested_max_length}")
    print(f"这将覆盖95%的文本，平衡效率和信息保留")
    
    return stats

if __name__ == "__main__":
    # 测试数据加载
    print("测试BERT数据加载器...")
    
    try:
        # 准备数据
        train_loader, val_loader, test_loader, tokenizer = prepare_data(
            batch_size=8,
            max_length=256
        )
        
        # 测试一个批次
        print("\n测试数据批次:")
        for batch in train_loader:
            print(f"input_ids shape: {batch['input_ids'].shape}")
            print(f"attention_mask shape: {batch['attention_mask'].shape}")
            print(f"labels shape: {batch['labels'].shape}")
            
            # 显示第一个样本的部分信息
            print(f"\n第一个样本:")
            print(f"input_ids: {batch['input_ids'][0][:20]}...")  # 显示前20个token
            print(f"attention_mask: {batch['attention_mask'][0][:20]}...")  # 显示前20个mask
            print(f"label: {batch['labels'][0]}")
            
            # 解码第一个样本
            decoded_text = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)
            print(f"解码文本: {decoded_text[:100]}...")  # 显示前100个字符
            
            break
        
        print("\n数据加载器测试成功！")
        
    except Exception as e:
        print(f"测试时出现错误: {e}")
        print("请确保已安装所需依赖: pip install transformers datasets")