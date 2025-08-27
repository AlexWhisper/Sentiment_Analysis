#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2024-present, Mr. Yao.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
#!/usr/bin/env python3
"""
IMDB情感分析完整示例
使用PyTorch实现的LSTM神经网络进行情感分析
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import time
from model import TransformerSentimentAnalyzer, binary_accuracy, save_model_info
from data_loader import prepare_data

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def train_epoch(model, iterator, optimizer, criterion, epoch, total_epochs):
    """
    训练一个epoch
    
    Args:
        model: Transformer模型
        iterator: 训练数据迭代器
        optimizer: 优化器
        criterion: 损失函数
        epoch: 当前epoch
        total_epochs: 总epoch数
        
    Returns:
        (epoch_loss, epoch_acc): 平均损失和准确率
    """
    model.train()
    epoch_loss, epoch_acc = 0, 0
    
    progress_bar = tqdm(iterator, desc=f'Epoch [{epoch + 1}/{total_epochs}]', leave=False)
    
    for i, (text, label) in enumerate(progress_bar):
        optimizer.zero_grad()
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, label)
        acc = binary_accuracy(predictions, label)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{acc.item()*100:.2f}%'
        })
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate_epoch(model, iterator, criterion):
    """
    评估一个epoch
    
    Args:
        model: Transformer模型
        iterator: 验证/测试数据迭代器
        criterion: 损失函数
        
    Returns:
        (epoch_loss, epoch_acc): 平均损失和准确率
    """
    model.eval()
    epoch_loss, epoch_acc = 0, 0
    
    with torch.no_grad():
        for text, label in iterator:
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, label)
            acc = binary_accuracy(predictions, label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)



def train_model(epochs=10, learning_rate=0.001, batch_size=32, 
               embedding_dim=100, hidden_dim=128, max_vocab_size=25000):
    """
    完整的模型训练流程
    
    Args:
        epochs: 训练轮数
        learning_rate: 学习率
        batch_size: 批次大小
        embedding_dim: 词嵌入维度
        hidden_dim: LSTM隐藏层维度
        max_vocab_size: 词汇表最大大小
    """
    print("IMDB情感分析Transformer神经网络训练")
    
    # 1. 准备数据
    print("\n1. 准备数据...")
    data_components = prepare_data(
        max_vocab_size=max_vocab_size,
        batch_size=batch_size,
        max_seq_len=512
    )
    
    train_dataloader = data_components['train_dataloader']
    valid_dataloader = data_components['valid_dataloader']
    test_dataloader = data_components['test_dataloader']
    vocab_stoi = data_components['vocab_stoi']
    vocab_itos = data_components['vocab_itos']
    PAD_IDX = data_components['PAD_IDX']
    vocab_size = data_components['vocab_size']
    
    # 2. 创建模型
    print("\n2. 创建Transformer模型...")
    model = TransformerSentimentAnalyzer(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=1,
        pad_idx=PAD_IDX,
        learning_rate=learning_rate,
        num_heads=8,
        num_layers=2,
        max_seq_len=512
    )
    model = model.to(device)
    
    print(f"模型参数:")
    print(f"  词汇表大小: {vocab_size}")
    print(f"  词嵌入维度: {embedding_dim}")
    print(f"  Transformer隐藏层维度: {hidden_dim}")
    print(f"  注意力头数: 8")
    print(f"  Transformer层数: 2")
    print(f"  学习率: {learning_rate}")
    
    # 3. 训练模型
    print(f"\n3. 开始训练 ({epochs} epochs)...")
    start_time = time.time()
    
    train_losses, train_accs = [], []
    valid_losses, valid_accs = [], []
    
    best_valid_acc = 0
    
    for epoch in range(epochs):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_dataloader, model.optimizer, model.criterion, epoch, epochs
        )
        
        # 验证
        valid_loss, valid_acc = evaluate_epoch(
            model, valid_dataloader, model.criterion
        )
        
        # 记录历史
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        
        # 打印结果
        print(f'Epoch {epoch + 1:2d}/{epochs}:')
        print(f'\t训练损失: {train_loss:.3f} | 训练准确率: {train_acc * 100:.2f}%')
        print(f'\t验证损失: {valid_loss:.3f} | 验证准确率: {valid_acc * 100:.2f}%')
        
        # 保存最佳模型
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            save_model_info(model, vocab_stoi, vocab_itos)
            print(f'\t*** 新的最佳验证准确率: {best_valid_acc * 100:.2f}% ***')
        
        print()
    
    training_time = time.time() - start_time
    print(f"训练完成！总用时: {training_time:.2f}秒")
    
    # 4. 测试模型
    print("\n4. 在测试集上评估...")
    test_loss, test_acc = evaluate_epoch(model, test_dataloader, model.criterion)
    print(f'测试损失: {test_loss:.3f} | 测试准确率: {test_acc * 100:.2f}%')
    
    # 5. 总结
    print("\n训练总结:")
    print(f"  最佳验证准确率: {best_valid_acc * 100:.2f}%")
    print(f"  最终测试准确率: {test_acc * 100:.2f}%")
    print(f"  训练时间: {training_time:.2f}秒")
    print(f"  模型已保存到: ./saved_models/")
    
    return model, data_components

def main():
    """
    主函数
    """
    # 训练参数
    EPOCHS = 100
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EMBEDDING_DIM = 128  # 改为128，能被8整除
    HIDDEN_DIM = 256
    MAX_VOCAB_SIZE = 25000
    
    # 开始训练
    model, data_components = train_model(
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        max_vocab_size=MAX_VOCAB_SIZE
    )
    
    print("\n训练完成！可以运行 inference.py 进行预测测试。")

if __name__ == "__main__":
    main()