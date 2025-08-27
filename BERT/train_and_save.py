# Copyright (c) 2025 NIS Lab, Hebei Normal University
# Author: Wang Minghu <wangminghu41@163.com>
# Internal Use Only. Unauthorized distribution is strictly prohibited.

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime

# 导入自定义模块
from model import BERTSentimentAnalyzer, binary_accuracy, save_model_info
from data_loader import prepare_data

def train_epoch(model, data_loader, optimizer, scheduler, device, epoch):
    """
    训练一个epoch
    
    Args:
        model: BERT模型
        data_loader: 训练数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 设备
        epoch: 当前epoch
        
    Returns:
        tuple: (平均损失, 平均准确率)
    """
    model.train()
    total_loss = 0
    total_accuracy = 0
    total_samples = 0
    
    # 创建进度条
    pbar = tqdm(data_loader, desc=f'Epoch {epoch+1} [Train]')
    
    for batch_idx, batch in enumerate(pbar):
        # 将数据移动到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        # 计算准确率
        accuracy = binary_accuracy(logits, labels)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新参数
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        # 累计统计
        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        total_accuracy += accuracy * batch_size
        total_samples += batch_size
        
        # 更新进度条
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{accuracy:.4f}',
            'LR': f'{current_lr:.2e}'
        })
    
    avg_loss = total_loss / total_samples
    avg_accuracy = total_accuracy / total_samples
    
    return avg_loss, avg_accuracy

def evaluate_epoch(model, data_loader, device, epoch, phase='Val'):
    """
    评估一个epoch
    
    Args:
        model: BERT模型
        data_loader: 验证/测试数据加载器
        device: 设备
        epoch: 当前epoch
        phase: 阶段名称（Val/Test）
        
    Returns:
        tuple: (平均损失, 平均准确率)
    """
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_samples = 0
    
    # 创建进度条
    pbar = tqdm(data_loader, desc=f'Epoch {epoch+1} [{phase}]')
    
    with torch.no_grad():
        for batch in pbar:
            # 将数据移动到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # 计算准确率
            accuracy = binary_accuracy(logits, labels)
            
            # 累计统计
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_accuracy += accuracy * batch_size
            total_samples += batch_size
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.4f}'
            })
    
    avg_loss = total_loss / total_samples
    avg_accuracy = total_accuracy / total_samples
    
    return avg_loss, avg_accuracy



def main():
    """
    主训练函数
    """
    # 超参数设置
    EPOCHS = 3  # BERT微调通常只需要少量epoch
    LEARNING_RATE = 2e-5  # BERT推荐的学习率
    BATCH_SIZE = 16  # 根据GPU内存调整
    MAX_LENGTH = 256  # 序列最大长度
    MODEL_NAME = 'bert-base-uncased'  # 预训练模型名称
    WARMUP_RATIO = 0.1  # 预热步数比例
    WEIGHT_DECAY = 0.01  # 权重衰减
    
    print("开始BERT情感分析模型训练")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 准备数据
    print("准备数据...")
    train_loader, val_loader, test_loader, tokenizer = prepare_data(
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        model_name=MODEL_NAME
    )
    
    # 创建模型
    print("创建模型...")
    model = BERTSentimentAnalyzer(model_name=MODEL_NAME, num_classes=2)
    model.to(device)
    
    # 计算训练步数
    num_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
    
    # 设置优化器和调度器
    model.setup_optimizer(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps
    )
    
    # 训练历史记录
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_accuracy = 0
    best_model_state = None
    
    # 创建保存目录
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'bert')
    os.makedirs(save_dir, exist_ok=True)
    
    # 开始训练
    print("开始训练...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        
        # 训练阶段
        train_loss, train_acc = train_epoch(
            model, train_loader, model.optimizer, model.scheduler, device, epoch
        )
        
        # 验证阶段
        val_loss, val_acc = evaluate_epoch(
            model, val_loader, device, epoch, 'Val'
        )
        
        # 记录历史
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # 打印epoch结果
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_state = model.state_dict().copy()
            print(f"新的最佳验证准确率: {best_val_accuracy:.4f}")
    
    # 训练完成
    training_time = time.time() - start_time
    print(f"训练完成! 总用时: {training_time:.2f}秒")
    print(f"最佳验证准确率: {best_val_accuracy:.4f}")
    
    # 加载最佳模型进行测试
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # 测试阶段
    print("开始测试...")
    test_loss, test_acc = evaluate_epoch(model, test_loader, device, EPOCHS-1, 'Test')
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    
    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f'bert_sentiment_model_{timestamp}.pth')
    torch.save({
        'model_state_dict': best_model_state if best_model_state else model.state_dict(),
        'model_config': {
            'model_name': MODEL_NAME,
            'num_classes': 2,
            'max_length': MAX_LENGTH
        },
        'training_config': {
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'weight_decay': WEIGHT_DECAY
        },
        'results': {
            'best_val_accuracy': best_val_accuracy,
            'test_accuracy': test_acc,
            'test_loss': test_loss
        }
    }, model_path)
    print(f"模型已保存到: {model_path}")
    
    # 保存模型信息
    info_path = os.path.join(save_dir, f'bert_model_info_{timestamp}.json')
    additional_info = {
        'training_time_seconds': training_time,
        'best_validation_accuracy': best_val_accuracy,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'training_config': {
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'max_length': MAX_LENGTH,
            'weight_decay': WEIGHT_DECAY
        }
    }
    save_model_info(model, info_path, additional_info)
    
    print("训练完成！")
    print(f"模型文件: {model_path}")
    print(f"信息文件: {info_path}")

if __name__ == "__main__":
    main()