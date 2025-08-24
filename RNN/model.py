import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Dict
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

class RNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int, num_classes: int, pad_idx: int, learning_rate: float = 0.001):
        """
        初始化循环神经网络
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            hidden_size: 隐藏层维度
            num_layers: RNN层数
            num_classes: 输出类别数
            pad_idx: 填充索引
            learning_rate: 学习率
        """
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.pad_idx = pad_idx


    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            text: 输入数据，形状为 (batch_size, seq_len)
            
        Returns:
            输出预测
        """
        embedded = self.embedding(text)
        h0 = torch.zeros(self.num_layers, text.size(0), self.hidden_size).to(text.device)
        out, _ = self.rnn(embedded, h0)
        out = self.fc(out[:, -1, :])
        return out

    def train_model(self, dataloader, epochs: int, verbose: bool = True) -> List[float]:
        """
        训练神经网络
        
        Args:
            dataloader: 数据加载器
            epochs: 训练轮数
            verbose: 是否打印训练信息
            
        Returns:
            losses: 每轮的损失值
        """
        losses = []
        self.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            for labels, texts in dataloader:
                self.optimizer.zero_grad()
                outputs = self(texts)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return losses

    def predict(self, text: str, vocab: Dict[str, int], max_len: int) -> np.ndarray:
        """
        预测
        
        Args:
            text: 输入文本
            vocab: 词汇表
            max_len: 最大长度
            
        Returns:
            预测结果
        """
        self.eval()
        with torch.no_grad():
            tokens = text.split()
            indexed = [vocab.get(t, vocab['<unk>']) for t in tokens]
            if len(indexed) > max_len:
                indexed = indexed[:max_len]
            tensor = torch.LongTensor(indexed).unsqueeze(0)
            output = self(tensor)
            probabilities = F.softmax(output, dim=1)
            return probabilities.numpy()

    def predict_classes(self, text: str, vocab: Dict[str, int], max_len: int) -> np.ndarray:
        """
        预测类别
        
        Args:
            text: 输入文本
            vocab: 词汇表
            max_len: 最大长度
            
        Returns:
            预测的类别索引
        """
        self.eval()
        with torch.no_grad():
            tokens = text.split()
            indexed = [vocab.get(t, vocab['<unk>']) for t in tokens]
            if len(indexed) > max_len:
                indexed = indexed[:max_len]
            tensor = torch.LongTensor(indexed).unsqueeze(0)
            output = self(tensor)
            _, predicted = torch.max(output, 1)
            return predicted.numpy()

    def accuracy(self, dataloader, vocab) -> float:
        """
        计算准确率
        
        Args:
            dataloader: 数据加载器
            vocab: 词汇表
            
        Returns:
            准确率
        """
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for labels, texts in dataloader:
                outputs = self(texts)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

