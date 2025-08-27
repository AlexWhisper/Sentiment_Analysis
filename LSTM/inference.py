# Copyright (c) 2025 NIS Lab, Hebei Normal University
# Author: Wang Minghu <wangminghu41@163.com>
# Internal Use Only. Unauthorized distribution is strictly prohibited.

#!/usr/bin/env python3
"""
加载并使用保存的LSTM模型进行情感分析预测
"""

import torch
from model import load_model_info
from data_loader import spacy_tokenizer, load_spacy_tokenizer

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_sentiment(model, sentence, vocab_stoi, tokenizer_func, UNK_IDX, threshold=0.5):
    """
    预测单个句子的情感
    
    Args:
        model: 训练好的LSTM模型
        sentence: 输入句子
        vocab_stoi: 词汇表（字符串到索引映射）
        tokenizer_func: 分词函数
        UNK_IDX: 未知词索引
        threshold: 分类阈值
        
    Returns:
        (prediction, probability): 预测结果和概率
    """
    model.eval()
    
    # 分词和索引转换
    tokenized = tokenizer_func(sentence)
    indexed = [vocab_stoi.get(token, UNK_IDX) for token in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)  # 添加 batch 维度
    
    # 预测
    with torch.no_grad():
        prediction = torch.sigmoid(model(tensor))
    
    probability = prediction.item()
    sentiment = 'positive' if probability > threshold else 'negative'
    
    return sentiment, probability

def print_prediction(sentence, prediction, probability, true_sentiment=None):
    """
    打印预测结果
    
    Args:
        sentence: 输入句子
        prediction: 预测结果
        probability: 预测概率
        true_sentiment: 真实情感（可选）
    """
    print(f"\n句子: '{sentence}'")
    print(f"预测情感: {prediction}")
    print(f"预测概率: {probability:.4f}")
    
    if true_sentiment:
        correct = prediction == true_sentiment
        print(f"真实情感: {true_sentiment}")
        print(f"预测正确: {'✓' if correct else '✗'}")



def test_sample_predictions(model, vocab_stoi, tokenizer_func, UNK_IDX):
    """
    测试一些示例预测
    
    Args:
        model: 训练好的LSTM模型
        vocab_stoi: 词汇表
        tokenizer_func: 分词函数
        UNK_IDX: 未知词索引
    """
    print("\n示例预测测试")
    
    # 测试样例
    test_samples = [
        ("What a fantastic film! I absolutely loved it.", "positive"),
        ("This movie was terrible. I wasted my time.", "negative"),
        ("The acting was brilliant and the story was engaging.", "positive"),
        ("Boring and predictable. Not worth watching.", "negative"),
        ("An okay movie, nothing special but not bad either.", "neutral"),
        ("Outstanding performance by the lead actor!", "positive"),
        ("The worst movie I've ever seen in my life.", "negative"),
        ("A masterpiece of cinema with incredible visuals.", "positive")
    ]
    
    correct_predictions = 0
    total_predictions = 0
    
    for sentence, expected in test_samples:
        prediction, probability = predict_sentiment(
            model, sentence, vocab_stoi, tokenizer_func, UNK_IDX
        )
        
        # 对于neutral，我们不计入准确率统计
        if expected != "neutral":
            is_correct = prediction == expected
            correct_predictions += is_correct
            total_predictions += 1
        
        print_prediction(sentence, prediction, probability, expected if expected != "neutral" else None)
        print("-" * 40)
    
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\n示例测试准确率: {accuracy * 100:.2f}% ({correct_predictions}/{total_predictions})")



def main():
    """
    主函数
    """
    print("LSTM情感分析模型推理")
    
    try:
        # 1. 加载模型和词汇表
        print("\n1. 加载保存的模型...")
        model, vocab_stoi, vocab_itos = load_model_info()
        model = model.to(device)
        
        # 2. 加载分词器
        print("\n2. 加载分词器...")
        nlp = load_spacy_tokenizer()
        tokenizer_func = lambda text: spacy_tokenizer(text, nlp)
        UNK_IDX = vocab_stoi['<unk>']
        
        print(f"模型加载成功！")
        print(f"词汇表大小: {len(vocab_stoi)}")
        print(f"设备: {device}")
        
        # 3. 示例预测测试
        test_sample_predictions(model, vocab_stoi, tokenizer_func, UNK_IDX)
        
    except FileNotFoundError:
        print("\n错误: 找不到保存的模型文件！")
        print("请先运行 train_and_save.py 训练并保存模型。")
    except Exception as e:
        print(f"\n加载模型时出错: {e}")
        print("请检查模型文件是否完整。")

if __name__ == "__main__":
    main()