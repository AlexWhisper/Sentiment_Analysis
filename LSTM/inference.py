# Copyright (c) 2025 NIS Lab, Hebei Normal University
# Author: Wang Minghu <wangminghu41@163.com>
# Internal Use Only. Unauthorized distribution is strictly prohibited.

#!/usr/bin/env python3
"""
加载并使用保存的LSTM模型进行情感分析预测
"""

import torch
import matplotlib.pyplot as plt
from model import load_model_info
from data_loader import spacy_tokenizer, load_spacy_tokenizer
import random

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

def visualize_prediction(sentence, prediction, probability, true_sentiment=None):
    """
    可视化预测结果
    
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
    
    # 创建概率可视化
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    categories = ['Negative', 'Positive']
    probabilities = [1 - probability, probability]
    colors = ['red' if prediction == 'negative' else 'lightcoral', 
              'green' if prediction == 'positive' else 'lightgreen']
    
    bars = ax.bar(categories, probabilities, color=colors, alpha=0.7)
    ax.set_ylabel('概率')
    ax.set_title(f'情感分析预测结果\n句子: "{sentence[:50]}..."' if len(sentence) > 50 else f'情感分析预测结果\n句子: "{sentence}"')
    ax.set_ylim(0, 1)
    
    # 添加数值标签
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def interactive_prediction(model, vocab_stoi, tokenizer_func, UNK_IDX):
    """
    交互式预测模式
    
    Args:
        model: 训练好的LSTM模型
        vocab_stoi: 词汇表
        tokenizer_func: 分词函数
        UNK_IDX: 未知词索引
    """
    print("\n=" * 60)
    print("交互式情感分析预测")
    print("输入句子进行情感分析，输入 'quit' 退出")
    print("=" * 60)
    
    while True:
        sentence = input("\n请输入句子: ").strip()
        
        if sentence.lower() in ['quit', 'exit', '退出', 'q']:
            print("退出交互模式")
            break
        
        if not sentence:
            print("请输入有效的句子")
            continue
        
        try:
            prediction, probability = predict_sentiment(
                model, sentence, vocab_stoi, tokenizer_func, UNK_IDX
            )
            visualize_prediction(sentence, prediction, probability)
        except Exception as e:
            print(f"预测时出错: {e}")

def test_sample_predictions(model, vocab_stoi, tokenizer_func, UNK_IDX):
    """
    测试一些示例预测
    
    Args:
        model: 训练好的LSTM模型
        vocab_stoi: 词汇表
        tokenizer_func: 分词函数
        UNK_IDX: 未知词索引
    """
    print("\n=" * 60)
    print("示例预测测试")
    print("=" * 60)
    
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
        
        visualize_prediction(sentence, prediction, probability, expected if expected != "neutral" else None)
        print("-" * 40)
    
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\n示例测试准确率: {accuracy * 100:.2f}% ({correct_predictions}/{total_predictions})")

def analyze_model_vocabulary(vocab_stoi, vocab_itos):
    """
    分析模型词汇表
    
    Args:
        vocab_stoi: 词汇表（字符串到索引映射）
        vocab_itos: 词汇表（索引到字符串映射）
    """
    print("\n=" * 60)
    print("词汇表分析")
    print("=" * 60)
    
    print(f"词汇表大小: {len(vocab_stoi)}")
    print(f"特殊符号: <unk> (索引: {vocab_stoi.get('<unk>', 'N/A')}), <pad> (索引: {vocab_stoi.get('<pad>', 'N/A')})")
    
    # 显示一些高频词
    print("\n前20个高频词:")
    for i in range(2, min(22, len(vocab_itos))):
        if vocab_itos[i] is not None:
            print(f"{i-1:2d}. {vocab_itos[i]} (索引: {i})")
    
    # 随机显示一些词汇
    print("\n随机词汇样例:")
    random_indices = random.sample(range(100, min(1000, len(vocab_itos))), 10)
    for i, idx in enumerate(random_indices, 1):
        if vocab_itos[idx] is not None:
            print(f"{i:2d}. {vocab_itos[idx]} (索引: {idx})")

def main():
    """
    主函数
    """
    print("=" * 60)
    print("LSTM情感分析模型推理")
    print("=" * 60)
    
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
        
        # 3. 词汇表分析
        analyze_model_vocabulary(vocab_stoi, vocab_itos)
        
        # 4. 示例预测测试
        test_sample_predictions(model, vocab_stoi, tokenizer_func, UNK_IDX)
        
        # 5. 交互式预测
        interactive_prediction(model, vocab_stoi, tokenizer_func, UNK_IDX)
        
    except FileNotFoundError:
        print("\n错误: 找不到保存的模型文件！")
        print("请先运行 train_and_save.py 训练并保存模型。")
    except Exception as e:
        print(f"\n加载模型时出错: {e}")
        print("请检查模型文件是否完整。")

if __name__ == "__main__":
    main()