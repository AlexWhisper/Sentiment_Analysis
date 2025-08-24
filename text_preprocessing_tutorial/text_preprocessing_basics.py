# -*- coding: utf-8 -*-

"""
目标：让学生理解文本无法直接输入模型，必须转化为数字，并学习最核心的概念——词向量。

内容：
1. 文本清洗 (Text Cleaning)
2. 分词 (Tokenization)
3. 构建词表 (Vocabulary Building)
4. 数值化 (Numericalization)
5. 词嵌入入门 (Word Embeddings)
"""

import re
# !pip install spacy
# !python -m spacy download en_core_web_sm
import spacy

# 首次使用需要下载模型
# python -m spacy download en_core_web_sm
# 加载spacy模型
nlp = spacy.load("en_core_web_sm")


# -------------------------------------------------------------------
# 1. 文本清洗 (Text Cleaning)
# -------------------------------------------------------------------
print("--- 1. 文本清洗 ---")

# 原始文本通常包含很多"噪音"，比如HTML标签、标点符号、大小写不一致等。
# 这些噪音会干扰模型的学习，所以需要先清洗。
# 我们使用一个情感分析的例子：电影评论
raw_text = "<p>This movie is absolutely amazing! I love the story and characters. It's fantastic!</p>"
print(f"原始文本: {raw_text}")

# a. 去除HTML标签
cleaned_text = re.sub(r'<.*?>', '', raw_text)
print(f"去除HTML标签后: {cleaned_text}")

# b. 转换为小写
cleaned_text = cleaned_text.lower()
print(f"转换为小写后: {cleaned_text}")

# c. 去除标点符号
cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
print(f"去除标点符号后: {cleaned_text}")

print("\n" + "="*50 + "\n")


# -------------------------------------------------------------------
# 2. 分词 (Tokenization)
# -------------------------------------------------------------------
print("--- 2. 分词 ---")

# 计算机无法理解完整的句子，但可以理解独立的“词”（Token）。
# 分词就是将文本切分成一个个词语的过程。
doc = nlp(cleaned_text)
tokens = [token.text for token in doc]
print(f"清洗后的文本: {cleaned_text}")
print(f"分词结果: {tokens}")

print("\n" + "="*50 + "\n")


# -------------------------------------------------------------------
# 3. 构建词表 (Vocabulary Building)
# -------------------------------------------------------------------
print("--- 3. 构建词表 ---")

# 模型只能处理数字，所以我们需要将每个词映射到一个唯一的整数ID。
# “词表”就是这个词语到索引的映射字典。
# 我们通常会加入一些特殊的token：
# <unk>: "unknown", 用于表示在训练集中未出现过的词。
# <pad>: "padding", 用于将所有文本序列填充到相同长度。

# 从分词结果中构建词集合
vocab_set = set(tokens)
vocab = {word: i+2 for i, word in enumerate(vocab_set)} # i+2是为了给<pad>和<unk>留出位置
vocab['<pad>'] = 0
vocab['<unk>'] = 1

print(f"构建的词表: {vocab}")
print(f"词表大小: {len(vocab)}")

print("\n" + "="*50 + "\n")


# -------------------------------------------------------------------
# 4. 数值化 (Numericalization)
# -------------------------------------------------------------------
print("--- 4. 数值化 ---")

# 有了词表，我们就可以将文本序列（tokens）转换为整数索引序列。
numerical_sequence = [vocab.get(token, vocab['<unk>']) for token in tokens]

print(f"原始Tokens: {tokens}")
print(f"数值化序列: {numerical_sequence}")

# 模拟一个未在词表中出现过的词
test_sentence = "this movie is terrible and boring"
test_doc = nlp(test_sentence)
test_tokens = [token.text for token in test_doc]
numerical_test = [vocab.get(token, vocab['<unk>']) for token in test_tokens]
print(f"\n测试句子 '{test_sentence}' 的数值化结果:")
print(numerical_test)
print("注意：'terrible' 和 'boring' 因为不在词表中，被映射为了 <unk> 的索引 1")


print("\n" + "="*50 + "\n")


# -------------------------------------------------------------------
# 5. 词嵌入入门 (Word Embeddings)
# -------------------------------------------------------------------
print("--- 5. 词嵌入入门 ---")

# a. 方法一：One-Hot 编码 (独热编码)
print("\n--- a. One-Hot 编码 ---")
# One-Hot编码是最简单的表示方法。假设词表大小为 V，那么每个词就被表示成一个V维向量，
# 其中，该词在词表中的索引位置为1，其他所有位置都为0。

# 举例: 使用情感分析相关的词汇
simple_vocab = {'amazing': 0, 'fantastic': 1, 'terrible': 2}
vocab_size_simple = len(simple_vocab)

def one_hot_encode(word, vocab):
    vec = [0] * len(vocab)
    if word in vocab:
        vec[vocab[word]] = 1
    return vec

amazing_one_hot = one_hot_encode('amazing', simple_vocab)
fantastic_one_hot = one_hot_encode('fantastic', simple_vocab)
terrible_one_hot = one_hot_encode('terrible', simple_vocab)

print(f"词表: {simple_vocab}")
print(f"'amazing' 的 One-Hot 编码: {amazing_one_hot}")
print(f"'fantastic' 的 One-Hot 编码: {fantastic_one_hot}")
print(f"'terrible' 的 One-Hot 编码: {terrible_one_hot}")

print("\nOne-Hot编码的缺点:")
print("1. 维度灾难 (Curse of Dimensionality): 如果词表有10000个词，每个向量的维度就是10000，非常稀疏，计算效率低。")
print("2. 无法表达语义相似度: 任何两个词的one-hot向量都是正交的，它们的点积为0。这意味着无法从向量层面判断'amazing'和'fantastic'比'amazing'和'terrible'更相似。")


# b. 方法二：词嵌入 (Word Embedding)
print("\n--- b. 词嵌入 (Word Embedding) ---")
# 词嵌入用一个低维度的、稠密的浮点数向量来表示一个词。
# 这个向量是在模型训练过程中“学习”到的，它能够捕捉词的语义信息。
# 语义相似的词，它们的词向量在向量空间中的距离也更近。

# 举例 (手动模拟已经学习好的词向量):
# 假设我们的嵌入向量是4维的
# 这些向量是"学习"的结果，我们这里只是为了演示
# 注意 "amazing" 和 "fantastic" 的向量在某些维度上比较接近（都是正面情感）
mock_embedding_table = {
    'amazing':   [0.9, 0.8, 0.1, 0.2],
    'fantastic': [0.85, 0.82, 0.15, 0.18],
    'terrible':  [0.1, 0.2, 0.9, 0.7]
}

amazing_vec = mock_embedding_table['amazing']
fantastic_vec = mock_embedding_table['fantastic']
terrible_vec = mock_embedding_table['terrible']

print(f"'amazing' 的词嵌入 (模拟): {amazing_vec}")
print(f"'fantastic' 的词嵌入 (模拟): {fantastic_vec}")
print(f"'terrible' 的词嵌入 (模拟): {terrible_vec}")

# 使用余弦相似度来比较向量间的关系
import numpy as np

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
# 计算单词'amazing'和单词'fantastic'的余弦相似度
sim_amazing_fantastic = cosine_similarity(amazing_vec, fantastic_vec)
# 计算单词'amazing'和单词'terrible'的余弦相似度
sim_amazing_terrible = cosine_similarity(amazing_vec, terrible_vec)

print(f"\n'amazing' 和 'fantastic' 的余弦相似度: {sim_amazing_fantastic:.4f}")
print(f"'amazing' 和 'terrible' 的余弦相似度: {sim_amazing_terrible:.4f}")
print("结论: 词嵌入向量成功地捕捉到了'amazing'和'fantastic'在情感上更相近这一事实（都是正面情感）。")


# c. 深度学习框架中的实现
print("\n--- c. 深度学习框架中的实现 ---")
# 在PyTorch中，词嵌入通常通过一个Embedding层实现。
# 它本质上是一个大的权重矩阵（查找表），形状为 (vocab_size, embedding_dim)。
# 当我们输入一个词的整数索引时，它就去这个表中查找并返回对应的行（即该词的嵌入向量）。
# 这个权重矩阵在模型训练开始时随机初始化，然后通过反向传播算法不断更新，从而“学习”到词的语义。

print("\nPyTorch nn.Embedding 示例:")
try:
    import torch
    import torch.nn as nn

    # 假设我们有更完整的词表和数值化序列
    # vocab from section 3
    # numerical_sequence from section 4
    
    # 首先回顾一下我们的处理流程，方便学生对比理解
    print("\n=== 回顾处理流程 ===")
    print(f"原始句子: {raw_text}")
    print(f"清洗后文本: {cleaned_text}")
    print(f"分词结果: {tokens}")
    print(f"构建的词表: {vocab}")
    print(f"数值化序列: {numerical_sequence}")
    print("\n=== 现在进行词嵌入 ===")
    
    vocab_size = len(vocab)
    embedding_dim = 5 # 使用5维向量

    embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

    # 将之前的数值化序列转换为Tensor
    input_tensor = torch.LongTensor(numerical_sequence)
    
    # 获取整句话的嵌入向量
    embedded_output = embedding_layer(input_tensor)

    print(f"\n输入到Embedding层的数值化序列: {numerical_sequence}")
    print(f"输入Tensor的Shape: {input_tensor.shape}")
    print(f"\n经过Embedding层后的输出 (Shape: {embedded_output.shape}):")
    print(embedded_output)
    print("\n注意: 输出的shape是 (sequence_length, embedding_dim)")
    print("每一行都代表了输入序列中对应位置的词的5维嵌入向量。")
    print("\n对比理解:")
    for i, (token, num_id) in enumerate(zip(tokens, numerical_sequence)):
        print(f"  词 '{token}' -> 索引 {num_id} -> 嵌入向量 {(embedded_output[i].tolist())[:5]}")
    
    print("\n=== 重要区分：Token级别 vs 句子级别的Embedding ===\n")
    
    # Token级别的embedding（Hidden States）
    print("1. Token级别的Embedding（Hidden States）:")
    print(f"   - 当前输出shape: {embedded_output.shape}")
    print(f"   - 含义: 每个token都有自己的{embedding_dim}维向量")
    print(f"   - 特点: sequence长度会因句子长短而变化，但embedding维度固定")
    print("   - 用途: 适用于需要每个词位置信息的任务（如命名实体识别、词性标注）")
    
    # 演示不同长度句子的token级别embedding
    short_sentence = "good movie"
    short_doc = nlp(short_sentence)
    short_tokens = [token.text for token in short_doc]
    short_numerical = [vocab.get(token, vocab['<unk>']) for token in short_tokens]
    short_tensor = torch.LongTensor(short_numerical)
    short_embedded = embedding_layer(short_tensor)
    
    print(f"\n   示例对比:")
    print(f"   原句: '{cleaned_text}' -> shape: {embedded_output.shape}")
    print(f"   短句: '{short_sentence}' -> shape: {short_embedded.shape}")
    print("   可以看到：句子长度不同，第一个维度（seq_len）就不同")
    
    # 句子级别的embedding（Sentence Embedding）
    print("\n2. 句子级别的Embedding（Sentence Embedding）:")
    print("   - 目标: 将整个句子表示为一个固定长度的向量")
    print("   - 常见方法:")
    
    # 方法1: 平均池化
    sentence_embedding_mean = torch.mean(embedded_output, dim=0)  # 对sequence维度求平均
    short_sentence_embedding_mean = torch.mean(short_embedded, dim=0)
    
    print(f"     a) 平均池化 (Mean Pooling):")
    print(f"        原句句向量shape: {sentence_embedding_mean.shape}")
    print(f"        短句句向量shape: {short_sentence_embedding_mean.shape}")
    print(f"        特点: 无论句子多长，都得到固定的{embedding_dim}维向量")
    
    # 方法2: 最大池化
    sentence_embedding_max, _ = torch.max(embedded_output, dim=0)
    short_sentence_embedding_max, _ = torch.max(short_embedded, dim=0)
    
    print(f"     b) 最大池化 (Max Pooling):")
    print(f"        原句句向量shape: {sentence_embedding_max.shape}")
    print(f"        短句句向量shape: {short_sentence_embedding_max.shape}")
    
    # 方法3: 取第一个token（模拟[CLS] token）
    sentence_embedding_first = embedded_output[0]  # 取第一个token
    short_sentence_embedding_first = short_embedded[0]
    
    print(f"     c) 取首个Token (模拟[CLS]):")
    print(f"        原句句向量shape: {sentence_embedding_first.shape}")
    print(f"        短句句向量shape: {short_sentence_embedding_first.shape}")
    
    print("\n   应用场景对比:")
    print("   - Token级别: 机器翻译、命名实体识别、问答系统（需要知道每个词的位置）")
    print("   - 句子级别: 文本分类、情感分析、文本相似度计算（只需要整体语义）")
    
    print("\n   实际使用中:")
    print("   - BERT等预训练模型输出的是Token级别的embedding")
    print("   - 如果任务需要句子级别，需要进一步池化处理")
    print("   - Sentence-BERT等模型直接输出句子级别的embedding")

except ImportError:
    print("\nPyTorch 未安装，跳过PyTorch示例。")


print("\n总结：文本预处理将原始文本一步步转化为可供模型学习的数值化输入，而词嵌入层则是连接文本与深度学习模型的桥梁，它比One-Hot编码更高效、更能捕捉语义信息。")