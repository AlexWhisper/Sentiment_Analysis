import spacy
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from collections import Counter
from tqdm import tqdm
from typing import Tuple, Dict, List

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_spacy_tokenizer():
    """
    加载spacy分词器
    
    Returns:
        spacy分词器实例
    """
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("正在下载 'en_core_web_sm' spacy模型...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp

def spacy_tokenizer(text: str, nlp=None) -> List[str]:
    """
    使用spacy进行分词
    
    Args:
        text: 输入文本
        nlp: spacy分词器实例
        
    Returns:
        分词结果列表
    """
    if nlp is None:
        nlp = load_spacy_tokenizer()
    return [tok.text for tok in nlp.tokenizer(text)]

def build_vocab(dataset, tokenizer_func, max_size: int = 25000) -> Tuple[Dict[str, int], List[str]]:
    """
    构建词汇表
    
    Args:
        dataset: 数据集
        tokenizer_func: 分词函数
        max_size: 词汇表最大大小
        
    Returns:
        (vocab_stoi, vocab_itos): 词汇表的字符串到索引映射和索引到字符串映射
    """
    # 使用 Counter 统计词频
    counter = Counter()
    for item in tqdm(dataset, desc="构建词汇表"):
        counter.update(tokenizer_func(item['text']))
    
    # 定义特殊符号
    specials = ['<unk>', '<pad>']
    # 从高频词创建 stoi 字典
    stoi = {word: i for i, (word, _) in enumerate(counter.most_common(max_size), len(specials))}
    for i, special in enumerate(specials):
        stoi[special] = i
        
    # 创建 itos 列表
    itos = [None] * len(stoi)
    for word, i in stoi.items():
        itos[i] = word
        
    return stoi, itos

def load_imdb_data(test_size: float = 0.2, seed: int = 22) -> Tuple:
    """
    加载IMDB数据集
    
    Args:
        test_size: 验证集比例
        seed: 随机种子
        
    Returns:
        (train_dataset, valid_dataset, test_dataset): 训练集、验证集、测试集
    """
    print("正在加载IMDB数据集...")
    
    # 加载IMDB数据集，统一使用根目录的data文件夹
    import os
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    try:
        imdb_dataset = load_dataset("imdb", cache_dir=cache_dir)
        print("成功从本地缓存加载数据集")
    except Exception as e:
        print(f"从缓存加载失败: {e}")
        print("尝试在线下载...")
        try:
            imdb_dataset = load_dataset("imdb")
            print("成功在线下载数据集")
        except Exception as e2:
            print(f"在线下载也失败: {e2}")
            raise ConnectionError(f"无法加载IMDB数据集。请检查网络连接或本地缓存。错误信息: {e2}")
    
    train_valid_dataset = imdb_dataset['train'].train_test_split(seed=seed, test_size=test_size)
    train_dataset = train_valid_dataset['train']
    valid_dataset = train_valid_dataset['test']
    test_dataset = imdb_dataset['test']
    
    print(f"数据集加载完成:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(valid_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    
    return train_dataset, valid_dataset, test_dataset

def create_vocab_and_tokenizer(train_dataset, max_vocab_size: int = 25000) -> Tuple:
    """
    创建词汇表和分词器
    
    Args:
        train_dataset: 训练数据集
        max_vocab_size: 词汇表最大大小
        
    Returns:
        (vocab_stoi, vocab_itos, tokenizer_func, UNK_IDX, PAD_IDX): 词汇表和相关信息
    """
    # 加载分词器
    nlp = load_spacy_tokenizer()
    tokenizer_func = lambda text: spacy_tokenizer(text, nlp)
    
    # 构建词汇表
    vocab_stoi, vocab_itos = build_vocab(train_dataset, tokenizer_func, max_vocab_size)
    print(f"词汇表大小: {len(vocab_stoi)}")
    
    UNK_IDX = vocab_stoi['<unk>']
    PAD_IDX = vocab_stoi['<pad>']
    
    return vocab_stoi, vocab_itos, tokenizer_func, UNK_IDX, PAD_IDX

def collate_batch(batch, vocab_stoi: Dict[str, int], tokenizer_func, UNK_IDX: int, PAD_IDX: int, max_seq_len: int = 512):
    """
    批处理函数，用于DataLoader
    
    Args:
        batch: 批次数据
        vocab_stoi: 词汇表
        tokenizer_func: 分词函数
        UNK_IDX: 未知词索引
        PAD_IDX: 填充词索引
        max_seq_len: 最大序列长度
        
    Returns:
        (padded_text, label_list): 填充后的文本和标签
    """
    label_list, text_list = [], []
    for item in batch:
        label_list.append(float(item['label']))
        # 手动将 token 转换为 index
        tokens = tokenizer_func(item['text'])
        indices = [vocab_stoi.get(token, UNK_IDX) for token in tokens]
        # 截断过长的序列
        if len(indices) > max_seq_len:
            indices = indices[:max_seq_len]
        processed_text = torch.tensor(indices, dtype=torch.int64)
        text_list.append(processed_text)

    padded_text = nn.utils.rnn.pad_sequence(text_list, padding_value=PAD_IDX, batch_first=False)
    label_list = torch.tensor(label_list, dtype=torch.float32)
    return padded_text.to(device), label_list.to(device)

def create_data_loaders(train_dataset, valid_dataset, test_dataset, 
                       vocab_stoi: Dict[str, int], tokenizer_func, 
                       UNK_IDX: int, PAD_IDX: int, batch_size: int = 32, max_seq_len: int = 512) -> Tuple:
    """
    创建数据加载器
    
    Args:
        train_dataset: 训练数据集
        valid_dataset: 验证数据集
        test_dataset: 测试数据集
        vocab_stoi: 词汇表
        tokenizer_func: 分词函数
        UNK_IDX: 未知词索引
        PAD_IDX: 填充词索引
        batch_size: 批次大小
        max_seq_len: 最大序列长度
        
    Returns:
        (train_dataloader, valid_dataloader, test_dataloader): 数据加载器
    """
    # 创建collate函数的偏函数
    def collate_fn(batch):
        return collate_batch(batch, vocab_stoi, tokenizer_func, UNK_IDX, PAD_IDX, max_seq_len)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"数据加载器创建完成，批次大小: {batch_size}")
    return train_dataloader, valid_dataloader, test_dataloader

def prepare_data(max_vocab_size: int = 25000, batch_size: int = 32, test_size: float = 0.2, seed: int = 22, max_seq_len: int = 512):
    """
    准备所有数据相关的组件
    
    Args:
        max_vocab_size: 词汇表最大大小
        batch_size: 批次大小
        test_size: 验证集比例
        seed: 随机种子
        max_seq_len: 最大序列长度
        
    Returns:
        包含所有数据组件的字典
    """
    # 加载数据集
    train_dataset, valid_dataset, test_dataset = load_imdb_data(test_size, seed)
    
    # 创建词汇表和分词器
    vocab_stoi, vocab_itos, tokenizer_func, UNK_IDX, PAD_IDX = create_vocab_and_tokenizer(
        train_dataset, max_vocab_size
    )
    
    # 创建数据加载器
    train_dataloader, valid_dataloader, test_dataloader = create_data_loaders(
        train_dataset, valid_dataset, test_dataset, 
        vocab_stoi, tokenizer_func, UNK_IDX, PAD_IDX, batch_size, max_seq_len
    )
    
    return {
        'train_dataloader': train_dataloader,
        'valid_dataloader': valid_dataloader,
        'test_dataloader': test_dataloader,
        'vocab_stoi': vocab_stoi,
        'vocab_itos': vocab_itos,
        'tokenizer_func': tokenizer_func,
        'UNK_IDX': UNK_IDX,
        'PAD_IDX': PAD_IDX,
        'vocab_size': len(vocab_stoi)
    }

if __name__ == "__main__":
    # 测试数据加载功能
    print("测试数据加载功能...")
    data_components = prepare_data()
    print("\n数据加载测试完成！")
    print(f"词汇表大小: {data_components['vocab_size']}")
    print(f"UNK索引: {data_components['UNK_IDX']}")
    print(f"PAD索引: {data_components['PAD_IDX']}")