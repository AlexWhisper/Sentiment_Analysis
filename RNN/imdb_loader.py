import torch
from datasets import load_dataset
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def load_imdb_data(cache_path=None):
    """
    Loads the IMDb dataset from Hugging Face, caching it locally.

    Args:
        cache_path (str): The directory to cache the dataset.

    Returns:
        tuple: A tuple containing the training and test datasets.
    """
    # 设置默认缓存路径到根目录的data文件夹
    if cache_path is None:
        cache_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    print("Loading IMDb dataset...")
    
    # 尝试从本地缓存加载数据集
    try:
        dataset = load_dataset("imdb", cache_dir=cache_path)
        print("Dataset loaded successfully from cache.")
    except Exception as e1:
        print(f"Failed to load from cache: {e1}")
        try:
            # 尝试使用绝对路径的本地缓存
            abs_cache_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
            dataset = load_dataset("imdb", cache_dir=abs_cache_path)
            print("Dataset loaded successfully from absolute cache path.")
        except Exception as e2:
            print(f"Failed to load from absolute cache path: {e2}")
            try:
                # 最后尝试在线下载
                print("Attempting to download dataset online...")
                dataset = load_dataset("imdb")
                print("Dataset downloaded successfully.")
            except Exception as e3:
                print(f"Online download also failed: {e3}")
                raise ConnectionError(f"Unable to load IMDB dataset. Please check network connection or local cache. Error: {e3}")
    
    return dataset['train'], dataset['test']

def analyze_dataset_distribution(data):
    """
    Analyzes and displays the distribution of labels and review lengths.

    Args:
        data: The dataset to analyze.
    """
    print("Analyzing dataset distribution...")
    train_labels = []
    review_lengths = []
    for example in tqdm(data, desc="Analyzing data"):
        train_labels.append(example['label'])
        review_lengths.append(len(example['text'].split()))

    # 类别分布
    pos_count = train_labels.count(1)
    neg_count = train_labels.count(0)
    print(f"类别分布: Positive: {pos_count}, Negative: {neg_count}")

    # 评论长度分布
    plt.figure(figsize=(10, 5))
    plt.hist(review_lengths, bins=50, alpha=0.7)
    plt.title('Review Length Distribution')
    plt.xlabel('Review Length (number of words)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def visualize_samples(data):
    """
    Displays one positive and one negative review sample from the dataset.

    Args:
        data: The dataset to visualize samples from.
    """
    pos_sample_text = "Not found."
    neg_sample_text = "Not found."
    pos_sample_found = False
    neg_sample_found = False

    for example in data:
        if not pos_sample_found and example['label'] == 1:
            pos_sample_text = example['text']
            pos_sample_found = True
        
        if not neg_sample_found and example['label'] == 0:
            neg_sample_text = example['text']
            neg_sample_found = True
        
        if pos_sample_found and neg_sample_found:
            break

    print("\n--- Sample Reviews ---")
    print("Positive Sample:")
    print(pos_sample_text)

    print("\nNegative Sample:")
    print(neg_sample_text)
    print("--------------------")


if __name__ == "__main__":
    # Load the data
    train_data, test_data = load_imdb_data()

    # Print basic information
    print(f"\n训练集样本数量: {len(train_data)}")
    print(f"测试集样本数量: {len(test_data)}")

    # Visualize a few samples
    visualize_samples(train_data)

    # Analyze the training data distribution
    analyze_dataset_distribution(train_data)