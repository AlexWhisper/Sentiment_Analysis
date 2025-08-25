#!/usr/bin/env python3
"""
IMDb sentiment analysis example
Using a PyTorch RNN for sentiment analysis
"""
import numpy as np
import matplotlib.pyplot as plt
from model import RNN
from imdb_loader import load_imdb_data
import time
import json
import os
import torch
from collections import Counter
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def build_vocab(data, min_freq=5):
    """Build a vocabulary from the text data."""
    counter = Counter()
    for example in data:
        counter.update(example['text'].split())
    
    vocab = {"<unk>": 0, "<pad>": 1}
    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
            
    return vocab

def collate_fn(batch, vocab, max_len=100):
    """Process a batch of data for the DataLoader."""
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    texts = []
    for item in batch:
        tokens = item['text'].split()
        indexed = [vocab.get(t, vocab["<unk>"]) for t in tokens]
        if len(indexed) > max_len:
            indexed = indexed[:max_len]
        texts.append(torch.tensor(indexed))
        
    texts = pad_sequence(texts, batch_first=True, padding_value=vocab["<pad>"])
    return labels, texts

def save_model_info(network, vocab, model_dir=None):
    """
    Save the model and its configuration (including vocabulary).
    """
    # 设置默认保存目录到根目录的saved_models/rnn文件夹
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'rnn')
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save PyTorch model weights
    torch.save(network.state_dict(), os.path.join(model_dir, 'rnn_model.pth'))

    # Combine model parameters and vocabulary into a single dictionary
    model_info = {
        'vocab_size': network.vocab_size,
        'embedding_dim': network.embedding_dim,
        'hidden_dim': network.hidden_dim,
        'output_dim': network.output_dim,
        'n_layers': network.n_layers,
        'pad_idx': network.pad_idx,
        'vocab': vocab
    }

    # Save the combined information to a single JSON file
    with open(os.path.join(model_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"Model saved to {model_dir}/")
    print(f"   - Model info (structure and vocab): model_info.json")
    print(f"   - PyTorch model: rnn_model.pth")

def main():
    print("=" * 60)
    print("IMDb Sentiment Analysis RNN Demo (PyTorch)")
    print("=" * 60)
    
    # 1. Load IMDb data
    print("\n1. Loading IMDb dataset...")
    train_data, test_data = load_imdb_data()
    
    # 2. Build vocabulary
    print("\n2. Building vocabulary...")
    vocab = build_vocab(train_data)
    
    # 3. Create DataLoaders
    print("\n3. Creating DataLoaders...")
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=lambda b: collate_fn(b, vocab))
    test_loader = DataLoader(test_data, batch_size=32, collate_fn=lambda b: collate_fn(b, vocab))
    
    # 4. Create neural network
    print("\n4. Creating neural network...")
    embedding_dim = 100
    hidden_size = 128
    num_layers = 2
    num_classes = 2
    pad_idx = vocab["<pad>"]
    
    network = RNN(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        pad_idx=pad_idx
    )
    
    # 5. Train network
    print("\n5. Starting training...")
    losses = network.train_model(train_loader, epochs=10, verbose=True)
    
    # 6. Evaluate model
    print("\n6. Evaluating model performance...")
    train_accuracy = network.accuracy(train_loader, vocab)
    print(f"Training accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    
    test_accuracy = network.accuracy(test_loader, vocab)
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # 7. Save the trained model
    print("\n7. Saving the trained model...")
    save_model_info(network, vocab, model_dir='./saved_models')

if __name__ == "__main__":
    main()
