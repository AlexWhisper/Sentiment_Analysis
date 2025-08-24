#!/usr/bin/env python3
"""
Load and use saved neural network model for sentiment analysis prediction
"""

import json
import os
import torch
from model import RNN
import re

def load_model_and_vocab(model_dir='./saved_models'):
    """
    Load the saved model and vocabulary.
    """
    # Load model info
    with open(os.path.join(model_dir, 'model_info.json'), 'r') as f:
        model_info = json.load(f)

    # Get model parameters from info
    vocab_size = model_info['vocab_size']
    embedding_dim = model_info['embedding_dim']
    hidden_dim = model_info['hidden_dim']
    output_dim = model_info['output_dim']
    n_layers = model_info['n_layers']
    pad_idx = model_info['pad_idx']
    vocab = model_info['vocab']

    # Create network with saved structure
    network = RNN(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, pad_idx=pad_idx)
    
    # Load PyTorch model state
    model_path = os.path.join(model_dir, 'rnn_model.pth')
    network.load_state_dict(torch.load(model_path))
    network.eval()
    
    print(f"Model loaded from {model_path}")
    return network, vocab

def tokenize_text(text):
    """
    Simple text tokenizer.
    """
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    return text.lower().strip().split()

def predict_sentiment(network, vocab, text):
    """
    Predict the sentiment of a given text.
    """
    # Tokenize text
    tokens = tokenize_text(text)
    if not tokens:
        return "Neutral", 0.5, [0.5, 0.5]

    # Convert tokens to indices
    indexed = [vocab.get(t, 0) for t in tokens]  # Use 0 for unknown tokens
    
    # Convert to tensor
    tensor = torch.LongTensor(indexed).unsqueeze(0)  # Add batch dimension
    
    # Get prediction
    with torch.no_grad():
        prediction = network(tensor)
        probabilities = torch.softmax(prediction, dim=1)
    
    confidence, predicted_class = torch.max(probabilities, 1)
    
    # Map class index to label
    class_labels = ['Negative', 'Positive']
    sentiment = class_labels[predicted_class.item()]
    
    return sentiment, confidence.item(), probabilities.squeeze().tolist()

def main():
    print("=" * 60)
    print("Sentiment Analysis Inference")
    print("=" * 60)
    
    # Check if saved model exists
    model_dir = './saved_models'
    if not os.path.exists(os.path.join(model_dir, 'model_info.json')):
        print("No saved model found!")
        print("Please run 'train_and_save.py' first to train and save a model.")
        return
    
    # Load the saved model and vocabulary
    print("\n1. Loading saved model and vocabulary...")
    network, vocab = load_model_and_vocab(model_dir)
    
    # Interactive prediction mode
    print("\n2. Interactive prediction mode...")
    print("Enter a movie review to get its sentiment, or 'q' to quit.")
    
    while True:
        try:
            user_input = input("\nReview: ")
            if user_input.lower() == 'q':
                break
            
            if not user_input.strip():
                print("Please enter some text.")
                continue

            # Make prediction
            sentiment, confidence, probabilities = predict_sentiment(network, vocab, user_input)
            
            print(f"  -> Sentiment: {sentiment}")
            print(f"  -> Confidence: {confidence:.3f}")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
    
    print("\n" + "=" * 60)
    print("Inference completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
