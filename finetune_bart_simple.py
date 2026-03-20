"""
Simplified BART fine-tuning without using Trainer class.
Works with standard PyTorch training loop.
"""

import os
import sys
import torch
import pandas as pd
from pathlib import Path
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from datasets import Dataset
from torch.utils.data import DataLoader
import numpy as np


def prepare_dataset(csv_path: str, num_samples: int = 1000):
    """Load minimal dataset for quick testing."""
    print(f"Loading {num_samples} samples from {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path, nrows=num_samples)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        print("\nCreating synthetic data for demonstration...")
        data = {
            'title': [f'Research Paper {i}' for i in range(num_samples)],
            'summary': [f'This paper discusses important topics in AI and ML for sample {i}.' for i in range(num_samples)]
        }
        df = pd.DataFrame(data)
    
    # Clean data
    df = df.dropna(subset=['summary'])
    df = df[df['summary'].str.len() > 20]
    
    # Create training samples
    train_data = []
    for idx, row in df.head(num_samples).iterrows():
        try:
            title = str(row.get('title', '')).strip()
            summary = str(row.get('summary', '')).strip()
            
            if title and summary and len(summary) > 20:
                train_data.append({
                    'text': f"summarize: {title}. {summary[:300]}",
                    'summary': summary[:256]
                })
        except:
            continue
    
    print(f"Created {len(train_data)} training samples")
    
    if len(train_data) < 100:
        print("⚠️  Warning: Less than 100 samples!")
    
    # Split
    n_train = int(len(train_data) * 0.8)
    train_samples = train_data[:n_train]
    val_samples = train_data[n_train:]
    
    return train_samples, val_samples


class BartDataset(torch.utils.data.Dataset):
    """Custom dataset for BART fine-tuning."""
    
    def __init__(self, samples, tokenizer, max_input_length=512, max_target_length=128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize input
        input_encoding = self.tokenizer(
            sample['text'],
            max_length=self.max_input_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            sample['summary'],
            max_length=self.max_target_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze(),
        }


def train_bart_simple(
    csv_path: str,
    num_samples: int = 1000,
    num_epochs: int = 2,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    output_dir: str = "./checkpoints/bart_finetuned_simple"
):
    """Simple BART fine-tuning without Trainer."""
    
    print("=" * 80)
    print("BART Fine-tuning (Simple - No Trainer)")
    print("=" * 80)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    print("\nLoading BART model...")
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    
    # Load data
    print("\nLoading data...")
    train_samples, val_samples = prepare_dataset(csv_path, num_samples)
    
    # Create datasets
    train_dataset = BartDataset(train_samples, tokenizer)
    val_dataset = BartDataset(val_samples, tokenizer)
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            print(f"✅ Saving best model (val_loss: {best_loss:.4f})...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
    
    print("\n" + "=" * 80)
    print("✅ Training complete!")
    print("=" * 80)
    
    return model, tokenizer, output_dir


def test_generation(model, tokenizer, device="cuda"):
    """Test the model's generation capability."""
    
    print("\n" + "=" * 80)
    print("Testing Model Generation")
    print("=" * 80)
    
    test_texts = [
        "This is a revolutionary approach to deep learning using transformer architectures with attention mechanisms for improved performance on various tasks.",
        "We propose a novel deep learning method for natural language understanding that leverages pre-trained language models fine-tuned on downstream applications.",
    ]
    
    model.eval()
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}:")
        print(f"Input: {text}")
        
        inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
        inputs = inputs.to(device)
        
        with torch.no_grad():
            summary_ids = model.generate(
                inputs,
                max_length=100,
                min_length=20,
                num_beams=4,
                early_stopping=True
            )
        
        summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]
        print(f"Summary: {summary}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple BART fine-tuning")
    parser.add_argument("--csv", type=str, default="data/raw/arXiv Scientific Research Papers Dataset.csv",
                       help="Path to CSV dataset")
    parser.add_argument("--samples", type=int, default=1000,
                       help="Number of samples to use")
    parser.add_argument("--epochs", type=int, default=2,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--output", type=str, default="./checkpoints/bart_finetuned_simple",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Check dataset exists
    if not os.path.exists(args.csv):
        print(f"⚠️  Dataset not found: {args.csv}")
        print("Creating synthetic dataset for demonstration...")
    
    # Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, output_dir = train_bart_simple(
        csv_path=args.csv,
        num_samples=args.samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output
    )
    
    # Test
    test_generation(model, tokenizer, device=device)
    
    print("\n✅ All done!")
