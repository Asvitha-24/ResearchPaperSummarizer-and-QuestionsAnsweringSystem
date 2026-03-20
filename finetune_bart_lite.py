"""
Lightweight fine-tuning script for BART - simpler version for quick testing.
Good for initial experimentation before full training.
"""

import os
import sys
import torch
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset, DatasetDict
import numpy as np
from tqdm import tqdm


def prepare_minimal_dataset(csv_path: str, num_samples: int = 1000):
    """
    Load minimal dataset for quick testing.
    
    Args:
        csv_path: Path to CSV file
        num_samples: Number of samples to load
        
    Returns:
        DatasetDict with train/val samples
    """
    print(f"Loading {num_samples} samples from {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path, nrows=num_samples)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        print("\nCreating synthetic data for demonstration...")
        # Fallback: create synthetic data
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
    
    # Convert to datasets
    train_dataset = Dataset.from_dict({
        'text': [s['text'] for s in train_samples],
        'summary': [s['summary'] for s in train_samples]
    })
    
    val_dataset = Dataset.from_dict({
        'text': [s['text'] for s in val_samples],
        'summary': [s['summary'] for s in val_samples]
    })
    
    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })


def preprocess_data(examples, tokenizer, max_input_length=512, max_target_length=128):
    """Tokenize text and summaries."""
    
    model_inputs = tokenizer(
        examples['text'],
        max_length=max_input_length,
        truncation=True,
        padding='max_length'
    )
    
    # Tokenize labels
    labels = tokenizer(
        examples['summary'],
        max_length=max_target_length,
        truncation=True,
        padding='max_length'
    )
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


def train_minimal_bart(
    csv_path: str,
    num_samples: int = 1000,
    num_epochs: int = 2,
    batch_size: int = 4,
    output_dir: str = "./checkpoints/bart_finetuned_lite"
):
    """
    Minimal BART fine-tuning pipeline.
    
    Args:
        csv_path: Path to CSV dataset
        num_samples: Number of samples to use
        num_epochs: Training epochs
        batch_size: Batch size
        output_dir: Output directory
    """
    
    print("=" * 80)
    print("BART Fine-tuning (Lightweight)")
    print("=" * 80)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load model and tokenizer
    print("\nLoading BART model...")
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    
    # Load data
    print("\nLoading data...")
    datasets = prepare_minimal_dataset(csv_path, num_samples)
    
    # Preprocess
    print("Preprocessing data...")
    preprocess_fn = lambda examples: preprocess_data(
        examples, tokenizer, max_input_length=512, max_target_length=128
    )
    
    datasets = datasets.map(
        preprocess_fn,
        batched=True,
        remove_columns=['text', 'summary'],
        desc="Tokenizing"
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True
    )
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        learning_rate=5e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        logging_steps=10,
        predict_with_generate=True,
        fp16=device == "cuda",
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting training...")
    print("-" * 80)
    trainer.train()
    print("-" * 80)
    
    # Save
    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n✅ Training complete!")
    print(f"Model saved to: {output_dir}")
    
    return model, tokenizer


def test_finetuned_model(model, tokenizer, device="cuda"):
    """Test the fine-tuned model."""
    
    print("\n" + "=" * 80)
    print("Testing Fine-tuned Model")
    print("=" * 80)
    
    test_texts = [
        "This is a revolutionary approach to deep learning using transformer architectures with attention mechanisms for better performance.",
        "We propose a novel method for natural language understanding using pre-trained language models fine-tuned on downstream tasks.",
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
    
    parser = argparse.ArgumentParser(description="Lightweight BART fine-tuning")
    parser.add_argument("--csv", type=str, default="data/raw/arXiv Scientific Research Papers Dataset.csv",
                       help="Path to CSV dataset")
    parser.add_argument("--samples", type=int, default=1000,
                       help="Number of samples to use")
    parser.add_argument("--epochs", type=int, default=2,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--output", type=str, default="./checkpoints/bart_finetuned_lite",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Check dataset exists
    if not os.path.exists(args.csv):
        print(f"⚠️  Dataset not found: {args.csv}")
        print("Creating synthetic dataset for demonstration...")
    
    # Train
    model, tokenizer = train_minimal_bart(
        csv_path=args.csv,
        num_samples=args.samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output
    )
    
    # Test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_finetuned_model(model, tokenizer, device=device)
