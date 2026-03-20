"""
Lightweight DistilBERT QA fine-tuning without Trainer class.
Uses standard PyTorch training loop for more control.
Perfect for quick testing and debugging.
"""

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import DataLoader, Dataset
import random

# Set random seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Configuration
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./checkpoints/distilbert_qa_lightweight"
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
EPOCHS = 2
NUM_SAMPLES = 500  # Small dataset for testing
MAX_LENGTH = 384


def generate_simple_qa_dataset(csv_path, num_samples=500):
    """Generate simple QA dataset from papers."""
    print(f"Generating {num_samples} QA pairs...")
    
    try:
        df = pd.read_csv(csv_path, nrows=num_samples * 2)
    except:
        print("Creating synthetic QA data...")
        qa_data = []
        for i in range(num_samples):
            qa_data.append({
                'question': f'What is the topic of document {i}?',
                'context': f'This document discusses important topics in machine learning. Topic {i} is covered extensively.',
                'answer_start': 25,
                'answer_text': 'important topics in machine learning'
            })
        return qa_data
    
    qa_data = []
    for idx, row in df.iterrows():
        if len(qa_data) >= num_samples:
            break
        
        try:
            context = str(row.get('summary', '')).strip()
            if len(context) < 50:
                continue
            
            # Simple Q&A from context
            qa_data.append({
                'question': 'What is described in this text?',
                'context': context,
                'answer_start': 0,
                'answer_text': context[:80]
            })
        except:
            continue
    
    print(f"Created {len(qa_data)} QA pairs")
    return qa_data


class SimpleQADataset(Dataset):
    """Simple QA dataset."""
    
    def __init__(self, qa_data, tokenizer, max_length=384):
        self.qa_data = qa_data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.qa_data)
    
    def __getitem__(self, idx):
        qa = self.qa_data[idx]
        
        # Tokenize question and context
        encoding = self.tokenizer(
            qa['question'],
            qa['context'],
            max_length=self.max_length,
            truncation='only_second',
            return_offsets_mapping=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Find answer position in tokens
        answer_start_char = qa['answer_start']
        answer_end_char = answer_start_char + len(qa['answer_text'])
        
        sequence_ids = encoding.sequence_ids(0)
        offset_mapping = encoding['offset_mapping'][0]
        
        start_pos = 0
        end_pos = 0
        for i, (offset_start, offset_end) in enumerate(offset_mapping):
            if offset_start <= answer_start_char < offset_end:
                start_pos = i
            if offset_start < answer_end_char <= offset_end:
                end_pos = i
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'start_positions': torch.tensor(start_pos),
            'end_positions': torch.tensor(end_pos),
        }


def train_distilbert_simple(
    model_name=MODEL_NAME,
    output_dir=OUTPUT_DIR,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
    num_samples=NUM_SAMPLES,
):
    """Simple PyTorch training loop for DistilBERT QA."""
    
    print("\n" + "="*70)
    print("DISTILBERT QA FINE-TUNING (LIGHTWEIGHT VERSION)")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model and tokenizer
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.to(device)
    
    # Prepare data
    print("\nPreparing dataset...")
    qa_data = generate_simple_qa_dataset(
        "data/raw/arXiv Scientific Research Papers Dataset.csv",
        num_samples=num_samples
    )
    
    # Split dataset
    n_train = int(len(qa_data) * 0.8)
    train_qa = qa_data[:n_train]
    val_qa = qa_data[n_train:]
    
    train_dataset = SimpleQADataset(train_qa, tokenizer, MAX_LENGTH)
    val_dataset = SimpleQADataset(val_qa, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"\nTraining for {epochs} epoch(s)...")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        total_train_loss = 0
        
        train_bar = tqdm(train_loader, desc="Training", leave=False)
        for batch in train_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
            )
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"  Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Validation", leave=False)
            for batch in val_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions,
                )
                
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"  Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  ✓ Best model saved!")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print(f"Model saved to: {output_dir}")
    print("="*70)
    
    return model, tokenizer


def test_model(model, tokenizer, model_path=OUTPUT_DIR):
    """Test the fine-tuned model."""
    print("\n" + "="*70)
    print("TESTING FINE-TUNED MODEL")
    print("="*70)
    
    # Load best model if needed
    if model is None:
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Test examples
    test_examples = [
        {
            'context': 'Machine learning is a branch of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. Deep learning is a subset of machine learning using neural networks.',
            'question': 'What is machine learning?'
        },
        {
            'context': 'Python is a popular programming language used for web development, data science, and machine learning. It has a simple syntax and large community support.',
            'question': 'What is Python used for?'
        }
    ]
    
    with torch.no_grad():
        for i, example in enumerate(test_examples, 1):
            inputs = tokenizer.encode_plus(
                example['question'],
                example['context'],
                return_tensors='pt',
                max_length=384,
                truncation=True
            )
            
            input_ids = inputs['input_ids'].to(device)
            outputs = model(input_ids=input_ids)
            
            answer_start = torch.argmax(outputs.start_logits, dim=1)
            answer_end = torch.argmax(outputs.end_logits, dim=1) + 1
            
            answer = tokenizer.decode(
                input_ids[0][answer_start[0]:answer_end[0]]
            )
            
            print(f"\nExample {i}:")
            print(f"  Q: {example['question']}")
            print(f"  A: {answer}")


if __name__ == "__main__":
    # Check GPU
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Train model
    model, tokenizer = train_distilbert_simple(
        num_samples=NUM_SAMPLES,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    
    # Test model
    test_model(model, tokenizer)
    
    print("\n✓ Fine-tuning script completed successfully!")
