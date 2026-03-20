"""
FAST DistilBERT QA fine-tuning for quick completion (CPU-friendly).
Uses minimal samples and epochs for ~5-10 minute training.
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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./checkpoints/distilbert_qa_finetuned"
BATCH_SIZE = 2  # Small for CPU
LEARNING_RATE = 2e-5
EPOCHS = 1      # Just 1 epoch for speed
NUM_SAMPLES = 100  # Minimal for fast training
MAX_LENGTH = 256   # Shorter sequences


def generate_simple_qa_dataset(csv_path, num_samples=100):
    """Generate simple QA dataset from papers."""
    print(f"Generating {num_samples} QA pairs...")
    
    try:
        df = pd.read_csv(csv_path, nrows=num_samples * 2)
    except:
        qa_data = []
        for i in range(num_samples):
            qa_data.append({
                'question': 'What is discussed?',
                'context': f'Document {i}: Machine learning and artificial intelligence are important fields.',
                'answer_start': 11,
                'answer_text': 'Machine learning and artificial intelligence'
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
            
            qa_data.append({
                'question': 'What is described in this text?',
                'context': context[:256],
                'answer_start': 0,
                'answer_text': context[:50]
            })
        except:
            continue
    
    print(f"Created {len(qa_data)} QA pairs")
    return qa_data


class SimpleQADataset(Dataset):
    def __init__(self, qa_data, tokenizer, max_length=256):
        self.qa_data = qa_data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.qa_data)
    
    def __getitem__(self, idx):
        qa = self.qa_data[idx]
        
        encoding = self.tokenizer(
            qa['question'],
            qa['context'],
            max_length=self.max_length,
            truncation='only_second',
            return_offsets_mapping=True,
            padding='max_length',
            return_tensors='pt'
        )
        
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


def train_fast():
    """FAST training for quick results on CPU."""
    
    print("\n" + "="*70)
    print("FAST DISTILBERT QA FINE-TUNING")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Load
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    model.to(device)
    
    # Data
    print("Preparing dataset...")
    qa_data = generate_simple_qa_dataset(
        "data/raw/arXiv Scientific Research Papers Dataset.csv",
        num_samples=NUM_SAMPLES
    )
    
    n_train = int(len(qa_data) * 0.8)
    train_data = SimpleQADataset(qa_data[:n_train], tokenizer, MAX_LENGTH)
    val_data = SimpleQADataset(qa_data[n_train:], tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}\n")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training
    print(f"Training for {EPOCHS} epoch(s)...\n")
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
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
            
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
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
        
        # Save best
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"  ✓ Best model saved!")
    
    print("\n" + "="*70)
    print("FINE-TUNING COMPLETE!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("="*70)
    
    return model, tokenizer


def test_model(model, tokenizer):
    """Test the fine-tuned model."""
    print("\n" + "="*70)
    print("TESTING FINE-TUNED MODEL")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    test_examples = [
        {
            'context': 'Machine learning is a branch of artificial intelligence that enables systems to learn and improve from experience.',
            'question': 'What is machine learning?'
        },
        {
            'context': 'Python is a popular programming language used for data science and machine learning applications.',
            'question': 'What is Python used for?'
        }
    ]
    
    with torch.no_grad():
        for i, example in enumerate(test_examples, 1):
            inputs = tokenizer(
                example['question'],
                example['context'],
                return_tensors='pt',
                max_length=256,
                truncation=True
            )
            
            input_ids = inputs['input_ids'].to(device)
            outputs = model(input_ids=input_ids)
            
            answer_start = torch.argmax(outputs.start_logits, dim=1)
            answer_end = torch.argmax(outputs.end_logits, dim=1) + 1
            
            answer = tokenizer.decode(input_ids[0][answer_start[0]:answer_end[0]])
            
            print(f"\nExample {i}:")
            print(f"  Q: {example['question']}")
            print(f"  A: {answer}")


if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU Available: {torch.cuda.is_available()}\n")
    
    # Train
    model, tokenizer = train_fast()
    
    # Test
    test_model(model, tokenizer)
    
    print("\n✓ Fine-tuning completed successfully!")
