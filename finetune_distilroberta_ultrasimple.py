"""
Fine-tune paraphrase-distilroberta-base-v1 - Ultra-Simple Version
Uses only SentenceTransformers without Trainer API
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

print("Starting fine-tuning setup...")

try:
    from sentence_transformers import SentenceTransformer, InputExample
    from sentence_transformers.losses import CosineSimilarityLoss
    from torch.utils.data import DataLoader
    import torch
    print("✓ All libraries loaded")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Configuration
CONFIG = {
    'model_name': 'paraphrase-distilroberta-base-v1',
    'output_dir': 'checkpoints/distilroberta_paraphrase_finetuned',
    'epochs': 2,
    'batch_size': 8,
}

Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 80)
print("FINE-TUNING: paraphrase-distilroberta-base-v1")
print("=" * 80)

# Load training data
print("\nLoading training data...")

csv_path = "data/raw/arXiv Scientific Research Papers Dataset.csv"
if os.path.exists(csv_path) and os.path.getsize(csv_path) > 1000:
    df = pd.read_csv(csv_path, nrows=500)
    print(f"✓ Loaded {len(df)} papers")
    
    # Get first two columns
    col1, col2 = df.columns[0], df.columns[1]
    
    train_examples = []
    for idx, row in df.iterrows():
        try:
            t1 = str(row[col1]).strip()[:300]
            t2 = str(row[col2]).strip()[:300]
            if len(t1) > 5 and len(t2) > 5:
                train_examples.append(InputExample(texts=[t1, t2], label=0.8))
                if len(train_examples) >= 100:
                    break
        except:
            pass
else:
    # Synthetic data
    train_examples = [
        InputExample(texts=["Machine learning is AI", "Artificial intelligence includes machine learning"], label=0.9),
        InputExample(texts=["Deep learning uses neural networks", "Neural networks are used in deep learning"], label=0.9),
        InputExample(texts=["NLP handles text data", "Text processing by NLP"], label=0.85),
        InputExample(texts=["Computer vision analyzes images", "Images analyzed using computer vision"], label=0.85),
        InputExample(texts=["Data science combines stats", "Statistics and coding for data science"], label=0.8),
        InputExample(texts=["Python programming", "Programming with Python"], label=0.9),
        InputExample(texts=["The weather is sunny", "Today is a sunny day"], label=0.95),
        InputExample(texts=["Dogs are animals", "Cats play with yarn"], label=0.2),
        InputExample(texts=["I love coding", "I hate sports"], label=0.1),
        InputExample(texts=["Hello world", "The world says hello"], label=0.85),
    ]

print(f"✓ Using {len(train_examples)} training examples")
print(f"  Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

# Load model
print("\nLoading model...")
model = SentenceTransformer(CONFIG['model_name'])
model.max_seq_length = 128
print(f"✓ Model loaded: {CONFIG['model_name']}")

# Create data loader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=CONFIG['batch_size'])

# Create loss
train_loss = CosineSimilarityLoss(model)

# Fine-tune
print("\nFine-tuning with fit()...")
try:
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=CONFIG['epochs'],
        warmup_steps=50,
    )
    print("✓ Fine-tuning completed")
except Exception as e:
    print(f"⚠ fit() failed: {e}")
    print("Trying manual training loop...")
    
    # Manual training loop
    from torch.optim import AdamW
    
    model.train()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            # Get embeddings
            embeddings = model.encode(
                batch['texts'],
                convert_to_tensor=True,
                batch_size=CONFIG['batch_size']
            )
            
            # Split into pairs
            embeddings1 = embeddings[::2]
            embeddings2 = embeddings[1::2]
            
            # Calculate similarity loss
            cos_sim = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
            labels = torch.tensor(batch['label'], dtype=torch.float32)
            loss = torch.nn.functional.mse_loss(cos_sim, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (batch_idx + 1) % 5 == 0:
                print(f"  Batch {batch_idx + 1}: Loss = {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"  Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")

# Save model
print(f"\nSaving model to {CONFIG['output_dir']}...")
model.save(CONFIG['output_dir'])
print("✓ Model saved")

# Test
print("\n" + "=" * 80)
print("TESTING FINE-TUNED MODEL")
print("=" * 80)

test_pairs = [
    ("machine learning", "artificial intelligence"),
    ("deep learning", "neural networks"),
    ("dogs", "cats"),
]

for sent1, sent2 in test_pairs:
    embeddings = model.encode([sent1, sent2])
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    print(f"\n'{sent1}' vs '{sent2}': {sim:.4f}")

print("\n" + "=" * 80)
print("✓ FINE-TUNING COMPLETE")
print("=" * 80)
print(f"\nModel saved to: {CONFIG['output_dir']}")
print("\nTo use it:")
print(f"  from sentence_transformers import SentenceTransformer")
print(f"  model = SentenceTransformer('{CONFIG['output_dir']}')")
