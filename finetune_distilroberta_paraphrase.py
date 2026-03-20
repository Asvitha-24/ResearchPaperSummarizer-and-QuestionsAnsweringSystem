"""
Fine-tune paraphrase-distilroberta-base-v1 for Semantic Search
Optimized for fast training with good performance
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Sentence Transformers imports
from sentence_transformers import SentenceTransformer, InputExample, models
from sentence_transformers.losses import CosineSimilarityLoss, MultipleNegativesRankingLoss
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'model_name': 'paraphrase-distilroberta-base-v1',
    'output_dir': 'checkpoints/distilroberta_paraphrase_finetuned',
    'epochs': 2,
    'batch_size': 16,
    'warmup_steps': 100,
    'max_seq_length': 128,
    'weight_decay': 0.01,
    'learning_rate': 2e-5,
    'scheduler': 'WarmupLinear',
}

# Ensure output directory exists
Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 80)
print("FINE-TUNING: paraphrase-distilroberta-base-v1")
print("=" * 80)
print(f"\nConfiguration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")


# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

def load_training_data() -> List[InputExample]:
    """Load and prepare training data from CSV."""
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    csv_path = "data/raw/arXiv Scientific Research Papers Dataset.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ Data file not found: {csv_path}")
        print("Creating synthetic training data for demonstration...")
        return create_synthetic_data()
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} papers from {csv_path}")
        print(f"  Columns: {df.columns.tolist()}")
        
        # Create training examples from title and abstract
        train_examples = []
        
        # Get title column (try common names)
        title_col = None
        for col in ['title', 'Title', 'TITLE']:
            if col in df.columns:
                title_col = col
                break
        
        # Get abstract column
        abstract_col = None
        for col in ['abstract', 'Abstract', 'ABSTRACT', 'summary', 'Summary']:
            if col in df.columns:
                abstract_col = col
                break
        
        if title_col and abstract_col:
            print(f"  Using columns: {title_col}, {abstract_col}")
            
            for idx, row in df.iterrows():
                if pd.notna(row[title_col]) and pd.notna(row[abstract_col]):
                    title = str(row[title_col]).strip()
                    abstract = str(row[abstract_col]).strip()
                    
                    if title and abstract:
                        # Create similar pairs with high score
                        train_examples.append(InputExample(
                            texts=[title, abstract],
                            label=0.8
                        ))
                        
                        # Create pairs from abstract sentences (split by period)
                        sentences = abstract.split('.')
                        if len(sentences) > 1:
                            sent1 = sentences[0].strip()
                            sent2 = sentences[1].strip()
                            if sent1 and sent2:
                                train_examples.append(InputExample(
                                    texts=[sent1, sent2],
                                    label=0.6
                                ))
                
                if len(train_examples) >= 1000:  # Limit to 1000 examples for speed
                    break
        else:
            print(f"❌ Could not find title/abstract columns")
            return create_synthetic_data()
        
        print(f"✓ Created {len(train_examples)} training examples")
        return train_examples
    
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return create_synthetic_data()


def create_synthetic_data() -> List[InputExample]:
    """Create synthetic training data for demonstration."""
    print("Creating synthetic sentence pairs...")
    
    pairs = [
        ("Machine learning is a subset of artificial intelligence", 
         "Machine learning is part of AI"),
        ("Deep learning uses neural networks", 
         "Neural networks power deep learning"),
        ("Natural language processing handles text data", 
         "NLP processes written language"),
        ("Computer vision analyzes images", 
         "Images are analyzed by computer vision"),
        ("The weather is sunny today", 
         "It's a sunny day"),
        ("Dogs are loyal pets", 
         "Pets that are dogs are loyal"),
        ("Python is a programming language", 
         "Programming with Python"),
        ("Data science combines statistics and programming", 
         "Statistics and coding are parts of data science"),
    ]
    
    train_examples = []
    for sent1, sent2 in pairs:
        # High similarity pair
        train_examples.append(InputExample(texts=[sent1, sent2], label=0.85))
        
        # Create negative examples
        if len(train_examples) > 0:
            for other_sent1, _ in pairs[::2]:
                if other_sent1 != sent1:
                    train_examples.append(InputExample(
                        texts=[sent1, other_sent1], 
                        label=0.2
                    ))
                    break
    
    print(f"✓ Created {len(train_examples)} synthetic training examples")
    return train_examples


# ============================================================================
# FINE-TUNING
# ============================================================================

def finetune_model():
    """Fine-tune the model."""
    
    print("\n" + "=" * 80)
    print("INITIALIZING MODEL")
    print("=" * 80)
    
    # Load base model
    print(f"\nLoading {CONFIG['model_name']}...")
    model = SentenceTransformer(CONFIG['model_name'])
    model.max_seq_length = CONFIG['max_seq_length']
    print(f"✓ Model loaded")
    print(f"  Max seq length: {model.max_seq_length}")
    
    # Load training data
    train_examples = load_training_data()
    
    if not train_examples:
        print("❌ No training examples found")
        return
    
    # Create data loader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=CONFIG['batch_size']
    )
    
    # Define loss function
    # CosineSimilarityLoss is recommended for paraphrase models
    train_loss = CosineSimilarityLoss(model)
    
    # Create evaluation dataset (use same as training for demo)
    sentences1 = [ex.texts[0] for ex in train_examples[:100]]
    sentences2 = [ex.texts[1] for ex in train_examples[:100]]
    scores = [ex.label for ex in train_examples[:100]]
    
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1, sentences2, scores,
        batch_size=32,
        main_similarity='cosine',
        name='eval'
    )
    
    # Fine-tune model
    print("\n" + "=" * 80)
    print("FINE-TUNING")
    print("=" * 80)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=CONFIG['epochs'],
        warmup_steps=CONFIG['warmup_steps'],
        evaluator=evaluator,
        evaluation_steps=len(train_dataloader) // 2,
        output_path=CONFIG['output_dir'],
        save_best_model=True,
        show_progress_bar=True,
        checkpoint_save_steps=len(train_dataloader),
        checkpoint_save_total_limit=2,
    )
    
    print("\n" + "=" * 80)
    print("✓ FINE-TUNING COMPLETE")
    print("=" * 80)
    print(f"\nModel saved to: {CONFIG['output_dir']}")
    
    return model


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model):
    """Evaluate the fine-tuned model."""
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    
    test_pairs = [
        ("Machine learning is powerful", "ML is strong", 0.9),
        ("The cat is sleeping", "Dogs are running", 0.1),
        ("Python programming language", "Python is for coding", 0.85),
    ]
    
    print("\nSemantic Similarity Test:")
    print("-" * 80)
    
    for sent1, sent2, expected in test_pairs:
        embedding1 = model.encode(sent1, convert_to_tensor=True)
        embedding2 = model.encode(sent2, convert_to_tensor=True)
        
        similarity = torch.nn.functional.cosine_similarity(
            embedding1.unsqueeze(0),
            embedding2.unsqueeze(0)
        ).item()
        
        print(f"\nSentence 1: {sent1}")
        print(f"Sentence 2: {sent2}")
        print(f"  Expected Similarity: {expected:.2f}")
        print(f"  Predicted Similarity: {similarity:.2f}")
        print(f"  Match: {'✓' if abs(similarity - expected) < 0.3 else '✗'}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        # Check GPU availability
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\nUsing device: {device}")
        
        # Fine-tune model
        finetuned_model = finetune_model()
        
        # Evaluate
        if finetuned_model:
            evaluate_model(finetuned_model)
            
            print("\n" + "=" * 80)
            print("✓ FINE-TUNING PIPELINE COMPLETE")
            print("=" * 80)
            print(f"\nLatest checkpoint: {CONFIG['output_dir']}")
            print("\nTo use the fine-tuned model:")
            print(f"  from sentence_transformers import SentenceTransformer")
            print(f"  model = SentenceTransformer('{CONFIG['output_dir']}')")
            
    except Exception as e:
        print(f"\n❌ Error during fine-tuning: {str(e)}")
        import traceback
        traceback.print_exc()
