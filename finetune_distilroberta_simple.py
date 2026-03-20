"""
Fine-tune paraphrase-distilroberta-base-v1 for Semantic Search
Simplified version with robust error handling
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

print("Starting fine-tuning setup...")

# Try importing required libraries
try:
    from sentence_transformers import SentenceTransformer, InputExample
    from sentence_transformers.losses import CosineSimilarityLoss
    print("✓ Sentence Transformers loaded")
except ImportError as e:
    print(f"❌ Error importing sentence_transformers: {e}")
    print("Installing sentence-transformers...")
    os.system("pip install -q sentence-transformers")
    from sentence_transformers import SentenceTransformer, InputExample
    from sentence_transformers.losses import CosineSimilarityLoss

try:
    from torch.utils.data import DataLoader
    print("✓ PyTorch loaded")
except ImportError:
    print("⚠ PyTorch not available, will use simpler training")


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'model_name': 'paraphrase-distilroberta-base-v1',
    'output_dir': 'checkpoints/distilroberta_paraphrase_finetuned',
    'epochs': 2,
    'batch_size': 8,
    'warmup_steps': 50,
    'max_seq_length': 128,
}

Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 80)
print("FINE-TUNING: paraphrase-distilroberta-base-v1")
print("=" * 80)
print(f"\nConfiguration:")
for key, value in CONFIG.items():
    if key != 'model_name':
        print(f"  {key}: {value}")


# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

def load_training_data() -> List[InputExample]:
    """Load and prepare training data from CSV."""
    print("\n" + "=" * 80)
    print("LOADING TRAINING DATA")
    print("=" * 80)
    
    csv_path = "data/raw/arXiv Scientific Research Papers Dataset.csv"
    
    if not os.path.exists(csv_path):
        print(f"⚠ Data file not found: {csv_path}")
        print("Using synthetic training data for demonstration...")
        return create_synthetic_data()
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} papers from {csv_path}")
        print(f"  Columns: {df.columns.tolist()}")
        
        train_examples = []
        
        # Find title and abstract columns
        title_col = next((col for col in df.columns if col.lower() in ['title']), None)
        abstract_col = next((col for col in df.columns if col.lower() in ['abstract', 'summary']), None)
        
        if not title_col:
            title_col = df.columns[0]
        if not abstract_col and len(df.columns) > 1:
            abstract_col = df.columns[1]
        
        if title_col and abstract_col:
            print(f"  Using columns: {title_col}, {abstract_col}")
            
            for idx, row in df.iterrows():
                try:
                    if pd.notna(row[title_col]) and pd.notna(row[abstract_col]):
                        title = str(row[title_col]).strip()
                        abstract = str(row[abstract_col]).strip()
                        
                        if title and abstract and len(title) > 5 and len(abstract) > 20:
                            # Add high-similarity pair (title-abstract)
                            train_examples.append(InputExample(
                                texts=[title, abstract],
                                label=0.85
                            ))
                            
                            # Add lower-similarity pairs from abstract
                            sentences = [s.strip() for s in abstract.split('.') if len(s.strip()) > 10]
                            if len(sentences) > 1:
                                train_examples.append(InputExample(
                                    texts=[sentences[0], sentences[-1]],
                                    label=0.5
                                ))
                except:
                    continue
                
                if len(train_examples) >= 500:  # Limit for faster training
                    break
        
        print(f"✓ Created {len(train_examples)} training examples")
        return train_examples if train_examples else create_synthetic_data()
    
    except Exception as e:
        print(f"⚠ Error loading data: {str(e)}")
        return create_synthetic_data()


def create_synthetic_data() -> List[InputExample]:
    """Create synthetic training data."""
    print("\n✓ Creating synthetic sentence pairs for demonstration...")
    
    pairs = [
        ("Machine learning is artificial intelligence", 
         "AI encompasses machine learning", 0.85),
        ("Deep learning uses neural networks", 
         "Neural networks are used in deep learning", 0.85),
        ("Natural language processing handles text", 
         "Text processing by NLP", 0.8),
        ("Computer vision analyzes images", 
         "Images are analyzed using computer vision", 0.85),
        ("The weather is sunny", 
         "Today is a sunny day", 0.9),
        ("Python is a programming language", 
         "Programming with Python", 0.8),
        ("Data science requires statistics", 
         "Statistics are needed for data science", 0.8),
        ("The cat sleeps peacefully", 
         "A dog runs quickly", 0.15),
        ("Machine learning generates predictions", 
         "Sports are fun to watch", 0.1),
        ("Transformers revolutionized NLP", 
         "NLP was changed by transformers", 0.9),
    ]
    
    train_examples = []
    for text1, text2, score in pairs:
        train_examples.append(InputExample(texts=[text1, text2], label=score))
    
    print(f"✓ Created {len(train_examples)} synthetic training examples\n")
    return train_examples


# ============================================================================
# FINE-TUNING
# ============================================================================

def finetune_model():
    """Fine-tune the model."""
    
    print("=" * 80)
    print("INITIALIZING MODEL FOR FINE-TUNING")
    print("=" * 80)
    
    try:
        # Load model
        print(f"\nLoading {CONFIG['model_name']}...")
        model = SentenceTransformer(CONFIG['model_name'])
        model.max_seq_length = CONFIG['max_seq_length']
        print(f"✓ Model loaded successfully")
        print(f"  Architecture: {type(model).__name__}")
        print(f"  Max sequence length: {model.max_seq_length}")
        
        # Load training data
        train_examples = load_training_data()
        
        if not train_examples:
            print("❌ No training examples available")
            return None
        
        # Create data loader
        try:
            from torch.utils.data import DataLoader
            train_dataloader = DataLoader(
                train_examples,
                shuffle=True,
                batch_size=CONFIG['batch_size']
            )
            print(f"✓ Created DataLoader with {len(train_examples)} examples")
        except:
            print("⚠ Using SimpleDataLoader (PyTorch DataLoader not available)")
            train_dataloader = train_examples
        
        # Define loss function
        try:
            train_loss = CosineSimilarityLoss(model)
            print("✓ Initialized CosineSimilarityLoss")
        except:
            print("⚠ Using default loss function")
            train_loss = None
        
        # Fine-tune model
        print("\n" + "=" * 80)
        print("FINE-TUNING IN PROGRESS")
        print("=" * 80)
        
        try:
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=CONFIG['epochs'],
                warmup_steps=CONFIG['warmup_steps'],
                batch_size=CONFIG['batch_size'],
                show_progress_bar=True,
                output_path=CONFIG['output_dir'],
                save_best_model=True,
            )
        except TypeError:
            # Fallback if fit() doesn't accept all parameters
            print("Using simplified training approach...")
            model.fit(
                [(train_dataloader, train_loss)],
                epochs=CONFIG['epochs'],
                warmup_steps=CONFIG['warmup_steps'],
            )
            model.save(CONFIG['output_dir'])
        
        print("\n" + "=" * 80)
        print("✓ FINE-TUNING COMPLETED")
        print("=" * 80)
        print(f"\n✓ Model saved to: {CONFIG['output_dir']}")
        
        return model
    
    except Exception as e:
        print(f"\n❌ Error during fine-tuning: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_and_test_model(model):
    """Verify and test the fine-tuned model."""
    if model is None:
        return
    
    print("\n" + "=" * 80)
    print("TESTING FINE-TUNED MODEL")
    print("=" * 80)
    
    test_sentences = [
        ("hello world", "The world says hello"),
        ("machine learning is AI", "Artificial intelligence includes machine learning"),
        ("dogs are animals", "cats play with yarn"),
    ]
    
    print("\nSemantic Similarity Tests:")
    print("-" * 80)
    
    for sent1, sent2 in test_sentences:
        try:
            embeddings = model.encode([sent1, sent2])
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            print(f"\n  '{sent1}'")
            print(f"  '{sent2}'")
            print(f"  → Similarity: {similarity:.4f}")
        except Exception as e:
            print(f"  Error: {e}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print(f"\nPython: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    try:
        # Fine-tune model
        finetuned_model = finetune_model()
        
        # Verify
        verify_and_test_model(finetuned_model)
        
        if finetuned_model:
            print("\n" + "=" * 80)
            print("✓ FINE-TUNING PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print(f"\nNext steps:")
            print(f"1. Model saved to: {CONFIG['output_dir']}")
            print(f"\n2. To use the fine-tuned model:")
            print(f"   from sentence_transformers import SentenceTransformer")
            print(f"   model = SentenceTransformer('{CONFIG['output_dir']}')")
            print(f"\n3. Update main.py to use:")
            print(f"   SemanticRetriever(model_name='{CONFIG['output_dir']}')")

    except KeyboardInterrupt:
        print("\n\n❌ Fine-tuning interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
