"""
Domain-Specific Fine-tuning: paraphrase-distilroberta-base-v1 on arXiv Papers
Fine-tunes the model to better understand research paper content and terminology
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

print("Starting domain-specific fine-tuning...")

# Import libraries
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
    'model_name': 'checkpoints/distilroberta_paraphrase_finetuned',
    'output_dir': 'checkpoints/distilroberta_arxiv_finetuned',
    'epochs': 3,
    'batch_size': 16,
    'warmup_steps': 100,
    'max_examples': 1000,  # Limit to 1000 examples for reasonable training time
}

Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 80)
print("DOMAIN-SPECIFIC FINE-TUNING: arXiv Research Papers")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  Base model: {CONFIG['model_name']}")
print(f"  Output: {CONFIG['output_dir']}")
print(f"  Epochs: {CONFIG['epochs']}")
print(f"  Batch size: {CONFIG['batch_size']}")
print(f"  Max examples: {CONFIG['max_examples']}")


# ============================================================================
# LOAD AND PREPARE ARXIV DATA
# ============================================================================

def load_arxiv_data() -> List[InputExample]:
    """Load arXiv papers and create training pairs."""
    print("\n" + "=" * 80)
    print("LOADING ARXIV DATA")
    print("=" * 80)
    
    csv_path = "data/raw/arXiv Scientific Research Papers Dataset.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ Dataset not found: {csv_path}")
        return []
    
    try:
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} papers")
        print(f"  Columns: {df.columns.tolist()}")
        
        # Identify relevant columns
        title_col = None
        abstract_col = None
        category_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'title' in col_lower:
                title_col = col
            elif 'abstract' in col_lower or 'summary' in col_lower:
                abstract_col = col
            elif 'category' in col_lower or 'subject' in col_lower or 'tag' in col_lower:
                category_col = col
        
        if not title_col or not abstract_col:
            print("⚠ Could not find title/abstract columns")
            return []
        
        print(f"  Using: Title='{title_col}', Abstract='{abstract_col}'")
        if category_col:
            print(f"         Category='{category_col}'")
        
        # Create training pairs
        train_examples = []
        category_groups = {}
        
        # Group papers by category
        if category_col and category_col in df.columns:
            for cat in df[category_col].unique():
                if pd.notna(cat):
                    category_groups[str(cat)] = df[df[category_col] == cat]
        
        # Strategy 1: Title-Abstract pairs (highest similarity)
        print("\n  Creating training pairs...")
        print("    [1/4] Title-Abstract pairs...")
        for idx, row in df.iterrows():
            if len(train_examples) >= CONFIG['max_examples']:
                break
            
            try:
                title = str(row[title_col]).strip()
                abstract = str(row[abstract_col]).strip()
                
                # Clean up text
                if len(title) > 10 and len(abstract) > 50:
                    # Title-Abstract: high similarity (0.85)
                    train_examples.append(InputExample(
                        texts=[title, abstract],
                        label=0.85
                    ))
            except:
                continue
        
        # Strategy 2: Abstract sentence pairs (medium similarity)
        print("    [2/4] Abstract sentence pairs...")
        for idx, row in df.iterrows():
            if len(train_examples) >= CONFIG['max_examples']:
                break
            
            try:
                abstract = str(row[abstract_col]).strip()
                sentences = [s.strip() for s in abstract.split('.') if len(s.strip()) > 20]
                
                # Take pairs of consecutive sentences
                for i in range(len(sentences) - 1):
                    if len(train_examples) >= CONFIG['max_examples']:
                        break
                    train_examples.append(InputExample(
                        texts=[sentences[i], sentences[i+1]],
                        label=0.65
                    ))
            except:
                continue
        
        # Strategy 3: Same category papers (semantic similarity)
        print("    [3/4] Same-category paper pairs...")
        if category_groups:
            for category, cat_df in category_groups.items():
                if len(train_examples) >= CONFIG['max_examples']:
                    break
                if len(cat_df) < 2:
                    continue
                
                # Sample 2 papers from same category
                sampled = cat_df.sample(min(2, len(cat_df)))
                if len(sampled) >= 2:
                    try:
                        text1 = str(sampled.iloc[0][title_col]).strip()
                        text2 = str(sampled.iloc[1][title_col]).strip()
                        
                        if len(text1) > 5 and len(text2) > 5:
                            train_examples.append(InputExample(
                                texts=[text1, text2],
                                label=0.55  # Moderate similarity
                            ))
                    except:
                        continue
        
        # Strategy 4: Synthetic negative pairs (dissimilar categories)
        print("    [4/4] Synthetic negative pairs...")
        categories = list(category_groups.keys())
        for i in range(min(len(categories) - 1, 50)):
            if len(train_examples) >= CONFIG['max_examples']:
                break
            
            try:
                cat1, cat2 = categories[i], categories[(i + 1) % len(categories)]
                df1 = category_groups[cat1]
                df2 = category_groups[cat2]
                
                if len(df1) > 0 and len(df2) > 0:
                    text1 = str(df1.sample(1).iloc[0][title_col]).strip()
                    text2 = str(df2.sample(1).iloc[0][title_col]).strip()
                    
                    if len(text1) > 5 and len(text2) > 5:
                        train_examples.append(InputExample(
                            texts=[text1, text2],
                            label=0.15  # Low similarity
                        ))
            except:
                continue
        
        print(f"\n✓ Created {len(train_examples)} training examples")
        
        # Show statistics
        if train_examples:
            labels = [ex.label for ex in train_examples]
            print(f"  Label distribution:")
            print(f"    Min: {min(labels):.2f}, Max: {max(labels):.2f}")
            print(f"    Mean: {np.mean(labels):.2f}, Std: {np.std(labels):.2f}")
        
        return train_examples
    
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


# ============================================================================
# FINE-TUNING
# ============================================================================

def finetune_on_arxiv(train_examples: List[InputExample]):
    """Fine-tune model on arXiv papers."""
    
    if not train_examples:
        print("❌ No training examples available")
        return None
    
    print("\n" + "=" * 80)
    print("FINE-TUNING MODEL")
    print("=" * 80)
    
    try:
        # Load checkpoint
        print(f"\nLoading base model: {CONFIG['model_name']}")
        model = SentenceTransformer(CONFIG['model_name'])
        print(f"✓ Model loaded: {type(model).__name__}")
        print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
        print(f"  Max sequence length: {model.max_seq_length}")
        
        # Create data loader
        print(f"\nPreparing data loader...")
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=CONFIG['batch_size']
        )
        print(f"✓ DataLoader created: {len(train_examples)} examples, batch size {CONFIG['batch_size']}")
        print(f"  Total batches: {len(train_dataloader)}")
        
        # Define loss
        train_loss = CosineSimilarityLoss(model)
        print(f"✓ Loss function: CosineSimilarityLoss")
        
        # Fine-tune
        print(f"\n" + "=" * 80)
        print(f"TRAINING FOR {CONFIG['epochs']} EPOCHS")
        print("=" * 80)
        
        try:
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=CONFIG['epochs'],
                warmup_steps=CONFIG['warmup_steps'],
                output_path=CONFIG['output_dir'],
                save_best_model=True,
                show_progress_bar=True,
            )
            print(f"\n✓ Fine-tuning completed successfully")
        except TypeError as e:
            # Fallback for older SentenceTransformers versions
            print(f"⚠ Using simplified fine-tuning (fit() parameters limited)")
            model.fit(
                [(train_dataloader, train_loss)],
                epochs=CONFIG['epochs'],
                warmup_steps=CONFIG['warmup_steps'],
            )
            model.save(CONFIG['output_dir'])
            print(f"✓ Fine-tuning completed and model saved")
        
        return model
    
    except Exception as e:
        print(f"❌ Error during fine-tuning: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model):
    """Evaluate the fine-tuned model on arXiv-specific examples."""
    if model is None:
        return
    
    print("\n" + "=" * 80)
    print("EVALUATING DOMAIN-SPECIFIC MODEL")
    print("=" * 80)
    
    # Test queries and documents from typical research paper scenarios
    test_cases = [
        {
            "query": "deep learning neural networks",
            "docs": [
                "Deep learning is a subfield of machine learning using neural networks with multiple layers",
                "Neural networks are inspired by biological neurons and can learn complex patterns",
                "Model optimization techniques improve training efficiency",
                "Python is a popular programming language",
            ],
            "expected_top": 0,  # Message 0 and 1 should rank high
        },
        {
            "query": "natural language processing text analysis",
            "docs": [
                "NLP techniques enable computers to understand human language",
                "Text analysis uses linguistic features for document classification",
                "Computer vision processes images and video data",
                "Data preprocessing is an important step in machine learning",
            ],
            "expected_top": 0,
        },
        {
            "query": "data preprocessing feature engineering",
            "docs": [
                "Feature engineering creates meaningful features from raw data",
                "Data preprocessing involves cleaning and normalization steps",
                "Model training requires quality input data",
                "Hardware acceleration speeds up computations",
            ],
            "expected_top": 0,
        },
    ]
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    print("\nTest Case Evaluations:")
    print("-" * 80)
    
    all_correct = 0
    total_cases = 0
    
    for idx, test in enumerate(test_cases, 1):
        print(f"\n[Test {idx}] Query: '{test['query']}'")
        
        # Encode
        query_emb = model.encode(test['query'])
        doc_embs = model.encode(test['docs'])
        
        # Compute similarities
        sims = cosine_similarity([query_emb], doc_embs)[0]
        
        # Rank
        ranked = np.argsort(sims)[::-1]
        
        print(f"  Results:")
        for rank, doc_idx in enumerate(ranked[:3], 1):
            marker = "✓" if (rank == 1 and doc_idx == test['expected_top']) else " "
            print(f"    {marker} Rank {rank}: (sim: {sims[doc_idx]:.4f}) {test['docs'][doc_idx][:50]}...")
        
        if ranked[0] == test['expected_top']:
            all_correct += 1
        total_cases += 1
    
    accuracy = (all_correct / total_cases * 100) if total_cases > 0 else 0
    print(f"\n{'=' * 80}")
    print(f"Evaluation Accuracy: {all_correct}/{total_cases} ({accuracy:.1f}%)")
    print(f"{'=' * 80}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        print(f"\nPython: {sys.version}")
        print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        
        # Load data
        train_examples = load_arxiv_data()
        
        if train_examples:
            # Fine-tune
            finetuned_model = finetune_on_arxiv(train_examples)
            
            # Evaluate
            if finetuned_model:
                evaluate_model(finetuned_model)
                
                print("\n" + "=" * 80)
                print("✓ DOMAIN-SPECIFIC FINE-TUNING COMPLETE")
                print("=" * 80)
                
                print(f"\n✓ New checkpoint saved to: {CONFIG['output_dir']}")
                print(f"\n✓ arXiv-specific model details:")
                print(f"  - Base model: paraphrase-distilroberta-base-v1")
                print(f"  - Training data: {len(train_examples)} research paper pairs")
                print(f"  - Training epochs: {CONFIG['epochs']}")
                print(f"  - Domain: Computer Science (arXiv papers)")
                
                print(f"\nTo use the domain-specific model:")
                print(f"  from sentence_transformers import SentenceTransformer")
                print(f"  model = SentenceTransformer('{CONFIG['output_dir']}')")
                
                print(f"\nTo switch in your system:")
                print(f"  Update src/retrieval.py:")
                print(f"    model_name = '{CONFIG['output_dir']}'")
        else:
            print("\n❌ No training data available")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n❌ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
