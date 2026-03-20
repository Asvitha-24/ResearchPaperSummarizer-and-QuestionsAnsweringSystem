"""
Domain-Specific Fine-tuning: Manual Training Loop (No Accelerate dependency)
Fine-tunes the model on arXiv papers using a simple PyTorch training loop
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import json

print("Starting domain-specific fine-tuning (manual loop)...")

# Import libraries
try:
    from sentence_transformers import SentenceTransformer, InputExample
    from sentence_transformers.losses import CosineSimilarityLoss
    from torch.utils.data import DataLoader
    import torch
    from torch import nn
    print("✓ All libraries loaded")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Configuration
CONFIG = {
    'model_name': 'checkpoints/distilroberta_paraphrase_finetuned',
    'output_dir': 'checkpoints/distilroberta_arxiv_finetuned',
    'epochs': 3,
    'batch_size': 32,
    'learning_rate': 2e-5,
    'max_examples': 800,
}

Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 80)
print("DOMAIN-SPECIFIC FINE-TUNING: arXiv Research Papers (Manual Loop)")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  Base model: {CONFIG['model_name']}")
print(f"  Output: {CONFIG['output_dir']}")
print(f"  Epochs: {CONFIG['epochs']}")
print(f"  Batch size: {CONFIG['batch_size']}")
print(f"  Learning rate: {CONFIG['learning_rate']}")
print(f"  Max examples: {CONFIG['max_examples']}")
print(f"  Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")


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
        
        # Use title and summary columns
        title_col = 'title'
        summary_col = 'summary'
        category_col = 'category_code'
        
        print(f"\n  Using: Title='{title_col}', Abstract='{summary_col}'")
        print(f"         Category='{category_col}'")
        
        # Create training pairs
        train_examples = []
        category_groups = {}
        
        # Group papers by category
        for cat in df[category_col].unique():
            if pd.notna(cat):
                category_groups[str(cat)] = df[df[category_col] == cat]
        
        print(f"\n  Categories found: {len(category_groups)}")
        
        # Strategy 1: Title-Abstract pairs (highest similarity)
        print("\n  Creating training pairs...")
        print("    [1/4] Title-Abstract pairs (0.85 similarity)...")
        for idx, row in df.iterrows():
            if len(train_examples) >= CONFIG['max_examples']:
                break
            
            try:
                title = str(row[title_col]).strip()
                abstract = str(row[summary_col]).strip()
                
                # Clean up text
                if len(title) > 10 and len(abstract) > 50:
                    # Title-Abstract: high similarity (0.85)
                    train_examples.append(InputExample(
                        texts=[title, abstract],
                        label=0.85
                    ))
            except:
                continue
        
        count_before = len(train_examples)
        
        # Strategy 2: Abstract sentence pairs (medium similarity)
        print("    [2/4] Abstract sentence pairs (0.65 similarity)...")
        for idx, row in df.iterrows():
            if len(train_examples) >= CONFIG['max_examples']:
                break
            
            try:
                abstract = str(row[summary_col]).strip()
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
        
        print(f"       Added {len(train_examples) - count_before} examples")
        count_before = len(train_examples)
        
        # Strategy 3: Same category papers (semantic similarity)
        print("    [3/4] Same-category paper pairs (0.55 similarity)...")
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
                            label=0.55
                        ))
                except:
                    continue
        
        print(f"       Added {len(train_examples) - count_before} examples")
        count_before = len(train_examples)
        
        # Strategy 4: Synthetic negative pairs (dissimilar categories)
        print("    [4/4] Synthetic negative pairs (0.15 similarity)...")
        categories = list(category_groups.keys())
        for i in range(min(len(categories) - 1, 100)):
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
                            label=0.15
                        ))
            except:
                continue
        
        print(f"       Added {len(train_examples) - count_before} examples")
        
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
# MANUAL FINE-TUNING LOOP
# ============================================================================

def finetune_manual(model, train_examples: List[InputExample]):
    """Fine-tune using manual PyTorch training loop."""
    
    if not train_examples:
        print("❌ No training examples available")
        return None
    
    print("\n" + "=" * 80)
    print("MANUAL FINE-TUNING LOOP")
    print("=" * 80)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print(f"✓ Model moved to device: {device}")
        
        # Create data loader
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=CONFIG['batch_size']
        )
        print(f"✓ DataLoader: {len(train_examples)} examples, batch size {CONFIG['batch_size']}")
        print(f"  Batches per epoch: {len(train_dataloader)}")
        
        # Define loss function
        loss_model = CosineSimilarityLoss(model)
        print(f"✓ Loss: CosineSimilarityLoss")
        
        # Define optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
        print(f"✓ Optimizer: AdamW (lr={CONFIG['learning_rate']})")
        
        # Training loop
        print(f"\n" + "=" * 80)
        print(f"TRAINING ({CONFIG['epochs']} EPOCHS)")
        print("=" * 80)
        
        total_loss = 0
        total_steps = 0
        
        for epoch in range(CONFIG['epochs']):
            model.train()
            epoch_loss = 0
            batch_count = 0
            
            print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
            
            for batch_idx, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                
                try:
                    # Forward pass
                    loss_value = loss_model(batch, optimizer)
                    
                    # Backward pass
                    loss_value.backward()
                    optimizer.step()
                    
                    epoch_loss += loss_value.item()
                    batch_count += 1
                    total_steps += 1
                    
                    # Progress
                    if (batch_idx + 1) % max(1, len(train_dataloader) // 5) == 0:
                        avg_loss = epoch_loss / batch_count
                        print(f"  Batch {batch_idx + 1}/{len(train_dataloader)} - Loss: {avg_loss:.4f}")
                
                except Exception as e:
                    print(f"  ⚠ Batch error: {str(e)[:50]}")
                    continue
            
            epoch_avg_loss = epoch_loss / max(batch_count, 1)
            total_loss += epoch_avg_loss
            print(f"  Epoch {epoch + 1} Loss: {epoch_avg_loss:.4f}")
        
        print(f"\n✓ Fine-tuning completed")
        print(f"  Total steps: {total_steps}")
        print(f"  Average loss: {total_loss / CONFIG['epochs']:.4f}")
        
        return model
    
    except Exception as e:
        print(f"❌ Error in training loop: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# SAVE AND EVALUATE
# ============================================================================

def save_and_evaluate(model):
    """Save model and evaluate on test cases."""
    if model is None:
        return
    
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)
    
    try:
        model.save(CONFIG['output_dir'])
        print(f"✓ Model saved to: {CONFIG['output_dir']}")
        
        # Save metadata
        metadata = {
            'base_model': 'paraphrase-distilroberta-base-v1',
            'training_data': 'arXiv Computer Science Papers',
            'epochs': CONFIG['epochs'],
            'batch_size': CONFIG['batch_size'],
            'learning_rate': CONFIG['learning_rate'],
            'embedding_dimension': 768,
            'max_sequence_length': 128,
            'purpose': 'Domain-specific semantic search on research papers'
        }
        
        with open(f"{CONFIG['output_dir']}/arxiv_finetuning_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved")
    
    except Exception as e:
        print(f"❌ Save error: {str(e)}")
        return
    
    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATING DOMAIN-SPECIFIC MODEL")
    print("=" * 80)
    
    test_cases = [
        {
            "query": "deep learning neural networks transformer",
            "docs": [
                "Transformers use attention mechanisms to process sequential data effectively",
                "Deep neural networks with multiple layers learn hierarchical representations",
                "Natural language processing has been revolutionized by transformer models",
                "Computer graphics involves 3D rendering and visualization techniques",
            ],
        },
        {
            "query": "natural language processing text embedding",
            "docs": [
                "Word embeddings capture semantic relationships between words",
                "Sentence transformers create fixed-size vector representations for texts",
                "Text classification using neural networks achieves state-of-the-art results",
                "Image segmentation partitions images into semantic regions",
            ],
        },
        {
            "query": "machine learning optimization algorithm",
            "docs": [
                "Gradient descent optimization algorithms minimize loss functions",
                "Stochastic optimization methods provide faster convergence than batch methods",
                "Hyperparameter tuning affects model performance significantly",
                "Database indexing improves query execution speed",
            ],
        },
    ]
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    print("\nTest Evaluations:")
    print("-" * 80)
    
    all_scores = []
    
    for idx, test in enumerate(test_cases, 1):
        print(f"\n[Test {idx}] Query: '{test['query']}'")
        
        # Encode
        query_emb = model.encode(test['query'], convert_to_tensor=True)
        doc_embs = model.encode(test['docs'], convert_to_tensor=True)
        
        # Compute similarities
        sims = cosine_similarity([query_emb.cpu().numpy()], 
                                doc_embs.cpu().numpy())[0]
        
        # Rank
        ranked = np.argsort(sims)[::-1]
        
        print(f"  Results:")
        for rank, doc_idx in enumerate(ranked[:3], 1):
            marker = "→"
            print(f"    {marker} Rank {rank}: (sim: {sims[doc_idx]:.4f}) {test['docs'][doc_idx][:50]}...")
            all_scores.append(sims[doc_idx])
    
    print(f"\n{'=' * 80}")
    print(f"Average Similarity Score: {np.mean(all_scores):.4f}")
    print(f"{'=' * 80}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        print(f"\nPython: {sys.version}")
        print(f"PyTorch: {torch.__version__}")
        print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        
        # Load data
        train_examples = load_arxiv_data()
        
        if train_examples:
            # Load model
            print("\n" + "=" * 80)
            print("LOADING BASE MODEL")
            print("=" * 80)
            
            try:
                model = SentenceTransformer(CONFIG['model_name'])
                print(f"✓ Model loaded: {type(model).__name__}")
                print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
                print(f"  Max sequence length: {model.max_seq_length}")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                sys.exit(1)
            
            # Fine-tune
            finetuned_model = finetune_manual(model, train_examples)
            
            # Save and evaluate
            if finetuned_model:
                save_and_evaluate(finetuned_model)
                
                print("\n" + "=" * 80)
                print("✓ DOMAIN-SPECIFIC FINE-TUNING COMPLETE")
                print("=" * 80)
                
                print(f"\n✓ New checkpoint saved to: {CONFIG['output_dir']}")
                print(f"\n✓ arXiv-Specific Model Details:")
                print(f"  - Base model: paraphrase-distilroberta-base-v1")
                print(f"  - Training data: {len(train_examples)} research paper pairs")
                print(f"  - Training epochs: {CONFIG['epochs']}")
                print(f"  - Domain: Computer Science (arXiv papers)")
                print(f"  - Learning rate: {CONFIG['learning_rate']}")
                
                print(f"\n✓ Usage:")
                print(f"  from sentence_transformers import SentenceTransformer")
                print(f"  model = SentenceTransformer('{CONFIG['output_dir']}')")
                
                print(f"\n✓ To update your system:")
                print(f"  src/retrieval.py: model_name = '{CONFIG['output_dir']}'")
        else:
            print("\n❌ No training data available")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
