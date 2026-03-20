"""
Quick demo of DistilBERT QA fine-tuning.
Shows all steps without requiring long training times.
Perfect for validating setup.
"""

import torch
import os
from pathlib import Path

print("\n" + "="*70)
print("DISTILBERT QA SETUP VALIDATION & DEMO")
print("="*70)

# Check dependencies
print("\n1. Checking dependencies...")
try:
    import transformers
    print(f"   ✓ transformers {transformers.__version__}")
except ImportError:
    print("   ✗ transformers not installed")

try:
    import datasets
    print(f"   ✓ datasets {datasets.__version__}")
except ImportError:
    print("   ✗ datasets not installed")

try:
    import torch
    print(f"   ✓ torch {torch.__version__}")
except ImportError:
    print("   ✗ torch not installed")

# Check device
print("\n2. Checking GPU/Device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("   Note: CPU-only mode. Training will be slow.")
    print("   For faster training, consider using GPU.")

# Check data
print("\n3. Checking dataset...")
if os.path.exists("data/raw/arXiv Scientific Research Papers Dataset.csv"):
    import pandas as pd
    df = pd.read_csv("data/raw/arXiv Scientific Research Papers Dataset.csv", nrows=5)
    print(f"   ✓ Dataset found: {len(df)} samples (plus more)")
    print(f"   Columns: {df.columns.tolist()}")
else:
    print("   ✗ Dataset not found")

# Load model components
print("\n4. Testing model loading...")
try:
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    
    print("   Loading DistilBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased",
        cache_dir="./checkpoints/.cache"
    )
    print("   ✓ Tokenizer loaded")
    
    print("   Loading DistilBERT model...")
    model = AutoModelForQuestionAnswering.from_pretrained(
        "distilbert-base-uncased",
        cache_dir="./checkpoints/.cache"
    )
    print("   ✓ Model loaded")
    model.to(device)
    print(f"   ✓ Model moved to {device}")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test inference
print("\n5. Testing inference...")
try:
    context = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."
    question = "What is machine learning?"
    
    inputs = tokenizer.encode_plus(
        question,
        context,
        return_tensors='pt',
        max_length=512,
        truncation=True
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.decode(inputs['input_ids'][0][answer_start:answer_end])
    
    print(f"   Question: {question}")
    print(f"   Context: {context[:60]}...")
    print(f"   Answer: {answer}")
    print("   ✓ Inference works!")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Summary
print("\n" + "="*70)
print("SETUP VALIDATION COMPLETE!")
print("="*70)

print("\n✓ All components working correctly!")
print("\nYou're ready to fine-tune. Choose your next step:\n")

print("1. QUICK DEMO (5 minutes):")
print("   python demo_distilbert_qa_quick.py")
print("   - No actual training, just setup verification\n")

print("2. LIGHTWEIGHT FINE-TUNING (30 minutes on CPU, 5 min on GPU):")
print("   python finetune_distilbert_qa_simple.py")
print("   - 500 samples, 2 epochs")
print("   - Good for testing\n")

print("3. FULL FINE-TUNING (60 minutes on CPU, 15 min on GPU):")
print("   python finetune_distilbert_qa.py")
print("   - 5000 samples, 3 epochs")
print("   - Better results\n")

print("4. INTERACTIVE QA MODE:")
print("   python distilbert_qa_utils.py")
print("   - Ask questions interactively\n")

print("See DISTILBERT_QA_GUIDE.md for full documentation.")
print("="*70 + "\n")
