"""
BART Fine-tuning Verification Script
Tests that everything is configured correctly.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys

print("=" * 80)
print("BART FINE-TUNING SETUP VERIFICATION")
print("=" * 80 + "\n")

# 1. Check PyTorch
print("1. PyTorch Setup:")
print(f"   ✅ PyTorch Version: {torch.__version__}")
print(f"   ✅ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    print(f"   ℹ️  Training on CPU (will be slower)")

# 2. Load BART Model
print("\n2. BART Model Loading:")
try:
    print("   Loading facebook/bart-large-cnn...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    print("   ✅ BART Model loaded successfully!")
    print(f"   ✅ Model parameters: {model.num_parameters():,}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# 3. Test Tokenization
print("\n3. Tokenization Test:")
test_text = "This is a test of BART fine-tuning for research paper summarization."
try:
    tokens = tokenizer(test_text, return_tensors="pt")
    print(f"   ✅ Text tokenized: {len(tokens['input_ids'][0])} tokens")
    print(f"   ✅ Sample text: {test_text}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# 4. Test Model Forward Pass
print("\n4. Model Forward Pass Test:")
try:
    with torch.no_grad():
        outputs = model(
            input_ids=tokens['input_ids'],
            attention_mask=tokens['attention_mask']
        )
    print(f"   ✅ Forward pass successful!")
    print(f"   ✅ Output shape: {outputs.logits.shape}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# 5. Test Generation
print("\n5. Generation Test:")
try:
    with torch.no_grad():
        summary_ids = model.generate(
            tokens['input_ids'],
            max_length=50,
            num_beams=4
        )
    summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]
    print(f"   ✅ Text generation successful!")
    print(f"   ✅ Generated summary: {summary}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# 6. Test Data Loading
print("\n6. Dataset Loading Test:")
try:
    import pandas as pd
    csv_path = "data/raw/arXiv Scientific Research Papers Dataset.csv"
    df = pd.read_csv(csv_path, nrows=10)
    print(f"   ✅ CSV loaded successfully!")
    print(f"   ✅ Sample shape: {df.shape}")
    print(f"   ✅ Columns: {list(df.columns)[:5]}")
except FileNotFoundError:
    print(f"   ⚠️  CSV not found at {csv_path} (synthetic data will be used)")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# 7. Test PyTorch DataLoader
print("\n7. DataLoader Test:")
try:
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create sample data
    sample_input_ids = torch.randint(0, 1000, (10, 128))
    sample_attention = torch.ones(10, 128)
    sample_labels = torch.randint(0, 1000, (10, 128))
    
    dataset = TensorDataset(sample_input_ids, sample_attention, sample_labels)
    dataloader = DataLoader(dataset, batch_size=2)
    
    for batch in dataloader:
        pass
    
    print(f"   ✅ DataLoader test successful!")
    print(f"   ✅ Batches created: {len(dataloader)}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# 8. Fine-tuning Scripts Check
print("\n8. Fine-tuning Scripts:")
import os
scripts = [
    'finetune_bart.py',
    'finetune_bart_lite.py',
    'finetune_bart_simple.py',
    'finetune_bart_inference.py',
    'finetune_config.py',
]

all_exist = True
for script in scripts:
    if os.path.exists(script):
        print(f"   ✅ {script}")
    else:
        print(f"   ❌ {script} - NOT FOUND")
        all_exist = False

if not all_exist:
    print("\n   ⚠️  Some scripts missing!")
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("✅ ALL CHECKS PASSED - READY FOR FINE-TUNING!")
print("=" * 80 + "\n")

print("Next Steps:")
print("1. Quick Test (fast, CPU-friendly):")
print("   python finetune_bart_simple.py --samples 50 --epochs 1 --batch-size 2")
print()
print("2. Full Training (when ready):")
print("   python finetune_bart.py")
print()
print("3. Use Fine-tuned Model:")
print("   python -c \"from finetune_bart_inference import FinetuedBartSummarizer\"")
print("   summarizer = FinetuedBartSummarizer()")
print()
print("Note: Training on CPU is slow. For faster training, use GPU or cloud platforms.")
print("=" * 80 + "\n")
