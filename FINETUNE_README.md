# BART Fine-tuning Guide

Complete guide for fine-tuning the BART model on research paper summarization task.

## 📋 Overview

Three fine-tuning scripts are provided, suited for different use cases:

| Script | Purpose | Use Case | Time |
|--------|---------|----------|------|
| `finetune_bart_lite.py` | **Quick testing** | Experiment with small dataset | 5-10 min |
| `finetune_bart.py` | **Full training** | Production fine-tuning | 1-4 hours |
| `finetune_bart_inference.py` | **Inference only** | Use pre-trained/fine-tuned models | - |

## 🚀 Quick Start

### Option 1: Quick Test (Lightweight)

Perfect for verifying setup and trying out fine-tuning:

```bash
python finetune_bart_lite.py --samples 1000 --epochs 2 --batch-size 4
```

**Parameters:**
- `--samples`: Number of samples (default: 1000)
- `--epochs`: Training epochs (default: 2)
- `--batch-size`: Batch size (default: 4)
- `--output`: Output directory (default: `./checkpoints/bart_finetuned_lite`)

**Expected time:** 5-10 minutes on GPU

### Option 2: Full Training

Complete fine-tuning on the entire dataset:

```bash
python finetune_bart.py
```

Edit `finetune_config.py` to adjust parameters before running.

**Expected time:** 1-4 hours depending on dataset size and GPU

### Option 3: Use Fine-tuned Model

Load and use the fine-tuned model for inference:

```python
from finetune_bart_inference import FinetuedBartSummarizer

summarizer = FinetuedBartSummarizer(
    model_dir="./checkpoints/bart_finetuned"
)

summary = summarizer.summarize(
    "Your text here...",
    max_length=150
)
print(summary)
```

## ⚙️ Configuration

### Model Parameters

Edit `finetune_config.py`:

```python
MODEL_CONFIG = {
    'model_name': 'facebook/bart-large-cnn',      # Base model
    'max_input_length': 1024,                      # Input token limit
    'max_target_length': 256,                      # Summary token limit
}
```

### Training Parameters

```python
TRAINING_CONFIG = {
    'batch_size': 8,              # Reduce if OOM
    'learning_rate': 5e-5,        # Learning rate
    'num_epochs': 3,              # Training epochs
    'warmup_steps': 500,          # Warmup steps
}
```

### Data Parameters

```python
DATA_CONFIG = {
    'sample_size': 5000,          # Set to None for full dataset
    'train_ratio': 0.7,           # 70% training
    'val_ratio': 0.15,            # 15% validation
    'test_ratio': 0.15,           # 15% testing
}
```

## 🔧 System Requirements

### Minimum Specs
- **GPU:** 8GB VRAM (NVIDIA with CUDA)
- **RAM:** 16GB
- **Storage:** 50GB (model + dataset)

### Recommended Specs
- **GPU:** 16GB+ VRAM (A100, V100, RTX 3090)
- **RAM:** 32GB+
- **Storage:** 100GB+

### CPU-only Setup
Works but much slower. Use smaller batch sizes:

```bash
# Set in environment or modify finetune_bart.py
export CUDA_VISIBLE_DEVICES=""
python finetune_bart_lite.py --batch-size 1 --epochs 1
```

## 📊 Dataset Format

Currently using arXiv Papers Dataset with columns:
- `title`: Paper title
- `summary`: Abstract/summary
- `category_code`: Research category (optional)

### Custom Dataset

Modify the data loading function in the scripts:

```python
# In finetune_bart.py, modify load_and_prepare_data():
def load_and_prepare_data(self, csv_path, ...):
    df = pd.read_csv(csv_path)
    # Adjust column mapping
    data_samples.append({
        'text': your_input_text,      # What to summarize
        'summary': your_target_summary # Target summary
    })
```

## 📈 Expected Performance

### After Fine-tuning
- **Training Loss:** Decreases from ~4.0 to ~1.5-2.0
- **Validation Loss:** Decreases from ~4.0 to ~2.0-2.5
- **ROUGE-1 Score:** ~35-45 (depending on data quality)

### Comparison with Pre-trained
- Pre-trained: Generic summaries, slower
- Fine-tuned: Domain-specific, more accurate

## 🎯 Common Issues & Solutions

### Out of Memory (OOM)

```bash
# Reduce batch size
python finetune_bart_lite.py --batch-size 2

# Or modify config
TRAINING_CONFIG = {'batch_size': 2}
```

### Slow Training

- Use gradient accumulation (set in config)
- Use smaller dataset (increase `--samples` gradually)
- Use mixed precision (enabled by default)

### Model Not Loading

Ensure model directory structure:
```
checkpoints/bart_finetuned/
├── config.json
├── pytorch_model.bin
├── tokenizer.json
└── tokenizer_config.json
```

### CUDA Not Available

```python
# Force CPU
import torch
torch.cuda.is_available = lambda: False

# Or run with CPU
python finetune_bart_lite.py  # Will auto-detect
```

## 📝 Monitoring Training

### Real-time Logs

Check training logs in `./checkpoints/bart_finetuned/runs/`:

```bash
tensorboard --logdir ./checkpoints/bart_finetuned/runs/
```

### Training Metrics

The Trainer logs:
- **Training Loss**
- **Validation Loss**
- **Learning Rate Schedule**
- **Batch Processing Speed**

## 🔄 Fine-tuning Tips

### Best Practices

1. **Start small:** Begin with 1000 samples, then scale up
2. **Monitor validation loss:** Stop if it increases (overfitting)
3. **Adjust learning rate:** Lower for larger datasets (1e-5 to 1e-4)
4. **Use appropriate batch size:** 8-16 for most GPUs
5. **Save checkpoints:** Enabled by default (keep 3 best models)

### Hyperparameter Tuning

```python
# For better results, experiment with:
TRAINING_CONFIG = {
    'learning_rate': 1e-5,    # Try: 1e-5 to 1e-4
    'warmup_steps': 1000,     # Try: 500 to 2000
    'weight_decay': 0.01,     # Try: 0 to 0.1
}

GENERATION_CONFIG = {
    'num_beams': 6,           # Try: 4 to 8
    'length_penalty': 2.5,    # Try: 1.5 to 3.0
}
```

## 📦 Output Files

After training, your model directory contains:

```
checkpoints/bart_finetuned/
├── config.json                 # Model configuration
├── pytorch_model.bin           # Model weights
├── tokenizer.json              # Tokenizer vocab
├── trainer_state.json          # Training state
├── training_args.bin           # Training arguments
└── runs/                       # TensorBoard logs
```

## 🧪 Evaluation

### Automatic Metrics

The trainer computes:
- **Loss:** Cross-entropy loss
- **Perplexity:** exp(loss)
- **ROUGE:** (if enabled in config)

### Manual Evaluation

```python
from finetune_bart_inference import FinetuedBartSummarizer

summarizer = FinetuedBartSummarizer()

# Compare fine-tuned vs pre-trained
results = summarizer.compare_with_pretrained(
    text="Your research paper text...",
    max_length=150
)

print(f"Fine-tuned: {results['finetuned']}")
print(f"Pre-trained: {results['pretrained']}")
```

## 🚀 Next Steps

1. **Test quick version** → Run `finetune_bart_lite.py`
2. **Adjust config** → Modify `finetune_config.py`
3. **Full training** → Run `finetune_bart.py`
4. **Integration** → Use `finetune_bart_inference.py`
5. **Production** → Deploy fine-tuned model

## 📚 References

- [Hugging Face BART Documentation](https://huggingface.co/docs/transformers/model_doc/bart)
- [Sequence-to-Sequence Fine-tuning Guide](https://huggingface.co/docs/transformers/tasks/summarization)
- [BART Paper](https://arxiv.org/abs/1910.13461)

## 💬 Support

For issues:
1. Check Common Issues section above
2. Check model directory structure
3. Verify dataset format
4. Try with smaller sample size
5. Check GPU VRAM availability

---

**Happy fine-tuning! 🎓**
