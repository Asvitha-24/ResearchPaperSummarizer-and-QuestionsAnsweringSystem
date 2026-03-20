# DistilBERT Question Answering Fine-tuning Guide

## Overview

This guide walks you through fine-tuning **DistilBERT** for the **Question Answering (QA)** task. DistilBERT is a lighter, faster version of BERT that maintains 97% of BERT's performance while being 40% smaller and 60% faster.

## What You Get

Three ready-to-use scripts:

1. **`finetune_distilbert_qa_simple.py`** - Lightweight version for quick testing
2. **`finetune_distilbert_qa.py`** - Full version with HuggingFace Trainer
3. **`distilbert_qa_utils.py`** - Utilities for inference and evaluation
4. **`run_distilbert_qa.py`** - Quick start launcher

## Quick Start

### Option 1: Run Lightweight Version (Recommended for Testing)

```bash
python finetune_distilbert_qa_simple.py
```

This will:
- ✓ Generate synthetic QA pairs from your dataset
- ✓ Fine-tune DistilBERT for 2 epochs
- ✓ Save the model to `checkpoints/distilbert_qa_lightweight/`
- ✓ Run inference tests on sample QA pairs
- ✓ Show training/validation losses

**Time estimate**: 10-30 minutes (depending on GPU)

### Option 2: Run Full Version with Trainer

```bash
python finetune_distilbert_qa.py
```

This will:
- ✓ Load SQuAD dataset or generate synthetic QA pairs
- ✓ Use HuggingFace Trainer for advanced training
- ✓ Include learning rate scheduling and warmup
- ✓ Save best model automatically
- ✓ Support GPU/TPU acceleration

**Time estimate**: 30-60 minutes

### Option 3: Interactive Mode

Use your fine-tuned model in interactive QA mode:

```bash
python distilbert_qa_utils.py
```

Or use it programmatically:

```python
from distilbert_qa_utils import DistilBertQAModel

# Load model
model = DistilBertQAModel("checkpoints/distilbert_qa_lightweight")

# Ask a question
context = "Machine learning is a subset of artificial intelligence..."
question = "What is machine learning?"

predictions = model.predict(question, context, top_k=3)
for pred in predictions:
    print(f"Answer: {pred['answer']} (score: {pred['score']:.4f})")
```

## System Requirements

**Minimum:**
- RAM: 8GB
- Disk: 5GB (for model + dataset)
- GPU: Optional (but recommended)

**Recommended:**
- RAM: 16GB+
- GPU: NVIDIA GPU with 8GB+ VRAM
- Storage: 10GB+

## Model Architecture

DistilBERT for QA:
- **Base**: DistilBERT (6 transformer layers vs BERT's 12)
- **Task**: Question Answering (Span Extraction)
- **Output**: Start position and end position probabilities for answer spans
- **Size**: ~268MB
- **Speed**: 2x faster than BERT

## Configuration Options

Edit these in the scripts:

### `finetune_distilbert_qa_simple.py`

```python
BATCH_SIZE = 4              # Reduce for less memory, increase for faster training
LEARNING_RATE = 2e-5        # Typical for fine-tuning
EPOCHS = 2                  # More epochs = better results but longer training
NUM_SAMPLES = 500           # Number of QA pairs to train on
MAX_LENGTH = 384            # Max sequence length
```

### `finetune_distilbert_qa.py`

```python
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 3
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128            # For handling long documents
TRAIN_DATASET_SIZE = 5000
```

## Data Format

The scripts support two data formats:

### Format 1: Synthetic QA (Automatic)

Generates from your existing dataset automatically:
- Takes titles and summaries from your CSV
- Creates question-answer pairs
- No additional setup needed

### Format 2: SQuAD Format (Manual)

If you have data in SQuAD format:

```json
{
    "version": "1.1",
    "data": [
        {
            "paragraphs": [
                {
                    "context": "Machine learning is...",
                    "qas": [
                        {
                            "id": "1",
                            "question": "What is machine learning?",
                            "answers": [
                                {
                                    "text": "a subset of AI",
                                    "answer_start": 30
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    ]
}
```

Set `use_squad=True` in the script.

## Training Tips

### GPU Memory Issues?

Reduce batch size:
```python
BATCH_SIZE = 2  # or 1 for very limited memory
```

### Want Better Results?

- Increase `NUM_SAMPLES` to 10000+
- Increase `EPOCHS` to 3-5
- Use `BATCH_SIZE = 8` or higher
- Fine-tune longer (more epochs)

### Training Too Slow?

- Use smaller `NUM_SAMPLES`
- Reduce `MAX_LENGTH` to 256
- Increase `BATCH_SIZE` (if GPU allows)
- Ensure GPU is being used

## Model Output Locations

After fine-tuning:

- **Lightweight version**: `checkpoints/distilbert_qa_lightweight/`
- **Full version**: `checkpoints/distilbert_qa_finetuned/`

Each location contains:
- `config.json` - Model configuration
- `pytorch_model.bin` - Model weights
- `tokenizer.json` - Tokenizer files

## Using Your Fine-tuned Model

### Inference

```python
from distilbert_qa_utils import DistilBertQAModel

model = DistilBertQAModel.from_finetuned("checkpoints/distilbert_qa_lightweight")

# Single prediction
context = "Your text here..."
question = "Your question?"
answers = model.predict(question, context, top_k=3)

# Batch predictions
qa_pairs = [
    {'question': 'Q1?', 'context': 'Context 1...'},
    {'question': 'Q2?', 'context': 'Context 2...'},
]
results = model.batch_predict(qa_pairs)
```

### Evaluation

```python
from distilbert_qa_utils import evaluate_qa_model

test_data = [
    {
        'question': 'What is AI?',
        'context': 'AI is...',
        'answers': ['artificial intelligence', 'AI']
    }
]

metrics = evaluate_qa_model(model, test_data)
print(f"Exact Match: {metrics['exact_match']:.2%}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
```

## Evaluation Metrics

- **Exact Match (EM)**: Percentage of predictions matching exactly with reference
- **F1 Score**: Average overlap between predicted and reference answers

## Troubleshooting

### CUDA Out of Memory
```
Solution: Reduce BATCH_SIZE from 8 to 4 or 2
```

### Model not improving
```
Solutions:
- Increase training data (NUM_SAMPLES)
- More epochs (EPOCHS = 5)
- Lower learning rate (LEARNING_RATE = 1e-5)
- Check data quality
```

### Tokenization errors
```
Solution: Update transformers library
pip install --upgrade transformers
```

### Slow training
```
Solutions:
- Check if GPU is being used (should say "cuda" at start)
- Increase batch size (if memory allows)
- Reduce MAX_LENGTH to 256
```

## Performance Benchmarks

On typical hardware with 500 samples, 2 epochs:

| Hardware | Time | Memory |
|----------|------|--------|
| CPU | 30-60 min | 8GB |
| GPU (8GB) | 5-15 min | 6GB |
| GPU (16GB) | 3-10 min | ~8GB |

## Next Steps

1. **Start with lightweight version** for quick testing
2. **Increase dataset size** if results are poor
3. **Adjust hyperparameters** based on results
4. **Switch to full version** for production use
5. **Deploy model** using HuggingFace or your framework

## Resources

- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [HuggingFace Documentation](https://huggingface.co/docs/transformers/)
- [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
- [DistilBERT Models](https://huggingface.co/models?search=distilbert)

## Common Questions

**Q: Which version should I use?**
A: Start with `finetune_distilbert_qa_simple.py` for testing, then use the full version for production.

**Q: Can I use my own dataset?**
A: Yes! Place CSV in `data/raw/` or provide SQuAD JSON format.

**Q: How much data do I need?**
A: Minimum 100 pairs, recommended 1000+, ideal 10000+

**Q: Will it work on CPU?**
A: Yes, but very slowly. GPU is highly recommended.

**Q: Can I use other models?**
A: Yes! Change `MODEL_NAME = "roberta-base"` or any HuggingFace model.

**Q: How do I deploy the model?**
A: Use `model.push_to_hub()` to upload to HuggingFace Hub, then deploy using their inference API.

## Support

For issues:
1. Check the Troubleshooting section above
2. Verify all dependencies are installed: `pip install -r requirements.txt`
3. Check GPU compatibility with PyTorch
4. Review HuggingFace documentation
