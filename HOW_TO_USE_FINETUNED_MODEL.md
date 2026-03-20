# How to Use Fine-tuned BART - Quick Start

## 🚀 Three Steps to Get Started

### Step 1: Train the Model (One-time, ~20-30 min)
```bash
python finetune_bart_simple.py --samples 1000 --epochs 2 --batch-size 2
```

### Step 2: Use the Trained Model
```python
from use_finetuned_bart import FinetuedBARTSummarizer

# Load the fine-tuned model
summarizer = FinetuedBARTSummarizer()

# Summarize a research paper
paper_text = "Your paper text here..."
summary = summarizer.summarize(paper_text)

print("Original:", paper_text[:100])
print("Summary:", summary)
```

### Step 3: Try Interactive Examples
```bash
python use_finetuned_bart.py
```
Then select from 6 different examples!

---

## 📝 Simple Code Examples

### Example 1: Summarize One Paper
```python
from use_finetuned_bart import FinetuedBARTSummarizer

summarizer = FinetuedBARTSummarizer()

text = """
This paper introduces a revolutionary approach to deep learning using 
transformer architectures with attention mechanisms for improved performance 
on NLP tasks including question answering, summarization, and machine translation.
"""

summary = summarizer.summarize(text)
print(summary)
```

### Example 2: Summarize Multiple Papers
```python
from use_finetuned_bart import FinetuedBARTSummarizer

summarizer = FinetuedBARTSummarizer()

papers = [
    "Paper 1 text here...",
    "Paper 2 text here...",
    "Paper 3 text here...",
]

summaries = summarizer.batch_summarize(papers)

for paper, summary in zip(papers, summaries):
    print(f"Paper: {paper[:50]}...")
    print(f"Summary: {summary}\n")
```

### Example 3: Load From CSV
```python
import pandas as pd
from use_finetuned_bart import FinetuedBARTSummarizer

summarizer = FinetuedBARTSummarizer()

# Load papers
df = pd.read_csv("papers.csv")

# Summarize and save
df_results = summarizer.summarize_dataframe(
    df,
    text_column='abstract',
    output_column='summary'
)

# Save results
df_results.to_csv("papers_with_summaries.csv")
df_results.to_json("papers_with_summaries.json")
```

---

## ⚡ Training vs Inference

**Training** (One-time):
```bash
# Train for 1-3 hours (or minutes on GPU)
python finetune_bart_simple.py --samples 5000 --epochs 2 --batch-size 4
```

**Inference** (Repeat as needed):
```python
# Use the trained model instantly
summary = summarizer.summarize(text)  # Takes ~2-3 seconds
```

---

## 🎯 When to Use This

✅ **Use this when:**
- You want to summarize research papers
- You need domain-specific summarization
- You have a GPU (10-100x faster)
- You want better quality than pre-trained models

❌ **Don't use this when:**
- You only need to summarize once or twice
- You don't care about quality
- You need real-time performance

---

## 📊 What You Get

- ✅ Fine-tuned BART model (406M parameters)
- ✅ Saved checkpoints for reuse
- ✅ Support for batch processing
- ✅ Easy DataFrame integration
- ✅ CSV/JSON output support

---

## 💾 File Locations After Training

```
checkpoints/
└── bart_finetuned_simple/
    ├── config.json
    ├── pytorch_model.bin      (← The actual model, ~1.6GB)
    ├── tokenizer.json
    └── trainer_state.json
```

Once trained, you can:
1. Reuse the model anytime
2. Share with others
3. Deploy to production
4. Fine-tune further if needed

---

## 🔧 Customize Summary Length

```python
# Short summaries (30-80 tokens)
summary = summarizer.summarize(
    text,
    max_length=80,
    min_length=30
)

# Medium summaries (50-150 tokens)
summary = summarizer.summarize(
    text,
    max_length=150,
    min_length=50
)

# Long summaries (100-250 tokens)
summary = summarizer.summarize(
    text,
    max_length=250,
    min_length=100
)
```

---

## 🚨 Common Issues & Solutions

**Q: "Model not found" error?**
A: Train first: `python finetune_bart_simple.py --samples 100 --epochs 1`

**Q: Out of Memory?**
A: Reduce batch size: `--batch-size 1` or use fewer samples

**Q: Training very slow?**
A: This is normal on CPU! Use GPU for 10-100x speedup

**Q: Model quality not good enough?**
A: Train on more samples or more epochs

---

## 📈 Performance

| Setup | Speed | Quality |
|-------|-------|---------|
| Pre-trained (no training) | Fast | Generic |
| Fine-tuned on 100 papers | Fast | Good |
| Fine-tuned on 1000 papers | Fast | Better |
| Fine-tuned on 10K papers | Fast | Excellent |

---

## 🎓 Ready to Start?

```bash
# Step 1: Train (one-time)
python finetune_bart_simple.py --samples 500 --epochs 1 --batch-size 2

# Step 2: Try examples
python use_finetuned_bart.py

# Step 3: Use in your code
# See example code above!
```

---

**That's it! You're ready to use fine-tuned BART for research paper summarization! 🚀**
