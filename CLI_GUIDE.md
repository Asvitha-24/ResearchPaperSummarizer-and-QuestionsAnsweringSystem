# Main CLI Entry Point

This is the **primary entry point** for the Research Paper Summarizer & QA System. Use this script to run all operations.

## Quick Start

```powershell
# Show all available commands
python main.py --help

# Show help for specific command
python main.py preprocess --help
python main.py qa --help
```

## Available Commands

### 1. `preprocess` - Clean & Tokenize Data

Cleans text, removes URLs/emails, tokenizes into sentences and words.

```powershell
python main.py preprocess `
    --input "data/raw/arXiv Scientific Research Papers Dataset.csv" `
    --column "summary" `
    --output "data/processed/papers_processed.csv"
```

**Output columns:**
- `cleaned_{column}` - Cleaned text
- `word_count` - Number of tokens
- `sentence_count` - Number of sentences

---

### 2. `index` - Build Retrieval Index

Creates embeddings/indexes for fast document retrieval.

```powershell
python main.py index `
    --input "data/processed/papers_processed.csv" `
    --column "cleaned_summary" `
    --type "semantic" `
    --name "papers_index"
```

**Retriever types:**
- `tfidf` - Fast, lightweight (sklearn)
- `semantic` - Better quality, slower (~400MB model)
- `hybrid` - Combines both approaches

**Output:** Saved to `checkpoints/{name}.pkl`

---

### 3. `summarize` - Generate Abstractive Summaries

Uses BART model to create abstractive summaries.

```powershell
python main.py summarize `
    --input "data/processed/papers_processed.csv" `
    --column "cleaned_summary" `
    --max-length 150 `
    --min-length 50 `
    --output "data/processed/papers_summarized.csv"
```

**Options:**
- `--max-length` - Maximum summary length (tokens)
- `--min-length` - Minimum summary length (tokens)

---

### 4. `qa` - Answer Questions on Documents

Retrieves relevant documents and extracts answers.

```powershell
python main.py qa `
    --input "data/processed/papers_processed.csv" `
    --column "cleaned_summary" `
    --index "checkpoints/papers_index.pkl" `
    --question "What is the main contribution?" `
    --top-k 5 `
    --output "results.json"
```

**Output JSON structure:**
```json
{
  "question": "What is the main contribution?",
  "answers": [
    {
      "answer": "...",
      "score": 0.95,
      "document_index": 0,
      "relevance_score": 0.87
    }
  ],
  "num_results": 1
}
```

---

### 5. `demo` - Run Full Workflow

Automatically runs: preprocess → index → demo QA.

```powershell
python main.py demo
```

Expects raw data at: `data/raw/arXiv Scientific Research Papers Dataset.csv`

---

## Complete Workflow Example

```powershell
# Step 1: Preprocess your data
python main.py preprocess `
    --input "data/raw/arXiv Scientific Research Papers Dataset.csv" `
    --output "data/processed/papers_processed.csv"

# Step 2: Build semantic search index
python main.py index `
    --input "data/processed/papers_processed.csv" `
    --type semantic `
    --name my_papers

# Step 3: Answer a question
python main.py qa `
    --input "data/processed/papers_processed.csv" `
    --index "checkpoints/my_papers.pkl" `
    --question "What machine learning techniques are used?" `
    --top-k 5 `
    --output "results.json"

# Step 4: Summarize the papers
python main.py summarize `
    --input "data/processed/papers_processed.csv" `
    --output "data/processed/papers_summarized.csv"
```

---

## Input/Output Files

### Expected Directory Structure

```
ResearchPaperSummarizer-and-QuestionsAnsweringSystem/
├── data/
│   ├── raw/                    # Place your CSV files here
│   │   └── arXiv Scientific Research Papers Dataset.csv
│   └── processed/              # Output from preprocess
│       └── papers_processed.csv
├── checkpoints/                # Saved indexes
│   └── papers_index.pkl
├── main.py                     # This CLI script
└── requirements.txt
```

### CSV Column Requirements

**Input CSV** must have at least one text column (default: `summary`)

**After preprocessing**, new columns are created:
- `cleaned_summary` - Cleaned text
- `word_count` - Token count
- `sentence_count` - Sentence count

---

## System Requirements

- **Python:** 3.10+
- **RAM:** 8GB+ (16GB+ recommended for semantic models)
- **GPU:** Optional but recommended (NVIDIA with CUDA)
- **Disk:** ~1GB for model caches + data

---

## Troubleshooting

### "Input file not found"
→ Check file path and ensure it exists

### "Column not found in CSV"
→ Use `pandas` to inspect: `pd.read_csv('file.csv').columns`

### Slow on CPU
→ Models auto-detect GPU. Ensure CUDA is installed for acceleration

### Out of memory
→ Reduce batch size or use TF-IDF instead of semantic retriever

### Models downloading slowly
→ First run downloads models (~1-2GB total), be patient

---

## Advanced Usage

### Custom Model Selection

Edit `src/model.py` to change HuggingFace models:

```python
# Summarization models
"facebook/bart-large-cnn"        # Default
"t5-base"                        # Faster alternative
"pegasus-arxiv"                  # For scientific papers

# QA models
"distilbert-base-uncased-distilled-squad"  # Default, fast
"deepset/roberta-base-squad2"              # Higher accuracy
"allenai/longformer-base-4096"            # Long documents
```

### Batch Processing

Process multiple questions:

```python
from src.retrieval import SemanticRetriever
import pandas as pd

df = pd.read_csv("data/processed/papers_processed.csv")
retriever = SemanticRetriever()
retriever.fit(df["cleaned_summary"].tolist())

questions = [
    "What is the main contribution?",
    "What datasets were used?",
    "What are the limitations?"
]

results = retriever.batch_retrieve(questions, top_k=5)
```

---

## Environment Variables

Optionally set to customize behavior:

```powershell
$env:TORCH_HOME = "C:\path\to\cache"          # Model cache location
$env:HF_HOME = "C:\path\to\huggingface"       # HuggingFace cache
```

---

## License & Citation

See `README.md` and `PROJECT_SUMMARY.md` for project details.
