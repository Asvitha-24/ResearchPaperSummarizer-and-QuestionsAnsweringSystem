# Quick Start Guide

## Installation

```bash
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Commands

### 1. Preprocess Data
Cleans and tokenizes your raw CSV data.

```bash
python main.py preprocess --input data/raw/your_file.csv
```

**Options:**
- `--input`: Path to raw CSV file (default: `data/raw/arXiv Scientific Research Papers Dataset.csv`)
- `--column`: Column name to preprocess (default: `summary`)
- `--output`: Output file path (default: `data/processed/papers_processed.csv`)

### 2. Build Retrieval Index
Creates an index for fast document retrieval.

```bash
python main.py index --input data/processed/papers_processed.csv --type semantic
```

**Options:**
- `--input`: Path to processed CSV file
- `--column`: Text column name (default: `cleaned_summary`)
- `--type`: Retriever type - `tfidf`, `semantic`, or `hybrid` (default: `semantic`)
- `--name`: Name for saved index (default: `papers_index`)

### 3. Summarize Documents
Generates abstractive summaries using BART model.

```bash
python main.py summarize --input data/processed/papers_processed.csv
```

**Options:**
- `--input`: Path to processed CSV file
- `--column`: Column to summarize (default: `cleaned_summary`)
- `--max-length`: Max summary length in tokens (default: `150`)
- `--min-length`: Min summary length in tokens (default: `50`)
- `--output`: Output file path

### 4. Answer Questions
Retrieves relevant documents and answers questions.

```bash
python main.py qa --input data/processed/papers_processed.csv \
                  --index checkpoints/papers_index.pkl \
                  --question "What is the main contribution?"
```

**Options:**
- `--input`: Path to processed CSV
- `--column`: Text column name (default: `cleaned_summary`)
- `--index`: Path to saved retriever index
- `--question`: Question to answer (required)
- `--top-k`: Number of documents to retrieve (default: `3`)
- `--output`: Optional JSON output file

### 5. Run Full Demo
Runs the complete workflow with sample data.

```bash
python main.py demo
```

## Quick Example Workflow

```bash
# 1. Preprocess your data
python main.py preprocess --input data/raw/arXiv Scientific Research Papers Dataset.csv

# 2. Build retrieval index
python main.py index --input data/processed/papers_processed.csv --type semantic --name my_papers

# 3. Ask a question
python main.py qa --input data/processed/papers_processed.csv \
                  --index checkpoints/my_papers.pkl \
                  --question "What are the main findings?" \
                  --top-k 5
```

## Output Structure

```
data/
├── raw/                    # Input CSV files
└── processed/             # Preprocessed data with columns:
                           # - cleaned_{original_column}
                           # - word_count
                           # - sentence_count
checkpoints/
└── {index_name}.pkl       # Saved retriever indexes
```

## System Features

✅ **Preprocessing**: Text cleaning, tokenization, sentence/word counts  
✅ **Retrieval**: TF-IDF, semantic, or hybrid document retrieval  
✅ **Summarization**: Abstractive summaries using BART  
✅ **Question Answering**: Context-based QA using DistilBERT  
✅ **Flexible**: Works with any CSV with text columns

## Tips

- Use `--help` flag for any command to see detailed options:
  ```bash
  python main.py qa --help
  ```
- First run of semantic retriever will download and cache embeddings model (~400MB)
- GPU recommended for faster processing (auto-detected if available)
- Results can be saved to JSON with `--output` flag in QA command
