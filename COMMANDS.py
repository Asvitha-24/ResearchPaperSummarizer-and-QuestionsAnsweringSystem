#!/usr/bin/env python
"""
Quick reference cheat sheet - copy these commands to get started fast
"""

# =============================================================================
#                    RESEARCH PAPER QA SYSTEM - CHEAT SHEET
# =============================================================================

# STEP 1: Show available commands
# python main.py --help

# STEP 2: Preprocess your CSV file
# python main.py preprocess \
#     --input "data/raw/arXiv Scientific Research Papers Dataset.csv" \
#     --column "summary" \
#     --output "data/processed/papers_processed.csv"

# STEP 3: Build retrieval index (choose one type)
# FAST - TF-IDF based:
# python main.py index \
#     --input "data/processed/papers_processed.csv" \
#     --column "cleaned_summary" \
#     --type tfidf \
#     --name papers_index

# BETTER QUALITY - Semantic (recommended):
# python main.py index \
#     --input "data/processed/papers_processed.csv" \
#     --column "cleaned_summary" \
#     --type semantic \
#     --name papers_index

# BEST - Hybrid approach:
# python main.py index \
#     --input "data/processed/papers_processed.csv" \
#     --column "cleaned_summary" \
#     --type hybrid \
#     --name papers_index

# STEP 4: Ask questions!
# python main.py qa \
#     --input "data/processed/papers_processed.csv" \
#     --column "cleaned_summary" \
#     --index "checkpoints/papers_index.pkl" \
#     --question "What is the main contribution of this paper?" \
#     --top-k 5

# STEP 5: (Optional) Summarize papers
# python main.py summarize \
#     --input "data/processed/papers_processed.csv" \
#     --column "cleaned_summary" \
#     --output "data/processed/papers_summarized.csv"

# =============================================================================
#                              QUICK DEMO
# =============================================================================

# Run everything automatically with one command:
# python main.py demo

# =============================================================================
#                          COMMON USE CASES
# =============================================================================

# 1. Ask a specific question and save results to JSON
# python main.py qa \
#     --input "data/processed/papers_processed.csv" \
#     --index "checkpoints/papers_index.pkl" \
#     --question "What datasets are used in this research?" \
#     --output "results.json"

# 2. Fast TF-IDF search (no model download):
# python main.py index \
#     --input "data/processed/papers_processed.csv" \
#     --type tfidf \
#     --name fast_index

# 3. Create shorter summaries (50-100 tokens instead of 50-150):
# python main.py summarize \
#     --input "data/processed/papers_processed.csv" \
#     --max-length 100 \
#     --min-length 50

# 4. Retrieve more documents (10 instead of default 3):
# python main.py qa \
#     --input "data/processed/papers_processed.csv" \
#     --index "checkpoints/papers_index.pkl" \
#     --question "Your question here" \
#     --top-k 10

# =============================================================================
#                       GET HELP ON ANY COMMAND
# =============================================================================

# python main.py --help                    # All commands
# python main.py preprocess --help         # Preprocess help
# python main.py index --help              # Index help
# python main.py summarize --help          # Summarize help
# python main.py qa --help                 # QA help
# python main.py demo --help               # Demo help

# =============================================================================
#                          FILE STRUCTURE
# =============================================================================
# 
# data/
#   raw/
#     arXiv Scientific Research Papers Dataset.csv  <- PUT YOUR DATA HERE
#   processed/
#     papers_processed.csv                          <- CREATED BY preprocess
#     papers_summarized.csv                         <- CREATED BY summarize
#
# checkpoints/
#   papers_index.pkl                                <- CREATED BY index
#
# results.json                                      <- CREATED BY qa --output

# =============================================================================
#                        TROUBLESHOOTING
# =============================================================================

# "File not found" error
#   → Check the path and column names are correct
#   → Use: pd.read_csv('file.csv').columns  to see available columns

# Slow processing on first run
#   → First run downloads ML models (~1-2 GB)
#   → Be patient, it's a one-time operation

# GPU not being used
#   → Models auto-detect GPU
#   → Make sure CUDA/PyTorch is properly installed

# Out of memory error
#   → Use TF-IDF retriever instead of semantic
#   → Or use HF hub to cache models elsewhere:
#   → Set HF_HOME environment variable

# =============================================================================
#                       EXAMPLE QUESTIONS TO TRY
# =============================================================================

# What is the main contribution of this paper?
# What machine learning models are used?
# What datasets are mentioned in this research?
# What are the limitations of this approach?
# How does this compare to previous work?
# What are the future directions?
# What metrics are used for evaluation?
# What preprocessing steps are applied?

# =============================================================================
#                           SYSTEM INFO
# =============================================================================
#
# Models used:
#   - Summarization: facebook/bart-large-cnn
#   - QA: distilbert-base-uncased-distilled-squad
#   - Semantic Search: all-MiniLM-L6-v2
#
# Requires:
#   - Python 3.10+
#   - ~8GB RAM (16GB+ recommended)
#   - GPU optional but faster
#
# All dependencies in requirements.txt

# =============================================================================
