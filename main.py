#!/usr/bin/env python
"""
Research Paper Summarizer & QA System - CLI Entry Point
Main command-line interface for the research paper QA system.
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import pickle
import json

from src.preprocess import DataPreprocessor, DataSplitter
from src.retrieval import SemanticRetriever, TFIDFRetriever, HybridRetriever
from src.model import ResearchPaperQASystem, SummarizationModel, QuestionAnsweringModel


# Constants
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
CHECKPOINT_PATH = "checkpoints"


def ensure_dirs():
    """Create necessary directories if they don't exist."""
    for dir_path in [RAW_DATA_PATH, PROCESSED_DATA_PATH, CHECKPOINT_PATH]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def preprocess_command(args):
    """Handle preprocessing command."""
    ensure_dirs()
    
    csv_file = args.input
    text_column = args.column
    output_file = args.output
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: Input file '{csv_file}' not found.")
        sys.exit(1)
    
    print(f"📖 Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    if text_column not in df.columns:
        print(f"❌ Error: Column '{text_column}' not found in CSV.")
        print(f"   Available columns: {', '.join(df.columns)}")
        sys.exit(1)
    
    print(f"🔄 Preprocessing {len(df)} records...")
    preprocessor = DataPreprocessor(remove_stopwords=False)
    df_processed = preprocessor.preprocess_dataframe(df, text_column=text_column)
    
    print(f"💾 Saving processed data to {output_file}...")
    df_processed.to_csv(output_file, index=False)
    
    print(f"✅ Preprocessing complete!")
    print(f"   - Records: {len(df_processed)}")
    print(f"   - Avg word count: {df_processed['word_count'].mean():.1f}")
    print(f"   - Avg sentence count: {df_processed['sentence_count'].mean():.1f}")


def index_command(args):
    """Handle indexing/retrieval setup command."""
    ensure_dirs()
    
    csv_file = args.input
    text_column = args.column
    retriever_type = args.type
    index_name = args.name
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: Input file '{csv_file}' not found.")
        sys.exit(1)
    
    print(f"📖 Loading processed data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    if text_column not in df.columns:
        print(f"❌ Error: Column '{text_column}' not found.")
        print(f"   Available columns: {', '.join(df.columns)}")
        sys.exit(1)
    
    documents = df[text_column].astype(str).tolist()
    print(f"📚 Loaded {len(documents)} documents.")
    
    print(f"🔍 Building {retriever_type} retriever index...")
    
    if retriever_type == "tfidf":
        retriever = TFIDFRetriever()
    elif retriever_type == "semantic":
        retriever = SemanticRetriever(model_name="all-MiniLM-L6-v2")
    elif retriever_type == "hybrid":
        retriever = HybridRetriever()
    else:
        print(f"❌ Unknown retriever type: {retriever_type}")
        sys.exit(1)
    
    retriever.fit(documents)
    
    index_path = os.path.join(CHECKPOINT_PATH, f"{index_name}.pkl")
    print(f"💾 Saving index to {index_path}...")
    with open(index_path, 'wb') as f:
        pickle.dump(retriever, f)
    
    print(f"✅ Index created successfully!")
    print(f"   - Retriever type: {retriever_type}")
    print(f"   - Documents indexed: {len(documents)}")


def summarize_command(args):
    """Handle summarization command."""
    ensure_dirs()
    
    csv_file = args.input
    text_column = args.column
    max_length = args.max_length
    min_length = args.min_length
    output_file = args.output
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: Input file '{csv_file}' not found.")
        sys.exit(1)
    
    print(f"📖 Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    if text_column not in df.columns:
        print(f"❌ Error: Column '{text_column}' not found.")
        sys.exit(1)
    
    print(f"🤖 Initializing summarization model...")
    summarizer = SummarizationModel(model_name="facebook/bart-large-cnn")
    
    texts = df[text_column].astype(str).tolist()
    print(f"📝 Summarizing {len(texts)} documents...")
    
    summaries = summarizer.batch_summarize(texts, max_length=max_length, min_length=min_length)
    
    df['summary'] = summaries
    df.to_csv(output_file, index=False)
    
    print(f"✅ Summarization complete!")
    print(f"   - Documents processed: {len(summaries)}")
    print(f"   - Output saved to: {output_file}")


def qa_command(args):
    """Handle Q&A command."""
    ensure_dirs()
    
    csv_file = args.input
    text_column = args.column
    index_file = args.index
    top_k = args.top_k
    question = args.question
    output_file = args.output
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: Input file '{csv_file}' not found.")
        sys.exit(1)
    
    if not os.path.exists(index_file):
        print(f"❌ Error: Index file '{index_file}' not found.")
        print(f"   Please run: python main.py index --input <csv> --name <index_name>")
        sys.exit(1)
    
    print(f"📖 Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print(f"🔍 Loading retriever index from {index_file}...")
    with open(index_file, 'rb') as f:
        retriever = pickle.load(f)
    
    print(f"🤖 Initializing QA system...")
    qa_model = QuestionAnsweringModel()
    
    documents = df[text_column].astype(str).tolist()
    print(f"📚 Loaded {len(documents)} documents.")
    
    # Retrieve relevant documents
    print(f"🔎 Searching for relevant documents...")
    relevant_docs = retriever.retrieve(question, top_k=top_k)
    
    answers = []
    for doc_idx, doc_text, relevance_score in relevant_docs:
        answer = qa_model.answer_question(question, doc_text)
        answer['document_index'] = doc_idx
        answer['relevance_score'] = float(relevance_score)
        answers.append(answer)
        print(f"\n  📄 Document {doc_idx} (relevance: {relevance_score:.3f})")
        print(f"     Answer: {answer['answer']}")
        print(f"     Confidence: {answer['score']:.3f}")
    
    result = {
        'question': question,
        'answers': answers,
        'num_results': len(answers)
    }
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved to {output_file}")
    
    print(f"\n✅ Q&A complete! Found {len(answers)} answer(s).")


def demo_command(args):
    """Handle demo command with sample workflow."""
    ensure_dirs()
    
    print("🚀 Running demo workflow...\n")
    
    # Step 1: Check if we have data
    raw_file = "data/raw/arXiv Scientific Research Papers Dataset.csv"
    if not os.path.exists(raw_file):
        print(f"⚠️  No data file found at {raw_file}")
        print("   Please place your CSV dataset in data/raw/ directory")
        return
    
    # Step 2: Preprocess
    print("=" * 50)
    print("Step 1: Preprocessing data...")
    print("=" * 50)
    processed_file = "data/processed/papers_processed.csv"
    if not os.path.exists(processed_file):
        args_preprocess = argparse.Namespace(
            input=raw_file,
            column='summary',
            output=processed_file
        )
        preprocess_command(args_preprocess)
    else:
        print(f"✓ Already preprocessed: {processed_file}")
    
    # Step 3: Build index
    print("\n" + "=" * 50)
    print("Step 2: Building retrieval index...")
    print("=" * 50)
    index_file = os.path.join(CHECKPOINT_PATH, "papers_index.pkl")
    if not os.path.exists(index_file):
        args_index = argparse.Namespace(
            input=processed_file,
            column='cleaned_summary',
            type='semantic',
            name='papers_index'
        )
        index_command(args_index)
    else:
        print(f"✓ Index already exists: {index_file}")
    
    # Step 4: Run Q&A
    print("\n" + "=" * 50)
    print("Step 3: Running Q&A demonstration...")
    print("=" * 50)
    
    sample_questions = [
        "What is the main contribution of this paper?",
        "What methods are used in this research?",
    ]
    
    for i, q in enumerate(sample_questions, 1):
        print(f"\nQuestion {i}: {q}")
        args_qa = argparse.Namespace(
            input=processed_file,
            column='cleaned_summary',
            index=index_file,
            top_k=3,
            question=q,
            output=None
        )
        qa_command(args_qa)
    
    print("\n✅ Demo complete!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Research Paper Summarizer & QA System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess raw CSV data
  python main.py preprocess --input data/raw/papers.csv

  # Build retrieval index
  python main.py index --input data/processed/papers_processed.csv --type semantic

  # Summarize documents
  python main.py summarize --input data/processed/papers_processed.csv

  # Answer questions
  python main.py qa --input data/processed/papers_processed.csv \\
                    --index checkpoints/papers_index.pkl \\
                    --question "What is the main contribution?"

  # Run full demo
  python main.py demo
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess raw data')
    preprocess_parser.add_argument('--input', default='data/raw/arXiv Scientific Research Papers Dataset.csv',
                                   help='Input CSV file path')
    preprocess_parser.add_argument('--column', default='summary', help='Column name to preprocess')
    preprocess_parser.add_argument('--output', default='data/processed/papers_processed.csv',
                                   help='Output CSV file path')
    preprocess_parser.set_defaults(func=preprocess_command)
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Build retrieval index')
    index_parser.add_argument('--input', default='data/processed/papers_processed.csv',
                              help='Processed CSV file path')
    index_parser.add_argument('--column', default='cleaned_summary',
                              help='Column name containing text')
    index_parser.add_argument('--type', choices=['tfidf', 'semantic', 'hybrid'], default='semantic',
                              help='Retriever type')
    index_parser.add_argument('--name', default='papers_index',
                              help='Name for the saved index')
    index_parser.set_defaults(func=index_command)
    
    # Summarize command
    summarize_parser = subparsers.add_parser('summarize', help='Summarize documents')
    summarize_parser.add_argument('--input', default='data/processed/papers_processed.csv',
                                  help='Input CSV file path')
    summarize_parser.add_argument('--column', default='cleaned_summary',
                                  help='Column name to summarize')
    summarize_parser.add_argument('--max-length', type=int, default=150,
                                  help='Maximum summary length in tokens')
    summarize_parser.add_argument('--min-length', type=int, default=50,
                                  help='Minimum summary length in tokens')
    summarize_parser.add_argument('--output', default='data/processed/papers_summarized.csv',
                                  help='Output CSV file path')
    summarize_parser.set_defaults(func=summarize_command)
    
    # QA command
    qa_parser = subparsers.add_parser('qa', help='Answer questions on indexed documents')
    qa_parser.add_argument('--input', default='data/processed/papers_processed.csv',
                           help='Processed CSV file path')
    qa_parser.add_argument('--column', default='cleaned_summary',
                           help='Column name containing text')
    qa_parser.add_argument('--index', default='checkpoints/papers_index.pkl',
                           help='Path to saved retriever index')
    qa_parser.add_argument('--question', required=True, help='Question to answer')
    qa_parser.add_argument('--top-k', type=int, default=3,
                           help='Number of documents to retrieve')
    qa_parser.add_argument('--output', help='Optional output JSON file')
    qa_parser.set_defaults(func=qa_command)
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run full demo workflow')
    demo_parser.set_defaults(func=demo_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
