"""
Complete Guide: Using Fine-tuned BART Model for Research Paper Summarization
Shows how to load, use, and evaluate the fine-tuned model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
import pandas as pd
from typing import List, Dict


class FinetuedBARTSummarizer:
    """
    Simple wrapper for using the fine-tuned BART model.
    """
    
    def __init__(self, model_dir: str = "./checkpoints/bart_finetuned_simple"):
        """
        Load the fine-tuned BART model.
        
        Args:
            model_dir: Path to directory containing fine-tuned model
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading fine-tuned BART model from: {model_dir}")
        print(f"Using device: {self.device}\n")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
            self.model.to(self.device)
            self.model.eval()
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print(f"   Make sure model is trained first by running:")
            print(f"   python finetune_bart_simple.py --samples 1000 --epochs 2")
            raise
    
    def summarize(self, 
                 text: str, 
                 max_length: int = 150,
                 min_length: int = 50,
                 num_beams: int = 4) -> str:
        """
        Generate summary for a single text.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length in tokens
            min_length: Minimum summary length in tokens
            num_beams: Number of beams for beam search
            
        Returns:
            Generated summary
        """
        if not text or len(text.strip()) < 20:
            return "Text too short to summarize."
        
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(
                text,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            )
            inputs = inputs.to(self.device)
            
            # Generate summary
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
            
            # Decode
            summary = self.tokenizer.batch_decode(
                summary_ids,
                skip_special_tokens=True
            )[0]
            
            return summary
        
        except Exception as e:
            print(f"Error during summarization: {e}")
            return text[:500]  # Fallback
    
    def batch_summarize(self, 
                       texts: List[str],
                       max_length: int = 150,
                       **kwargs) -> List[str]:
        """Summarize multiple texts."""
        summaries = []
        for i, text in enumerate(texts, 1):
            print(f"Summarizing {i}/{len(texts)}...", end="\r")
            summary = self.summarize(text, max_length=max_length, **kwargs)
            summaries.append(summary)
        print(" " * 50, end="\r")  # Clear line
        return summaries
    
    def summarize_dataframe(self, 
                           df: pd.DataFrame,
                           text_column: str,
                           output_column: str = "summary") -> pd.DataFrame:
        """
        Summarize all texts in a DataFrame column.
        
        Args:
            df: Input DataFrame
            text_column: Column name containing texts to summarize
            output_column: Column name for output summaries
            
        Returns:
            DataFrame with added summary column
        """
        df = df.copy()
        summaries = self.batch_summarize(df[text_column].tolist())
        df[output_column] = summaries
        return df


def example_1_simple_summarization():
    """Example 1: Simple summarization of a single text."""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Simple Summarization")
    print("=" * 80 + "\n")
    
    # Load model
    summarizer = FinetuedBARTSummarizer()
    
    # Sample research paper abstract
    paper_text = """
    Deep learning has revolutionized the field of artificial intelligence and machine learning.
    In this work, we present a novel transformer-based architecture that combines the strengths
    of BERT and GPT models. Our approach uses multi-head attention mechanisms and a hierarchical
    encoding scheme to improve performance on various downstream NLP tasks. We evaluate our model
    on standard benchmarks including SQuAD, GLUE, and SuperGLUE. Results show significant
    improvements over previous state-of-the-art methods, achieving 95.3% accuracy on SQuAD v2.0
    and 87.4% on GLUE average score. The model is efficient and can be fine-tuned on domain-specific
    datasets with minimal computational resources. Code and pre-trained weights are available at
    our GitHub repository.
    """
    
    print("Input Text:")
    print("-" * 80)
    print(paper_text.strip())
    print("-" * 80)
    
    # Generate summary
    summary = summarizer.summarize(paper_text)
    
    print("\nGenerated Summary:")
    print("-" * 80)
    print(summary)
    print("-" * 80)


def example_2_batch_summarization():
    """Example 2: Summarize multiple papers."""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Batch Summarization")
    print("=" * 80 + "\n")
    
    # Load model
    summarizer = FinetuedBARTSummarizer()
    
    # Multiple papers
    papers = [
        """
        This paper introduces a new approach to computer vision using convolutional neural networks.
        We propose a novel architecture that combines spatial and channel attention mechanisms.
        Our model achieves state-of-the-art results on ImageNet, COCO, and Pascal VOC datasets.
        """,
        """
        Natural language processing has been transformed by transformer models. This work presents
        improvements to the BERT architecture for better contextual understanding. We evaluate
        on multiple benchmarks and show consistent improvements.
        """,
        """
        Reinforcement learning applications in robotics have shown promising results. Our approach
        uses deep Q-networks with experience replay. We demonstrate effectiveness on multiple
        robotic tasks including manipulation and navigation.
        """
    ]
    
    print(f"Summarizing {len(papers)} papers...\n")
    
    # Batch summarize
    summaries = summarizer.batch_summarize(papers)
    
    # Display results
    for i, (paper, summary) in enumerate(zip(papers, summaries), 1):
        print(f"Paper {i}:")
        print(f"  Original: {paper.strip()[:100]}...")
        print(f"  Summary:  {summary}\n")


def example_3_dataframe_processing():
    """Example 3: Process DataFrame with research papers."""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Processing DataFrame")
    print("=" * 80 + "\n")
    
    # Load model
    summarizer = FinetuedBARTSummarizer()
    
    # Create sample DataFrame from arXiv data
    print("Loading sample papers from arXiv dataset...")
    
    try:
        df = pd.read_csv("data/raw/arXiv Scientific Research Papers Dataset.csv", nrows=5)
        
        print(f"Loaded {len(df)} papers\n")
        print("Original data:")
        print(df[['title', 'summary']].head().to_string())
        
        # Create input text column (combine title + summary)
        df['input_text'] = df['title'] + '. ' + df['summary'].fillna('')
        
        # Summarize
        print("\n\nGenerating summaries...")
        df_with_summaries = summarizer.summarize_dataframe(
            df,
            text_column='input_text',
            output_column='finetuned_summary'
        )
        
        # Display results
        print("\nResults with Fine-tuned Summaries:")
        print("-" * 80)
        for idx, row in df_with_summaries.head(3).iterrows():
            print(f"\nTitle: {row['title']}")
            print(f"Original Summary: {row['summary'][:150]}...")
            print(f"Fine-tuned Summary: {row['finetuned_summary']}")
        
    except FileNotFoundError:
        print("⚠️  arXiv dataset not found. Creating example with synthetic data...")
        
        data = {
            'title': [
                'Attention is All You Need',
                'BERT: Pre-training of Deep Bidirectional Transformers',
                'ImageNet Classification with Deep Convolutional Networks'
            ],
            'summary': [
                'We propose a new simple network architecture based on attention mechanisms.',
                'We introduce BERT, a method of pre-training language representations.',
                'We show that convolutional networks trained on a large dataset are effective.'
            ]
        }
        
        df = pd.DataFrame(data)
        print("Sample data created\n")
        
        df['input_text'] = df['title'] + '. ' + df['summary']
        
        print("Generating summaries...")
        df_with_summaries = summarizer.summarize_dataframe(df, 'input_text')
        
        print("\nResults:")
        for idx, row in df_with_summaries.iterrows():
            print(f"\nTitle: {row['title']}")
            print(f"Summary: {row['finetuned_summary']}")


def example_4_save_results():
    """Example 4: Save summarization results to file."""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Save Results")
    print("=" * 80 + "\n")
    
    # Load model
    summarizer = FinetuedBARTSummarizer()
    
    # Sample papers
    papers = {
        'paper_1': "This paper introduces novel techniques for machine learning...",
        'paper_2': "We present a new approach to deep learning architectures...",
        'paper_3': "This work focuses on natural language understanding and processing...",
    }
    
    # Summarize
    print("Processing papers...")
    results = []
    for paper_id, text in papers.items():
        summary = summarizer.summarize(text)
        results.append({
            'paper_id': paper_id,
            'original_text': text,
            'summary': summary,
            'summary_length': len(summary.split())
        })
    
    # Save to CSV
    output_file = "summarization_results.csv"
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    
    print(f"\n✅ Results saved to: {output_file}\n")
    print(df_results.to_string())
    
    # Save to JSON for detailed view
    import json
    json_file = "summarization_results.json"
    df_results.to_json(json_file, orient='records', indent=2)
    print(f"\n✅ Detailed results saved to: {json_file}")


def example_5_compare_pretrained():
    """Example 5: Compare fine-tuned vs pre-trained model."""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Fine-tuned vs Pre-trained Model")
    print("=" * 80 + "\n")
    
    from transformers import pipeline
    
    # Load fine-tuned model
    print("Loading fine-tuned model...")
    summarizer_finetuned = FinetuedBARTSummarizer()
    
    # Load pre-trained model
    print("Loading pre-trained model...")
    summarizer_pretrained = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Test text
    test_text = """
    Deep learning models have achieved remarkable success in various domains including
    computer vision, natural language processing, and reinforcement learning. The success
    of these models is largely attributed to their ability to learn hierarchical
    representations of data. In this paper, we propose a novel architecture that improves
    upon existing approaches by incorporating new attention mechanisms and regularization
    techniques. Our experimental results demonstrate that the proposed model achieves
    state-of-the-art performance on multiple benchmarks.
    """
    
    print("Test Text:")
    print("-" * 80)
    print(test_text.strip()[:150] + "...")
    print("-" * 80)
    
    # Fine-tuned summary
    print("\n1. Fine-tuned BART Summary:")
    finetuned_summary = summarizer_finetuned.summarize(test_text)
    print(finetuned_summary)
    
    # Pre-trained summary
    print("\n2. Pre-trained BART Summary:")
    try:
        pretrained_result = summarizer_pretrained(
            test_text,
            max_length=130,
            min_length=30,
            do_sample=False
        )
        pretrained_summary = pretrained_result[0]['summary_text']
        print(pretrained_summary)
    except Exception as e:
        print(f"Could not generate pre-trained summary: {e}")
    
    print("\n" + "-" * 80)
    print("Comparison complete!")


def main():
    """Run all examples."""
    
    print("\n" + "=" * 80)
    print("FINE-TUNED BART MODEL - USAGE EXAMPLES")
    print("=" * 80)
    
    print("\n📝 NOTE: Make sure the model is fine-tuned first!")
    print("   Run: python finetune_bart_simple.py --samples 1000 --epochs 2")
    
    print("\n" + "=" * 80)
    print("Select which example to run:")
    print("=" * 80)
    print("1. Simple Summarization")
    print("2. Batch Summarization")
    print("3. DataFrame Processing")
    print("4. Save Results")
    print("5. Compare Fine-tuned vs Pre-trained")
    print("6. Run All Examples")
    print("0. Exit")
    
    choice = input("\nEnter your choice (0-6): ").strip()
    
    if choice == "1":
        example_1_simple_summarization()
    elif choice == "2":
        example_2_batch_summarization()
    elif choice == "3":
        example_3_dataframe_processing()
    elif choice == "4":
        example_4_save_results()
    elif choice == "5":
        example_5_compare_pretrained()
    elif choice == "6":
        example_1_simple_summarization()
        example_2_batch_summarization()
        example_3_dataframe_processing()
        example_4_save_results()
    elif choice == "0":
        print("Exiting...")
    else:
        print("Invalid choice!")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
