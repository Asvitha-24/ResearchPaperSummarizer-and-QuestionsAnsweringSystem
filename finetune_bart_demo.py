"""
BART Fine-tuning Demo - Shows actual usage without user interaction
Run this to see working examples of the fine-tuned model
"""

import sys
from pathlib import Path

# Check if model exists
model_dir = Path("./checkpoints/bart_finetuned_simple")

print(__doc__)
print("\n" + "=" * 80)

if not model_dir.exists():
    print("⚠️  NOTICE: Fine-tuned model not found!")
    print("=" * 80)
    print("\nTo use the fine-tuned model, you need to train it first:")
    print("\n  python finetune_bart_simple.py --samples 1000 --epochs 2 --batch-size 2")
    print("\nAfter training, run this demo again:")
    print("  python finetune_bart_demo.py")
    print("\n" + "=" * 80)
    sys.exit(0)

print("✅ Fine-tuned model found! Loading...\n")

try:
    from use_finetuned_bart import FinetuedBARTSummarizer
    import pandas as pd
except ImportError as e:
    print(f"Error importing: {e}")
    sys.exit(1)

# Load summarizer
print("Loading fine-tuned BART model...")
summarizer = FinetuedBARTSummarizer(model_dir=str(model_dir))

print("\n" + "=" * 80)
print("DEMO 1: Simple Summarization")
print("=" * 80 + "\n")

text1 = """
This paper presents a novel approach to deep learning for natural language processing.
Our model combines transformer architectures with attention mechanisms to achieve superior
performance on multiple benchmark datasets. We evaluate our approach on SQuAD, GLUE, and
SuperGLUE benchmarks. The results show significant improvements over previous methods,
achieving 96% accuracy on SQuAD 2.0 and 89% on GLUE. Our implementation is efficient and
can be fine-tuned with minimal computational resources. The source code is available on GitHub.
"""

print("Input Text:")
print("-" * 80)
print(text1.strip())
print("-" * 80)

summary1 = summarizer.summarize(text1)

print("\nGenerated Summary:")
print("-" * 80)
print(summary1)
print("-" * 80)

print("\n" + "=" * 80)
print("DEMO 2: Multiple Papers")
print("=" * 80 + "\n")

papers = [
    "This work addresses computer vision problems using convolutional neural networks with novel attention mechanisms.",
    "We propose a new technique for reinforcement learning in robotic control tasks.",
    "This paper introduces improvements to transformer models for better performance on NLP tasks.",
]

print(f"Processing {len(papers)} papers...\n")

summaries = summarizer.batch_summarize(papers)

for i, (paper, summary) in enumerate(zip(papers, summaries), 1):
    print(f"Paper {i}:")
    print(f"  Input:   {paper}")
    print(f"  Summary: {summary}\n")

print("=" * 80)
print("DEMO 3: Adjustable Summary Length")
print("=" * 80 + "\n")

text2 = """
Machine learning and artificial intelligence have revolutionized many industries.
Deep learning models, in particular, have shown remarkable success in computer vision,
natural language processing, and other domains. However, training these models requires
significant computational resources and large datasets. This paper proposes a new approach
to reduce computational requirements while maintaining performance. We achieve 95% of baseline
performance with 50% less computation. Our method can be applied to various deep learning tasks.
"""

print("Input Text:")
print("-" * 80)
print(text2.strip())
print("-" * 80)

print("\nSHORT Summary (30-80 tokens):")
short_summary = summarizer.summarize(text2, max_length=80, min_length=30)
print(short_summary)

print("\nMEDIUM Summary (50-150 tokens):")
medium_summary = summarizer.summarize(text2, max_length=150, min_length=50)
print(medium_summary)

print("\nLONG Summary (100-250 tokens):")
long_summary = summarizer.summarize(text2, max_length=250, min_length=100)
print(long_summary)

print("\n" + "=" * 80)
print("DEMO 4: DataFrame Processing")
print("=" * 80 + "\n")

# Create sample data
sample_data = {
    'title': [
        'BERT: Pre-training of Deep Bidirectional Transformers',
        'Attention is All You Need',
        'ImageNet Classification with Deep Convolutional Networks'
    ],
    'abstract': [
        'We introduce BERT, a method of pre-training language representations using masked language modeling.',
        'We propose a new simple network architecture based entirely on attention mechanisms, the Transformer.',
        'We train a large deep convolutional neural network to classify the 1.2 million high-resolution images in ImageNet.'
    ]
}

df = pd.DataFrame(sample_data)

print("Input DataFrame:")
print("-" * 80)
print(df.to_string(index=False))
print("-" * 80)

# Create input column
df['input'] = df['title'] + ': ' + df['abstract']

# Summarize
print("\nGenerating summaries...")
df_result = summarizer.summarize_dataframe(df, 'input', 'summary')

print("\nDataFrame with Summaries:")
print("-" * 80)
for idx, row in df_result.iterrows():
    print(f"\n{idx+1}. {row['title']}")
    print(f"   Abstract: {row['abstract'][:80]}...")
    print(f"   Summary:  {row['summary']}")

print("\n" + "=" * 80)
print("DEMO 5: Save Results")
print("=" * 80 + "\n")

# Save results
output_csv = "demo_results.csv"
output_json = "demo_results.json"

df_result[['title', 'summary']].to_csv(output_csv, index=False)
df_result[['title', 'summary']].to_json(output_json, orient='records', indent=2)

print(f"✅ Results saved to:")
print(f"   - {output_csv}")
print(f"   - {output_json}")

print("\n" + "=" * 80)
print("DEMO COMPLETE!")
print("=" * 80)

print("\n📚 Next Steps:")
print("   1. Integrate into your application")
print("   2. Process your own research papers")
print("   3. Fine-tune on more data for better results")
print("   4. Deploy to production")

print("\n🔗 Usage Examples:")
print("   - See: use_finetuned_bart.py")
print("   - Read: HOW_TO_USE_FINETUNED_MODEL.md")

print("\n" + "=" * 80 + "\n")
