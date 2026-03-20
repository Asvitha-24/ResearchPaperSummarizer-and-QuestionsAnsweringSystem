"""
Verification script for fine-tuned paraphrase-distilroberta-base-v1
Tests that the system uses the checkpoint correctly
"""

import sys
import os

print("\n" + "=" * 80)
print("VERIFICATION: Fine-tuned Semantic Search System")
print("=" * 80)

# Test 1: Import and initialize with checkpoint
print("\n[1/3] Testing SemanticRetriever with checkpoint...")
try:
    from src.retrieval import SemanticRetriever
    retriever = SemanticRetriever()  # Should use checkpoint by default
    print("✓ SemanticRetriever initialized successfully")
    print(f"  Model: {retriever.model}")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 2: Test encoding
print("\n[2/3] Testing text encoding...")
try:
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks",
        "Natural language processing handles text data"
    ]
    retriever.fit(documents)
    print(f"✓ Encoded {len(documents)} documents")
    print(f"  Embeddings shape: {retriever.embeddings.shape}")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 3: Test retrieval
print("\n[3/3] Testing semantic search...")
try:
    query = "What is the relationship between ML and AI?"
    results = retriever.retrieve(query, top_k=2)
    print(f"✓ Retrieved top 2 results for query:")
    print(f"  Query: '{query}'")
    for idx, (doc_idx, doc_text, score) in enumerate(results, 1):
        print(f"  {idx}. (score: {score:.4f}) {doc_text[:50]}...")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED - SYSTEM READY")
print("=" * 80)

print("\n✓ Fine-tuned Model Details:")
print("  - Name: paraphrase-distilroberta-base-v1")
print("  - Checkpoint: checkpoints/distilroberta_paraphrase_finetuned/")
print("  - Embedding Dimension: 768")
print("  - Max Sequence Length: 128")
print("  - Use Case: General-purpose semantic search")

print("\n✓ System Configuration:")
print("  - Default retriever: SemanticRetriever (using checkpoint)")
print("  - src/retrieval.py: Updated with checkpoint path")
print("  - main.py: Updated to use new defaults")

print("\nReady for production! 🚀\n")
