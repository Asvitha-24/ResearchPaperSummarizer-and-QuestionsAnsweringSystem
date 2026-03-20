"""
Verify and compare the arXiv-finetuned model with the pre-trained checkpoint
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 80)
print("COMPARING MODELS: Pre-trained vs Domain-Specific")
print("=" * 80)

# Load both models
print("\nLoading models...")
pretrained = SentenceTransformer('checkpoints/distilroberta_paraphrase_finetuned')
arxiv_model = SentenceTransformer('checkpoints/distilroberta_arxiv_finetuned')
print("✓ Both models loaded")

# Test data: research paper queries and documents
test_queries = [
    "transformer architecture with attention mechanisms",
    "deep learning neural network optimization",
    "semantic similarity in sentence embeddings",
    "knowledge distillation for model compression",
    "bert transformer training objective"
]

arxiv_documents = [
    "Transformers use self-attention mechanisms to process sequential data efficiently",
    "Deep neural networks with multiple layers learn hierarchical feature representations",
    "Sentence embeddings capture semantic relationships between texts in vector space",
    "Knowledge distillation transfers knowledge from large teacher models to smaller students",
    "BERT pretraining uses masked language modeling and next sentence prediction objectives",
    "GRU networks are gated recurrent units for sequence modeling",
    "CNNs use convolutional layers for image processing",
    "RNNs were the standard before transformers revolutionized NLP",
]

print("\n" + "=" * 80)
print("SEMANTIC SEARCH EVALUATION (Mean Reciprocal Rank)")
print("=" * 80)

# Compute MRR for both models
def compute_mrr(model, queries, documents):
    """Compute Mean Reciprocal Rank for queries over documents."""
    mrr_scores = []
    
    for query in queries:
        query_emb = model.encode(query, convert_to_tensor=False)
        doc_embs = model.encode(documents, convert_to_tensor=False)
        similarities = cosine_similarity([query_emb], doc_embs)[0]
        
        # Rank documents
        ranked = np.argsort(similarities)[::-1]
        
        # Find if any top-ranked doc is semantically matching
        # (based on manual inspection of the test data above)
        relevant_indices = {0, 1, 2, 3, 4}  # First 5 are relevant to our test queries
        
        # Check where first relevant doc appears
        for rank, doc_idx in enumerate(ranked, 1):
            if doc_idx in relevant_indices:
                mrr_scores.append(1.0 / rank)
                break
    
    return np.mean(mrr_scores) if mrr_scores else 0.0


mrr_pretrained = compute_mrr(pretrained, test_queries, arxiv_documents)
mrr_arxiv = compute_mrr(arxiv_model, test_queries, arxiv_documents)

print(f"\nPre-trained model MRR:        {mrr_pretrained:.4f}")
print(f"arXiv Domain-Specific MRR:    {mrr_arxiv:.4f}")
improvement = ((mrr_arxiv - mrr_pretrained) / mrr_pretrained * 100) if mrr_pretrained > 0 else 0
print(f"Improvement:                  {improvement:+.1f}%")

# Detailed comparison
print("\n" + "=" * 80)
print("DETAILED QUERY ANALYSIS")
print("=" * 80)

for i, query in enumerate(test_queries, 1):
    print(f"\n[Query {i}] '{query}'")
    
    # Get embeddings
    query_emb_pre = pretrained.encode(query, convert_to_tensor=False)
    query_emb_arxiv = arxiv_model.encode(query, convert_to_tensor=False)
    
    doc_embs_pre = pretrained.encode(arxiv_documents, convert_to_tensor=False)
    doc_embs_arxiv = arxiv_model.encode(arxiv_documents, convert_to_tensor=False)
    
    # Get similarities
    sims_pre = cosine_similarity([query_emb_pre], doc_embs_pre)[0]
    sims_arxiv = cosine_similarity([query_emb_arxiv], doc_embs_arxiv)[0]
    
    # Top 2 results
    top2_pre = np.argsort(sims_pre)[::-1][:2]
    top2_arxiv = np.argsort(sims_arxiv)[::-1][:2]
    
    print(f"\n  Pre-trained model:")
    for rank, doc_idx in enumerate(top2_pre, 1):
        print(f"    {rank}. (sim: {sims_pre[doc_idx]:.4f}) {arxiv_documents[doc_idx][:60]}...")
    
    print(f"\n  arXiv Domain-Specific:")
    for rank, doc_idx in enumerate(top2_arxiv, 1):
        print(f"    {rank}. (sim: {sims_arxiv[doc_idx]:.4f}) {arxiv_documents[doc_idx][:60]}...")

# Summary statistics
print("\n" + "=" * 80)
print("EMBEDDING STATISTICS")
print("=" * 80)

test_text = "transformer attention mechanism"

emb_pretrained = pretrained.encode(test_text, convert_to_tensor=False)
emb_arxiv = arxiv_model.encode(test_text, convert_to_tensor=False)

print(f"\nFor text: '{test_text}'")
print(f"\nPre-trained embedding:")
print(f"  Shape: {emb_pretrained.shape}")
print(f"  Mean: {np.mean(emb_pretrained):.6f}")
print(f"  Std: {np.std(emb_pretrained):.6f}")
print(f"  Min/Max: {np.min(emb_pretrained):.6f} / {np.max(emb_pretrained):.6f}")
print(f"  L2 norm: {np.linalg.norm(emb_pretrained):.6f}")

print(f"\narXiv Domain-Specific embedding:")
print(f"  Shape: {emb_arxiv.shape}")
print(f"  Mean: {np.mean(emb_arxiv):.6f}")
print(f"  Std: {np.std(emb_arxiv):.6f}")
print(f"  Min/Max: {np.min(emb_arxiv):.6f} / {np.max(emb_arxiv):.6f}")
print(f"  L2 norm: {np.linalg.norm(emb_arxiv):.6f}")

# Cosine similarity between the two embeddings
similarity = cosine_similarity([emb_pretrained], [emb_arxiv])[0][0]
print(f"\nCosine similarity between embeddings: {similarity:.4f}")
print(f"  (measures difference in learned representations)")

# Final summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
✓ Domain-specific fine-tuning complete!

Models Available:
  1. Pre-trained: checkpoints/distilroberta_paraphrase_finetuned/
     - General-purpose semantic search
     - Baseline performance

  2. arXiv-tuned: checkpoints/distilroberta_arxiv_finetuned/
     - Trained on 600 arXiv paper pairs
     - Optimized for research paper content
     - Better understanding of technical terminology

Evaluation:
  - Pre-trained MRR: {mrr_pretrained:.4f}
  - arXiv MRR: {mrr_arxiv:.4f}
  - Improvement: {improvement:+.1f}%

Next Steps:
  1. Update src/retrieval.py to use arXiv model:
     model_name = 'checkpoints/distilroberta_arxiv_finetuned'

  2. Test in your application
  
  3. Monitor performance on your specific queries

Recommendation:
  Use the {('arXiv-tuned' if improvement > 0 else 'pre-trained')} model based on your performance requirements
  and latency constraints. The arXiv model is optimized for research papers.
""")

print("=" * 80)
print("✓ Verification complete")
print("=" * 80)
