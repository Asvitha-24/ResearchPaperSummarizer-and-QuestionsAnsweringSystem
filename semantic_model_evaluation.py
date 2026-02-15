"""
Semantic Search Model Evaluation & Comparison
==============================================

This script evaluates and compares multiple sentence embedding models:
1. paraphrase-MiniLM-L6-v2
2. paraphrase-distilroberta-base-v1
3. distilbert-base-nli-stsb-mean-tokens
4. paraphrase-xlm-r-multilingual-v1

Evaluation Metrics:
- Cosine Similarity Scores
- Inference Speed (sentences/second)
- Model Size (parameters, memory)
- Semantic Textual Similarity (STS) Correlation
- Query-Document Retrieval Performance

Requirements:
    pip install sentence-transformers numpy pandas scikit-learn scipy
"""

import numpy as np
import pandas as pd
import time
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr
import os
import json
from collections import defaultdict

# Try importing sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("WARNING: sentence-transformers not installed. Installing now...")
    os.system("pip install -q sentence-transformers")
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True


# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_TO_EVALUATE = {
    'paraphrase-MiniLM-L6-v2': {
        'type': 'MiniLM',
        'language': 'English',
        'description': 'Fast and efficient, ideal for speed-critical applications'
    },
    'paraphrase-distilroberta-base-v1': {
        'type': 'Distilled RoBERTa',
        'language': 'English',
        'description': 'Good balance between speed and quality'
    },
    'distilbert-base-nli-stsb-mean-tokens': {
        'type': 'DistilBERT',
        'language': 'English',
        'description': 'General-purpose with solid performance'
    },
    'paraphrase-xlm-r-multilingual-v1': {
        'type': 'Multilingual XLM-R',
        'language': 'Multilingual (111+ languages)',
        'description': 'Best for multilingual applications'
    }
}

# Test datasets for semantic similarity evaluation
SEMANTIC_SIMILARITY_PAIRS = [
    # Exact paraphrases (high similarity expected: ~0.9-1.0)
    ("A dog is playing in the park", "A dog is playing in the park", 1.0),
    ("The quick brown fox jumps", "The quick brown fox jumps", 1.0),
    
    # Close paraphrases (high similarity: ~0.8-0.9)
    ("A dog is playing in the park", "A dog is playing outdoors", 0.85),
    ("The cat sat on the mat", "The feline rested on the rug", 0.80),
    ("I like eating apples", "I enjoy eating apples", 0.85),
    
    # Semantic similarity (moderate: ~0.6-0.8)
    ("The movie is great", "The film is excellent", 0.75),
    ("Cars are vehicles", "Automobiles are transportation", 0.70),
    ("She bought a house", "She purchased a home", 0.75),
    
    # Related but not similar (low: ~0.3-0.5)
    ("The weather is sunny", "I like to play soccer", 0.35),
    ("Coffee is hot", "Ice cream is cold", 0.40),
    
    # Opposite or unrelated (very low: ~0.0-0.2)
    ("I love this movie", "I hate everything", 0.10),
    ("The sky is blue", "Elephants are large", 0.15),
]

# Query-Document pairs for retrieval evaluation
RETRIEVAL_TEST_DATA = {
    "queries": [
        "What is machine learning?",
        "How to train a neural network?",
        "What are transformers in NLP?",
        "Best practices for data preprocessing?",
        "How to deploy a model?"
    ],
    "documents": [
        "Machine learning is a subset of artificial intelligence...",
        "Neural networks are computational models inspired by biological neurons...",
        "Transformers revolutionized NLP with attention mechanisms...",
        "Data preprocessing is crucial for model performance...",
        "Model deployment requires careful consideration of latency and throughput...",
        "Python is a popular programming language for data science...",
        "GPU acceleration can significantly speed up model training...",
        "Hyperparameter tuning is essential for optimal model performance...",
    ],
    # Relevant document indices for each query
    "relevant_docs": {
        0: [0],  # Query 0 is most relevant to document 0
        1: [1],  # Query 1 is most relevant to document 1
        2: [2],  # Query 2 is most relevant to document 2
        3: [3],  # Query 3 is most relevant to document 3
        4: [4],  # Query 4 is most relevant to document 4
    }
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

class SemanticModelEvaluator:
    """Evaluates semantic search models comprehensively."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.models = {}
        self.results = defaultdict(dict)
        self.load_times = {}
        
    def load_model(self, model_name: str) -> bool:
        """Load a sentence transformer model."""
        print(f"Loading {model_name}...", end=" ", flush=True)
        try:
            start_time = time.time()
            model = SentenceTransformer(model_name)
            load_duration = time.time() - start_time
            self.models[model_name] = model
            self.load_times[model_name] = load_duration
            print(f"✓ (Loaded in {load_duration:.2f}s)")
            return True
        except Exception as e:
            print(f"✗ Failed: {str(e)[:50]}")
            return False
    
    def load_all_models(self):
        """Load all models for evaluation."""
        print("=" * 80)
        print("LOADING SEMANTIC SEARCH MODELS")
        print("=" * 80)
        
        for model_name in MODELS_TO_EVALUATE.keys():
            self.load_model(model_name)
        
        print(f"\n✓ Successfully loaded {len(self.models)}/{len(MODELS_TO_EVALUATE)} models")
    
    def encode_texts(self, model_name: str, texts: List[str]) -> np.ndarray:
        """Encode texts using a model."""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings
    
    def evaluate_semantic_similarity(self):
        """Evaluate models on semantic similarity task."""
        print("\n" + "=" * 80)
        print("EVALUATION 1: SEMANTIC SIMILARITY")
        print("=" * 80)
        
        for model_name in self.models.keys():
            print(f"\n{model_name}:")
            print("-" * 80)
            
            similarities = []
            correlations = []
            
            # Extract texts and ground truth
            texts1 = [pair[0] for pair in SEMANTIC_SIMILARITY_PAIRS]
            texts2 = [pair[1] for pair in SEMANTIC_SIMILARITY_PAIRS]
            true_scores = [pair[2] for pair in SEMANTIC_SIMILARITY_PAIRS]
            
            # Encode texts
            embeddings1 = self.encode_texts(model_name, texts1)
            embeddings2 = self.encode_texts(model_name, texts2)
            
            # Compute cosine similarities
            for emb1, emb2 in zip(embeddings1, embeddings2):
                sim = cosine_similarity([emb1], [emb2])[0][0]
                similarities.append(sim)
            
            # Compute correlation with ground truth
            pearson_corr, pearson_p = pearsonr(similarities, true_scores)
            spearman_corr, spearman_p = spearmanr(similarities, true_scores)
            
            # Store results
            self.results[model_name]['pearson_correlation'] = pearson_corr
            self.results[model_name]['spearman_correlation'] = spearman_corr
            self.results[model_name]['similarity_scores'] = similarities
            
            print(f"  Pearson Correlation:  {pearson_corr:.4f} (p-value: {pearson_p:.2e})")
            print(f"  Spearman Correlation: {spearman_corr:.4f} (p-value: {spearman_p:.2e})")
            print(f"  Avg Similarity Score: {np.mean(similarities):.4f}")
            print(f"  Min/Max Similarity:   {np.min(similarities):.4f} / {np.max(similarities):.4f}")
    
    def evaluate_inference_speed(self, num_iterations: int = 100):
        """Evaluate inference speed of models."""
        print("\n" + "=" * 80)
        print(f"EVALUATION 2: INFERENCE SPEED ({num_iterations} iterations)")
        print("=" * 80)
        
        # Test sentences
        test_sentences = [
            "This is a sample sentence for speed testing.",
            "Machine learning models require efficient inference.",
            "Semantic search is crucial for modern applications.",
        ] * (num_iterations // 3)
        
        for model_name in self.models.keys():
            print(f"\n{model_name}:")
            
            # Warmup
            _ = self.encode_texts(model_name, test_sentences[:3])
            
            # Measure inference time
            start_time = time.time()
            _ = self.encode_texts(model_name, test_sentences)
            duration = time.time() - start_time
            
            sentences_per_second = len(test_sentences) / duration
            
            self.results[model_name]['inference_time_sec'] = duration
            self.results[model_name]['sentences_per_second'] = sentences_per_second
            
            print(f"  Total Time: {duration:.2f}s")
            print(f"  Speed: {sentences_per_second:.2f} sentences/second")
            print(f"  Avg per sentence: {(duration/len(test_sentences))*1000:.2f}ms")
    
    def evaluate_retrieval_performance(self, top_k: int = 1):
        """Evaluate retrieval performance (MRR, recall@k)."""
        print("\n" + "=" * 80)
        print(f"EVALUATION 3: RETRIEVAL PERFORMANCE (top-{top_k})")
        print("=" * 80)
        
        queries = RETRIEVAL_TEST_DATA["queries"]
        documents = RETRIEVAL_TEST_DATA["documents"]
        relevant_docs = RETRIEVAL_TEST_DATA["relevant_docs"]
        
        for model_name in self.models.keys():
            print(f"\n{model_name}:")
            
            # Encode queries and documents
            query_embeddings = self.encode_texts(model_name, queries)
            doc_embeddings = self.encode_texts(model_name, documents)
            
            # Compute similarities
            similarities = cosine_similarity(query_embeddings, doc_embeddings)
            
            # Compute metrics
            mrr_scores = []  # Mean Reciprocal Rank
            recall_at_k = []
            
            for query_idx, sim_scores in enumerate(similarities):
                # Get top-k documents
                top_doc_indices = np.argsort(sim_scores)[::-1][:top_k]
                
                # Check if relevant document is in top-k
                relevant = relevant_docs.get(query_idx, [])
                found_relevant = any(doc_idx in top_doc_indices for doc_idx in relevant)
                
                # MRR: reciprocal rank of first relevant document
                mrr = 0
                for rank, doc_idx in enumerate(top_doc_indices, 1):
                    if doc_idx in relevant:
                        mrr = 1.0 / rank
                        break
                
                mrr_scores.append(mrr)
                recall_at_k.append(1.0 if found_relevant else 0.0)
            
            avg_mrr = np.mean(mrr_scores)
            recall = np.mean(recall_at_k)
            
            self.results[model_name]['mrr'] = avg_mrr
            self.results[model_name]['recall_at_k'] = recall
            
            print(f"  Mean Reciprocal Rank (MRR): {avg_mrr:.4f}")
            print(f"  Recall@{top_k}: {recall:.4f}")
    
    def get_model_info(self):
        """Get model information (size, architecture)."""
        print("\n" + "=" * 80)
        print("MODEL INFORMATION")
        print("=" * 80)
        
        for model_name in MODELS_TO_EVALUATE.keys():
            info = MODELS_TO_EVALUATE[model_name]
            print(f"\n{model_name}:")
            print(f"  Type: {info['type']}")
            print(f"  Language: {info['language']}")
            print(f"  Description: {info['description']}")
            
            if model_name in self.models:
                model = self.models[model_name]
                # Try to get embedding dimension
                try:
                    test_embedding = model.encode(["test"])
                    embedding_dim = test_embedding.shape[1]
                    self.results[model_name]['embedding_dimension'] = embedding_dim
                    print(f"  Embedding Dimension: {embedding_dim}")
                except:
                    pass
    
    def print_summary(self):
        """Print comprehensive evaluation summary."""
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        # Create summary dataframe
        summary_data = []
        for model_name in self.results.keys():
            result = self.results[model_name]
            summary_data.append({
                'Model': model_name.split('-')[0][:20],  # Shortened name
                'Pearson Corr': f"{result.get('pearson_correlation', 0):.4f}",
                'Spearman Corr': f"{result.get('spearman_correlation', 0):.4f}",
                'MRR': f"{result.get('mrr', 0):.4f}",
                'Recall@1': f"{result.get('recall_at_k', 0):.4f}",
                'Speed (sent/s)': f"{result.get('sentences_per_second', 0):.2f}",
                'Load Time (s)': f"{self.load_times.get(model_name, 0):.2f}"
            })
        
        df_summary = pd.DataFrame(summary_data)
        print("\n" + df_summary.to_string(index=False))
    
    def get_recommendations(self):
        """Provide recommendations based on evaluation results."""
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        
        # Find best model for different criteria
        models_list = list(self.results.keys())
        
        if not models_list:
            print("No models evaluated yet.")
            return
        
        # Best for semantic similarity
        best_sim = max(models_list, 
                      key=lambda m: self.results[m].get('pearson_correlation', 0))
        print(f"\n✓ Best for Semantic Similarity: {best_sim}")
        print(f"  Pearson Correlation: {self.results[best_sim]['pearson_correlation']:.4f}")
        
        # Best for retrieval
        best_retrieval = max(models_list, 
                            key=lambda m: self.results[m].get('mrr', 0))
        print(f"\n✓ Best for Retrieval: {best_retrieval}")
        print(f"  MRR: {self.results[best_retrieval]['mrr']:.4f}")
        
        # Fastest
        best_speed = max(models_list, 
                        key=lambda m: self.results[m].get('sentences_per_second', 0))
        print(f"\n✓ Fastest Model: {best_speed}")
        print(f"  Speed: {self.results[best_speed]['sentences_per_second']:.2f} sent/sec")
        
        # Overall best (balanced)
        scores = {}
        for model_name in models_list:
            result = self.results[model_name]
            # Calculate weighted score (50% similarity, 30% retrieval, 20% speed)
            sim_score = result.get('pearson_correlation', 0)
            retrieval_score = result.get('mrr', 0)
            speed_score = result.get('sentences_per_second', 0) / 100.0  # Normalize
            
            overall = (sim_score * 0.5 + retrieval_score * 0.3 + speed_score * 0.2)
            scores[model_name] = overall
        
        best_overall = max(scores, key=scores.get)
        print(f"\n✓ Overall Best Choice: {best_overall}")
        print(f"  Balanced Score: {scores[best_overall]:.4f}")
        
        # Use case recommendations
        print("\n" + "-" * 80)
        print("USE CASE RECOMMENDATIONS:")
        print("-" * 80)
        
        print(f"\n1. For Speed Priority:")
        print(f"   Choose: {best_speed}")
        print(f"   Reason: {MODELS_TO_EVALUATE[best_speed]['description']}")
        
        print(f"\n2. For Quality Priority:")
        print(f"   Choose: {best_sim}")
        print(f"   Reason: {MODELS_TO_EVALUATE[best_sim]['description']}")
        
        print(f"\n3. For Balanced Performance:")
        print(f"   Choose: {best_overall}")
        
        # Language support note
        print(f"\n4. For Multilingual Support:")
        print(f"   Choose: paraphrase-xlm-r-multilingual-v1")
        print(f"   Reason: Supports 111+ languages for global applications")


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    """Main evaluation pipeline."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + "SEMANTIC SEARCH MODEL EVALUATION & COMPARISON".center(78) + "║")
    print("╚" + "=" * 78 + "╝\n")
    
    # Initialize evaluator
    evaluator = SemanticModelEvaluator()
    
    # Load models
    evaluator.load_all_models()
    
    if not evaluator.models:
        print("ERROR: No models loaded successfully!")
        return
    
    # Get model information
    evaluator.get_model_info()
    
    # Run evaluations
    evaluator.evaluate_semantic_similarity()
    evaluator.evaluate_inference_speed(num_iterations=100)
    evaluator.evaluate_retrieval_performance(top_k=1)
    
    # Print summary and recommendations
    evaluator.print_summary()
    evaluator.get_recommendations()
    
    print("\n" + "=" * 80)
    print("✓ Evaluation completed successfully!")
    print("=" * 80 + "\n")
    
    return evaluator


if __name__ == "__main__":
    evaluator = main()
