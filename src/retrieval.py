"""
Information retrieval module for QA system.
Implements TF-IDF and semantic search-based document retrieval.
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import torch


class TFIDFRetriever:
    """TF-IDF based document retriever."""
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize TF-IDF retriever.
        
        Args:
            max_features: Maximum number of features
            ngram_range: N-gram range for vectorizer
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        self.documents = None
        self.tfidf_matrix = None
    
    def fit(self, documents: List[str]) -> None:
        """
        Fit TF-IDF vectorizer on documents.
        
        Args:
            documents: List of documents
        """
        self.documents = documents
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, str, float]]:
        """
        Retrieve top-k most relevant documents.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of tuples (doc_index, doc_text, relevance_score)
        """
        if self.tfidf_matrix is None:
            raise ValueError("Retriever not fitted. Call fit() first.")
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((
                int(idx),
                self.documents[idx],
                float(similarities[idx])
            ))
        
        return results
    
    def batch_retrieve(self, queries: List[str], top_k: int = 5) -> List[List[Tuple[int, str, float]]]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of queries
            top_k: Number of documents per query
            
        Returns:
            List of retrieval results
        """
        results = []
        for query in queries:
            results.append(self.retrieve(query, top_k))
        return results


class SemanticRetriever:
    """Semantic search based document retriever using embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        """
        Initialize semantic retriever.
        
        Args:
            model_name: SentenceTransformer model name
            batch_size: Batch size for encoding
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.batch_size = batch_size
        self.documents = None
        self.embeddings = None
    
    def fit(self, documents: List[str]) -> None:
        """
        Encode documents.
        
        Args:
            documents: List of documents
        """
        self.documents = documents
        print(f"Encoding {len(documents)} documents...")
        self.embeddings = self.model.encode(
            documents,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            show_progress_bar=True,
            device=self.device
        )
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, str, float]]:
        """
        Retrieve top-k most relevant documents.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of tuples (doc_index, doc_text, relevance_score)
        """
        if self.embeddings is None:
            raise ValueError("Retriever not fitted. Call fit() first.")
        
        # Encode query
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Calculate similarities
        similarities = util.pytorch_cos_sim(query_embedding, self.embeddings)[0]
        
        # Get top-k results
        top_results = torch.topk(similarities, k=min(top_k, len(self.documents)))
        
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            results.append((
                int(idx),
                self.documents[idx],
                float(score)
            ))
        
        return results
    
    def batch_retrieve(self, queries: List[str], top_k: int = 5) -> List[List[Tuple[int, str, float]]]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of queries
            top_k: Number of documents per query
            
        Returns:
            List of retrieval results
        """
        results = []
        for query in queries:
            results.append(self.retrieve(query, top_k))
        return results


class HybridRetriever:
    """Hybrid retriever combining TF-IDF and semantic search."""
    
    def __init__(self,
                 tfidf_weight: float = 0.4,
                 semantic_weight: float = 0.6,
                 tfidf_max_features: int = 5000,
                 semantic_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize hybrid retriever.
        
        Args:
            tfidf_weight: Weight for TF-IDF scores
            semantic_weight: Weight for semantic scores
            tfidf_max_features: Max features for TF-IDF
            semantic_model: SentenceTransformer model
        """
        self.tfidf_weight = tfidf_weight
        self.semantic_weight = semantic_weight
        
        self.tfidf_retriever = TFIDFRetriever(max_features=tfidf_max_features)
        self.semantic_retriever = SemanticRetriever(model_name=semantic_model)
        
        self.documents = None
    
    def fit(self, documents: List[str]) -> None:
        """
        Fit both retrievers.
        
        Args:
            documents: List of documents
        """
        self.documents = documents
        print("Fitting TF-IDF retriever...")
        self.tfidf_retriever.fit(documents)
        print("Fitting semantic retriever...")
        self.semantic_retriever.fit(documents)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, str, float]]:
        """
        Retrieve documents using hybrid approach.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of tuples (doc_index, doc_text, combined_score)
        """
        if self.documents is None:
            raise ValueError("Retriever not fitted. Call fit() first.")
        
        # Get results from both retrievers with more results for scoring
        tfidf_results = self.tfidf_retriever.retrieve(query, top_k=top_k*3)
        semantic_results = self.semantic_retriever.retrieve(query, top_k=top_k*3)
        
        # Normalize scores
        tfidf_scores = {doc_idx: score for doc_idx, _, score in tfidf_results}
        semantic_scores = {doc_idx: score for doc_idx, _, score in semantic_results}
        
        # Combine scores
        all_indices = set(tfidf_scores.keys()) | set(semantic_scores.keys())
        combined_scores = {}
        
        for idx in all_indices:
            tfidf_score = tfidf_scores.get(idx, 0.0)
            semantic_score = semantic_scores.get(idx, 0.0)
            combined_scores[idx] = (
                self.tfidf_weight * tfidf_score +
                self.semantic_weight * semantic_score
            )
        
        # Get top-k by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_idx, score in sorted_results:
            results.append((
                doc_idx,
                self.documents[doc_idx],
                score
            ))
        
        return results
    
    def batch_retrieve(self, queries: List[str], top_k: int = 5) -> List[List[Tuple[int, str, float]]]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of queries
            top_k: Number of documents per query
            
        Returns:
            List of retrieval results
        """
        results = []
        for query in queries:
            results.append(self.retrieve(query, top_k))
        return results
