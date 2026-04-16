"""
Information retrieval module for QA system.
Implements TF-IDF based document retrieval.
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
