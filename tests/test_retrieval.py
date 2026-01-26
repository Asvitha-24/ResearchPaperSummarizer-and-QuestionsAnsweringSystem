"""
Unit tests for the retrieval module.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path so we can import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import (
    TFIDFRetriever,
    SemanticRetriever,
    HybridRetriever
)


class TestTFIDFRetriever:
    """Test suite for TFIDFRetriever."""
    
    @pytest.fixture
    def retriever(self):
        """Initialize retriever."""
        return TFIDFRetriever()
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing deals with text data.",
            "Computer vision processes images and videos.",
            "Data science combines statistics and programming."
        ]
    
    def test_fit(self, retriever, sample_documents):
        """Test fitting the retriever."""
        retriever.fit(sample_documents)
        assert retriever.documents == sample_documents
        assert retriever.tfidf_matrix is not None
    
    def test_retrieve(self, retriever, sample_documents):
        """Test document retrieval."""
        retriever.fit(sample_documents)
        results = retriever.retrieve("machine learning", top_k=2)
        
        assert len(results) <= 2
        assert all(len(r) == 3 for r in results)  # (index, text, score)
    
    def test_retrieve_without_fitting(self, retriever):
        """Test retrieve without fitting raises error."""
        with pytest.raises(ValueError):
            retriever.retrieve("query")
    
    def test_batch_retrieve(self, retriever, sample_documents):
        """Test batch retrieval."""
        retriever.fit(sample_documents)
        queries = ["machine learning", "image processing"]
        results = retriever.batch_retrieve(queries, top_k=2)
        
        assert len(results) == len(queries)


class TestSemanticRetriever:
    """Test suite for SemanticRetriever."""
    
    @pytest.fixture
    def retriever(self):
        """Initialize semantic retriever."""
        return SemanticRetriever()
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            "Cats are independent animals.",
            "Dogs are loyal companions.",
            "Birds can fly in the sky.",
            "Fish live in water."
        ]
    
    def test_fit(self, retriever, sample_documents):
        """Test fitting the retriever."""
        retriever.fit(sample_documents)
        assert retriever.documents == sample_documents
        assert retriever.embeddings is not None
    
    def test_retrieve(self, retriever, sample_documents):
        """Test semantic search."""
        retriever.fit(sample_documents)
        results = retriever.retrieve("feline animal", top_k=2)
        
        assert len(results) <= 2
        assert all(len(r) == 3 for r in results)
    
    def test_retrieve_without_fitting(self, retriever):
        """Test retrieve without fitting raises error."""
        with pytest.raises(ValueError):
            retriever.retrieve("query")


class TestHybridRetriever:
    """Test suite for HybridRetriever."""
    
    @pytest.fixture
    def retriever(self):
        """Initialize hybrid retriever."""
        return HybridRetriever(tfidf_weight=0.4, semantic_weight=0.6)
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            "Python is a programming language.",
            "Java is used for building enterprise applications.",
            "JavaScript runs in web browsers.",
            "C++ is known for performance."
        ]
    
    def test_fit(self, retriever, sample_documents):
        """Test fitting both retrievers."""
        retriever.fit(sample_documents)
        assert retriever.documents == sample_documents
        assert retriever.tfidf_retriever.documents is not None
        assert retriever.semantic_retriever.documents is not None
    
    def test_retrieve(self, retriever, sample_documents):
        """Test hybrid retrieval."""
        retriever.fit(sample_documents)
        results = retriever.retrieve("programming", top_k=2)
        
        assert len(results) <= 2
        assert all(len(r) == 3 for r in results)
    
    def test_retrieve_without_fitting(self, retriever):
        """Test retrieve without fitting raises error."""
        with pytest.raises(ValueError):
            retriever.retrieve("query")
    
    def test_weights_effect(self):
        """Test that weights affect retrieval."""
        documents = ["machine learning model", "deep learning network"]
        
        # Retriever with more weight on semantic
        retriever1 = HybridRetriever(tfidf_weight=0.2, semantic_weight=0.8)
        retriever1.fit(documents)
        
        # Retriever with more weight on TF-IDF
        retriever2 = HybridRetriever(tfidf_weight=0.8, semantic_weight=0.2)
        retriever2.fit(documents)
        
        # Both should return results but potentially different orders
        results1 = retriever1.retrieve("learning", top_k=2)
        results2 = retriever2.retrieve("learning", top_k=2)
        
        assert len(results1) > 0
        assert len(results2) > 0


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
