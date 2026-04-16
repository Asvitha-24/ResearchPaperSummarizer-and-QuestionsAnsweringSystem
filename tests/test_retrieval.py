"""
Unit tests for the retrieval module.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path so we can import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import (
    TFIDFRetriever
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


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
