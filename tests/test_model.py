"""
Unit tests for the model module.
"""

import pytest
import torch
from model import (
    SummarizationModel,
    QuestionAnsweringModel,
    SemanticSearcher,
    ResearchPaperQASystem
)


class TestSummarizationModel:
    """Test suite for SummarizationModel."""
    
    @pytest.fixture
    def summarizer(self):
        """Initialize summarizer."""
        return SummarizationModel()
    
    def test_summarize_short_text(self, summarizer):
        """Test summarization of short text."""
        text = "This is a short text."
        result = summarizer.summarize(text)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_summarize_long_text(self, summarizer):
        """Test summarization of long text."""
        text = " ".join(["This is a sample sentence about machine learning."] * 20)
        result = summarizer.summarize(text, max_length=50, min_length=20)
        assert isinstance(result, str)
        assert len(result.split()) >= 20
    
    def test_batch_summarize(self, summarizer):
        """Test batch summarization."""
        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing deals with text and speech."
        ]
        results = summarizer.batch_summarize(texts)
        assert len(results) == len(texts)
        assert all(isinstance(r, str) for r in results)


class TestQuestionAnsweringModel:
    """Test suite for QuestionAnsweringModel."""
    
    @pytest.fixture
    def qa_model(self):
        """Initialize QA model."""
        return QuestionAnsweringModel()
    
    def test_answer_question_valid(self, qa_model):
        """Test answering a valid question."""
        context = "John is a doctor. He works at the hospital."
        question = "What is John's profession?"
        
        result = qa_model.answer_question(question, context)
        assert 'answer' in result
        assert 'score' in result
        assert isinstance(result['score'], float)
    
    def test_answer_question_empty_context(self, qa_model):
        """Test with empty context."""
        question = "What is this?"
        context = ""
        
        result = qa_model.answer_question(question, context)
        assert result['score'] == 0.0
    
    def test_batch_answer(self, qa_model):
        """Test batch question answering."""
        context = "Paris is the capital of France. It is known for the Eiffel Tower."
        questions = [
            "What is the capital of France?",
            "What is Paris known for?"
        ]
        
        results = qa_model.batch_answer(questions, context)
        assert len(results) == len(questions)
        assert all('answer' in r for r in results)


class TestSemanticSearcher:
    """Test suite for SemanticSearcher."""
    
    @pytest.fixture
    def searcher(self):
        """Initialize searcher."""
        return SemanticSearcher()
    
    def test_index_documents(self, searcher):
        """Test document indexing."""
        documents = [
            "Machine learning is powerful.",
            "Deep learning uses neural networks.",
            "NLP processes text data."
        ]
        
        searcher.index_documents(documents)
        assert searcher.documents == documents
        assert searcher.embeddings is not None
    
    def test_search_valid(self, searcher):
        """Test semantic search."""
        documents = [
            "The cat sat on the mat.",
            "Dogs are loyal animals.",
            "Birds fly in the sky."
        ]
        
        searcher.index_documents(documents)
        results = searcher.search("feline", top_k=2)
        
        assert len(results) <= 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    
    def test_search_before_indexing(self, searcher):
        """Test search before indexing raises error."""
        with pytest.raises(ValueError):
            searcher.search("query")


class TestResearchPaperQASystem:
    """Test suite for ResearchPaperQASystem."""
    
    @pytest.fixture
    def qa_system(self):
        """Initialize complete QA system."""
        return ResearchPaperQASystem()
    
    def test_system_initialization(self, qa_system):
        """Test system initialization."""
        assert qa_system.summarizer is not None
        assert qa_system.qa_model is not None
        assert qa_system.searcher is not None
    
    def test_process_paper(self, qa_system):
        """Test paper processing."""
        paper_text = "This is a research paper about machine learning. " * 10
        
        result = qa_system.process_paper(paper_text, summary_length=100)
        assert 'original_text' in result
        assert 'summary' in result
        assert 'original_length' in result
        assert 'summary_length' in result


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
