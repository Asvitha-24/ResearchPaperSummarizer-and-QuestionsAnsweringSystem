"""
Base model module for summarization and QA using transformers.
Implements fine-tuned models for both tasks.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import (
    pipeline,
    BertTokenizer,
    BertModel,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering
)
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings('ignore')


class SummarizationModel:
    """Handles abstractive text summarization using pre-trained models."""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize summarization model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        # Load summarization pipeline
        self.pipeline = pipeline(
            "summarization",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
    
    def summarize(self, 
                  text: str,
                  max_length: int = 150,
                  min_length: int = 50) -> str:
        """
        Generate summary of input text.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length in tokens
            min_length: Minimum summary length in tokens
            
        Returns:
            Generated summary
        """
        if not text or len(text.split()) < 50:
            return text
        
        try:
            summary = self.pipeline(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Error in summarization: {e}")
            return text[:500]
    
    def batch_summarize(self,
                       texts: List[str],
                       max_length: int = 150,
                       min_length: int = 50) -> List[str]:
        """
        Summarize multiple texts.
        
        Args:
            texts: List of texts to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length
            
        Returns:
            List of summaries
        """
        summaries = []
        for text in texts:
            summaries.append(self.summarize(text, max_length, min_length))
        return summaries


class QuestionAnsweringModel:
    """Handles question answering using pre-trained QA models."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased-distilled-squad"):
        """
        Initialize QA model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        # Load QA pipeline
        self.pipeline = pipeline(
            "question-answering",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
    
    def answer_question(self,
                       question: str,
                       context: str,
                       confidence_threshold: float = 0.0) -> Dict:
        """
        Answer a question given context.
        
        Args:
            question: Question to answer
            context: Context/document to search for answer
            confidence_threshold: Minimum confidence score
            
        Returns:
            Dictionary with answer, score, and span info
        """
        if not context or not question:
            return {
                'answer': 'No answer found',
                'score': 0.0,
                'start': -1,
                'end': -1
            }
        
        try:
            result = self.pipeline(
                question=question,
                context=context,
                top_k=1
            )
            
            if result[0]['score'] >= confidence_threshold:
                return result[0]
            else:
                return {
                    'answer': 'Low confidence answer',
                    'score': result[0]['score'],
                    'start': -1,
                    'end': -1
                }
        except Exception as e:
            print(f"Error in QA: {e}")
            return {
                'answer': 'Error processing question',
                'score': 0.0,
                'start': -1,
                'end': -1
            }
    
    def batch_answer(self,
                    questions: List[str],
                    context: str) -> List[Dict]:
        """
        Answer multiple questions given a context.
        
        Args:
            questions: List of questions
            context: Context document
            
        Returns:
            List of answer dictionaries
        """
        answers = []
        for question in questions:
            answers.append(self.answer_question(question, context))
        return answers


class SemanticSearcher:
    """Handles semantic search for document retrieval in QA."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic search model.
        
        Args:
            model_name: SentenceTransformer model identifier
        """
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.embeddings = None
        self.documents = None
    
    def index_documents(self, documents: List[str]) -> None:
        """
        Create embeddings for documents.
        
        Args:
            documents: List of documents to index
        """
        self.documents = documents
        self.embeddings = self.model.encode(
            documents,
            convert_to_tensor=True,
            show_progress_bar=True
        )
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for most relevant documents.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of tuples (document, similarity_score)
        """
        if self.embeddings is None:
            raise ValueError("Documents not indexed. Call index_documents first.")
        
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Calculate cosine similarities
        from sentence_transformers.util import pytorch_cos_sim
        cos_scores = pytorch_cos_sim(query_embedding, self.embeddings)[0]
        
        # Get top-k results
        top_results = torch.topk(cos_scores, k=min(top_k, len(self.documents)))
        
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            results.append((self.documents[idx], score.item()))
        
        return results


class ResearchPaperQASystem:
    """Complete QA system for research papers combining all components."""
    
    def __init__(self,
                 summarization_model: str = "facebook/bart-large-cnn",
                 qa_model: str = "distilbert-base-uncased-distilled-squad",
                 search_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize complete QA system.
        
        Args:
            summarization_model: Model for summarization
            qa_model: Model for question answering
            search_model: Model for semantic search
        """
        self.summarizer = SummarizationModel(summarization_model)
        self.qa_model = QuestionAnsweringModel(qa_model)
        self.searcher = SemanticSearcher(search_model)
        self.indexed_papers = {}
    
    def index_papers(self, papers_data: Dict[str, str]) -> None:
        """
        Index research papers for retrieval.
        
        Args:
            papers_data: Dictionary of {paper_id: paper_text}
        """
        self.indexed_papers = papers_data
        self.searcher.index_documents(list(papers_data.values()))
    
    def process_paper(self, paper_text: str, summary_length: int = 150) -> Dict:
        """
        Process a single research paper (summarize).
        
        Args:
            paper_text: Full text of the paper
            summary_length: Target summary length
            
        Returns:
            Dictionary with original text and summary
        """
        summary = self.summarizer.summarize(paper_text, max_length=summary_length)
        
        return {
            'original_text': paper_text,
            'summary': summary,
            'original_length': len(paper_text.split()),
            'summary_length': len(summary.split())
        }
    
    def answer_question_on_papers(self,
                                  question: str,
                                  top_k_papers: int = 3) -> Dict:
        """
        Answer a question by searching papers and extracting answers.
        
        Args:
            question: Question to answer
            top_k_papers: Number of papers to consider
            
        Returns:
            Dictionary with answers and source information
        """
        # Search for relevant papers
        relevant_papers = self.searcher.search(question, top_k=top_k_papers)
        
        answers = []
        for paper, relevance_score in relevant_papers:
            answer = self.qa_model.answer_question(question, paper)
            answer['relevance_score'] = relevance_score
            answers.append(answer)
        
        return {
            'question': question,
            'answers': answers,
            'top_k': top_k_papers
        }
