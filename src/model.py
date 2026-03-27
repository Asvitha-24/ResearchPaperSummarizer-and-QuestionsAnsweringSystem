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
        
        # Load BART tokenizer and model directly
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.to(self.device)
            print(f"[OK] BART model loaded successfully: {model_name}")
        except Exception as e:
            print(f"[ERROR] Failed to load BART model: {e}")
            self.tokenizer = None
            self.model = None
    
    def summarize(self, 
                  text: str,
                  max_length: int = 250,
                  min_length: int = 100,
                  format_as_points: bool = True) -> str:
        """
        Generate summary of input text with improved formatting.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length in tokens (increased for longer summaries)
            min_length: Minimum summary length in tokens
            format_as_points: Whether to format output as bullet points
            
        Returns:
            Generated summary formatted as points/notes
        """
        if not text or len(text.split()) < 50:
            return text
        
        try:
            if self.model is None or self.tokenizer is None:
                print("[WARN] BART model not loaded, returning text truncated")
                return text[:500]
            
            # Tokenize input text
            inputs = self.tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate longer summary with better parameters
            summary_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                length_penalty=1.5,  # Reduced from 2.0 to allow longer output
                num_beams=5,  # Increased from 4 for better quality
                early_stopping=True,
                temperature=0.8,  # Add temperature for better diversity
            )
            
            # Decode summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # Fix spacing issues (ensure proper space between words)
            summary = self._fix_spacing(summary)
            
            # Format as bullet points if requested
            if format_as_points:
                summary = self._format_as_points(summary)
            
            return summary
            
        except Exception as e:
            print(f"[ERROR] Error in summarization: {e}")
            return text[:500]
    
    def _fix_spacing(self, text: str) -> str:
        """Fix spacing issues in text - handles periods without spaces, concatenated words, etc."""
        import re
        
        # Fix missing spaces after periods followed by capital letters (e.g., "word.Next" -> "word. Next")
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        
        # Fix other punctuation marks followed directly by capital letters
        text = re.sub(r'([!?])([A-Z])', r'\1 \2', text)
        
        # Fix missing spaces between lowercase and uppercase (CamelCase)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Fix multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # Fix spaces before punctuation (standard English rules)
        text = re.sub(r' ([,.!?;:])', r'\1', text)
        
        return text.strip()
    
    def _format_as_points(self, text: str) -> str:
        """Format summary as bullet points for better readability"""
        import re
        
        # Split by sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return text
        
        # Group into main points (2-3 sentences per point)
        points = []
        current_point = []
        
        for i, sentence in enumerate(sentences):
            current_point.append(sentence)
            
            # Form a point every 2-3 sentences or at end
            if len(current_point) >= 2 or i == len(sentences) - 1:
                point = ' '.join(current_point)
                # Capitalize if needed
                if point and not point[0].isupper():
                    point = point[0].upper() + point[1:]
                points.append(f"• {point}")
                current_point = []
        
        return '\n'.join(points) if points else text
    
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
                       confidence_threshold: float = 0.1) -> Dict:
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
            print("[QA] Missing context or question")
            return {
                'answer': 'No answer found',
                'score': 0.0,
                'start': -1,
                'end': -1
            }
        
        try:
            # Split context into chunks if too long (max ~512 tokens for BERT)
            # Each token ~= 1-1.3 words, so limit to ~400 words per chunk
            max_chunk_length = 2000  # characters
            
            print(f"[QA] Processing question: '{question}'")
            print(f"[QA] Context length: {len(context)} characters")
            
            # If context is too long, try to find relevant sentences first
            if len(context) > max_chunk_length:
                # Split by sentences and find relevant ones
                import re
                sentences = re.split(r'[.!?]+', context)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                print(f"[QA] Context too long, split into {len(sentences)} sentences")
                
                # Score each sentence based on question similarity (simple keyword matching)
                question_words = set(question.lower().split())
                scored_sentences = []
                for idx, sent in enumerate(sentences):
                    sent_words = set(sent.lower().split())
                    overlap = len(question_words & sent_words)
                    scored_sentences.append((idx, sent, overlap))
                
                # Sort by relevance and take top sentences
                scored_sentences.sort(key=lambda x: x[2], reverse=True)
                relevant_context = ' '.join([sent for _, sent, _ in scored_sentences[:5]])
                
                print(f"[QA] Using {len(relevant_context)} chars of relevant context")
            else:
                relevant_context = context
            
            # Truncate if still too long
            if len(relevant_context) > max_chunk_length:
                relevant_context = relevant_context[:max_chunk_length]
                print(f"[QA] Truncated to {len(relevant_context)} characters")
            
            result = self.pipeline(
                question=question,
                context=relevant_context,
                top_k=1
            )
            
            print(f"[QA] Got result with score: {result[0]['score']:.4f}")
            print(f"[QA] Answer: {result[0]['answer']}")
            
            if result[0]['score'] >= confidence_threshold:
                return result[0]
            else:
                print(f"[QA] Score below threshold ({result[0]['score']:.4f} < {confidence_threshold})")
                return {
                    'answer': result[0]['answer'],  # Still return the answer even if low confidence
                    'score': result[0]['score'],
                    'start': result[0].get('start', -1),
                    'end': result[0].get('end', -1),
                    'confidence': 'low'
                }
        except Exception as e:
            print(f"[ERROR] Error in QA processing: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'answer': 'Unable to process - please try a different question or provide more context',
                'score': 0.0,
                'start': -1,
                'end': -1,
                'error': str(e)
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
        # Initialize summarizer (required)
        try:
            self.summarizer = SummarizationModel(summarization_model)
            print("[OK] Summarizer initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize summarizer: {e}")
            self.summarizer = None
        
        # Initialize QA model (optional)
        try:
            self.qa_model = QuestionAnsweringModel(qa_model)
            print("[OK] QA model initialized")
        except Exception as e:
            print(f"[WARN] Failed to initialize QA model (optional): {e}")
            self.qa_model = None
        
        # Initialize searcher (optional)
        try:
            self.searcher = SemanticSearcher(search_model)
            print("[OK] Semantic searcher initialized")
        except Exception as e:
            print(f"[WARN] Failed to initialize semantic searcher (optional): {e}")
            self.searcher = None
        
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
