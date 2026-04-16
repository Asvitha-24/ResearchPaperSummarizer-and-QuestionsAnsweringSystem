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
    
    def preprocess_text(self, text: str) -> str:
        """
        Comprehensive preprocessing for OCR-extracted text.
        Removes metadata (affiliations, emails, locations) and fixes OCR spacing.
        Focus: return only research content.
        
        Args:
            text: Raw extracted text (possibly with OCR errors)
            
        Returns:
            Cleaned, properly spaced text with only research content
        """
        import re
        
        # ===== PHASE 0: BREAK UP CONCATENATIONS =====
        # Critical: fixes "Departmentof" so patterns below can match
        concat_fixes = {
            r'(?i)departmentof': 'Department of',
            r'(?i)instituteof': 'Institute of',
            r'(?i)universiteof': 'University of',
            r'(?i)collegeof': 'College of',
            r'(?i)schoolof': 'School of',
            r'(?i)techniquesand': 'Techniques and',
            r'(?i)systemand': 'System and',
        }
        for pattern, replacement in concat_fixes.items():
            text = re.sub(pattern, replacement, text)
        
        # ===== PHASE 1: REMOVE EMAILS =====
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        # ===== PHASE 2: REMOVE URLS =====
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        text = re.sub(r'doi\S*:\s*\S+', '', text)
        
        # ===== PHASE 3: REMOVE AFFILIATION BLOCKS =====
        # These appear at the beginning of papers, often with author bylines
        # Target: "Department of Computer Science and Engineering" and similar
        affiliation_patterns = [
            r'Department\s+of\s+(?:Computer\s+)?Science(?:\s+and\s+Engineering)?[^.]*?(?=\n|,|$)',
            r'Department\s+of\s+(?:Computer\s+)?Engineering[^.]*?(?=\n|,|$)',
            r'Department\s+of\s+Informatics[^.]*?(?=\n|,|$)',
            r'Military\s+Institute\s+of\s+Science\s+(?:and\s+)?Technology[^.]*?(?=\n|,|$)',
            r'University\s+of\s+\w+(?:\s+\w+)*',
            r'Institute\s+of\s+\w+(?:\s+\w+)*',
        ]
        for pattern in affiliation_patterns:
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
        
        # ===== PHASE 4: REMOVE GEOGRAPHIC LOCATIONS & COUNTRIES =====
        locations = [
            'Dhaka', 'Jakarta', 'Bandung', 'Tangerang',  # Cities
            'Bangladesh', 'Indonesia', 'Japan', 'Korea', 'Thailand', 'Vietnam',  # Countries
            'Singapore', 'Philippines', 'Malaysia', 'Pakistan', 'India',
        ]
        for loc in locations:
            text = re.sub(rf'\b{re.escape(loc)}\b', '', text, flags=re.IGNORECASE)
        
        # ===== PHASE 5: REMOVE JOURNAL METADATA =====
        # Volume, Issue, Page numbers, Years, Copyright
        text = re.sub(r'\b[Vv]ol(?:ume)?\s*[\.:]*\s*\d+', '', text)
        text = re.sub(r'\b[Nn]o(?:\.|\s)*\d+', '', text)
        text = re.sub(r'\bpp?\.?\s*\d+(?:\s*[-–]\s*\d+)?', '', text)
        text = re.sub(r'\b(19|20)\d{2}\b', '', text)
        text = re.sub(r'©\s*\d{4}', '', text)
        
        # ===== PHASE 6: REMOVE PAGE NUMBERS =====
        # Page numbers typically appear alone at end/start of lines
        text = re.sub(r'\b\d{1,4}\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d{1,4}\s+', '', text, flags=re.MULTILINE)
        
        # ===== PHASE 7: FIX OCR SPACING ERRORS =====
        text = self._fix_ocr_spacing(text)
        
        # ===== PHASE 8: FIX GENERAL SPACING =====
        text = self._fix_spacing(text)
        
        # ===== PHASE 9: FIX LIGATURES & SPECIAL CHARS =====
        text = text.replace('ﬁ', 'fi')
        text = text.replace('ﬂ', 'fl')
        text = text.replace('ﬀ', 'ff')
        text = text.replace('ﬃ', 'ffi')
        text = text.replace('ﬄ', 'ffl')
        
        # ===== PHASE 10: FINAL CLEANUP =====
        # Collapse multiple whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'^\s+|\s+$', '', text)
        
        return text.strip()
    
    def _fix_ocr_spacing(self, text: str) -> str:
        """
        Fix OCR-specific spacing errors like:
        - "appliedinthear eaof" → "applied in the area of"
        - "Processin g" → "Processing"
        - "Proceedingsofthe" → "Proceedings of the"
        - "Literat ure" → "Literature"
        """
        import re
        
        # Pattern 1: Remove spaces inside words (lowercase followed by space then lowercase)
        # "be rleant" → "berlant"
        text = re.sub(r"([a-z0-9'])\ ([a-z])", r'\1\2', text)
        
        # Pattern 2: Fix suffix splits - space before common suffixes
        suffixes_to_rejoin = [
            ('ing', r'(\w)\s+(ing\b)'),
            ('tion', r'(\w)\s+(tion\b)'),
            ('ment', r'(\w)\s+(ment\b)'),
            ('able', r'(\w)\s+(able\b)'),
            ('ness', r'(\w)\s+(ness\b)'),
            ('ful', r'(\w)\s+(ful\b)'),
            ('less', r'(\w)\s+(less\b)'),
            ('ized', r'(\w)\s+(ized\b)'),
            ('ized', r'(\w)\s+(ized\b)'),
        ]
        
        for suffix, pattern in suffixes_to_rejoin:
            text = re.sub(pattern, r'\1' + suffix, text, flags=re.IGNORECASE)
        
        # Pattern 3: Fix split words with capitals mid-word - DISABLED (too risky)
        # This pattern removes too many legitimate spaces, so we skip it
        # Users should rely on BART's language understanding for fixing these errors
        
        # Pattern 4: Remove extra spaces around numbers
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
        
        # Pattern 5: Insert spaces before common prepositions/articles stuck to words
        # "Proceedingsofthe" → "Proceedings of the"
        # "metricsfor" → "metrics for"
        # "Consistencyof" → "Consistency of"
        # "Modelforthe" → "Model for the"
        common_words = [
            ('ofthe', 'of the'),
            ('ofan', 'of an'),
            ('ofa', 'of a'),
            ('inthe', 'in the'),
            ('inan', 'in an'),
            ('ina', 'in a'),
            ('onthe', 'on the'),
            ('onan', 'on an'),
            ('ona', 'on a'),
            ('forthe', 'for the'),
            ('foran', 'for an'),
            ('fora', 'for a'),
            ('tothe', 'to the'),
            ('andit', 'and it'),
            ('andthe', 'and the'),
            ('withthe', 'with the'),
            ('arefactuall', 'are factually'),
            ('Modelforthe', 'Model for the'),
            ('modelforthe', 'model for the'),
            ('PretrainedLanguage', 'Pretrained Language'),
            ('pretrainedlanguage', 'pretrained language'),
            ('TaskAlfonso', 'Task\nAlfonso'),
            ('taskAlfonso', 'task\nAlfonso'),
        ]
        
        for concat, expanded in common_words:
            # Case-insensitive replacement
            text = re.sub(rf'\b{concat}\b', expanded, text, flags=re.IGNORECASE)
        
        # Pattern 6: Fix email addresses mixed with location
        # "Indonesiaalfonso" → "Indonesia\nalfonso" or "Indonesia\n\nalfonso"
        text = re.sub(r'(Indonesia)([a-z]+@)', r'\1\n\2', text, flags=re.IGNORECASE)
        text = re.sub(r'na\.id([A-Z])', r'na.id\n\1', text)  # Break after .id before capitals
        
        # Pattern 7: Fix common concatenated words that start with capital
        # "Themostcommon" → "The most common"
        # "Evaluatingthe" → "Evaluating the"
        capital_patterns = [
            (r'\bThemostcommon', 'The most common'),
            (r'\bEvaluatingthe', 'Evaluating the'),
            (r'\bConsistencyof', 'Consistency of'),
            (r'\bFactualConsistencyof', 'Factual Consistency of'),
            (r'\bMcCann', 'McCann'),  # Fix spacing in names
            (r'\bMc Cann', 'McCann'),
            (r'\bQuestion-AnsweringTask', 'Question-Answering Task'),
            (r'\bQuestion-AnsweringModelforthe', 'Question-Answering Model for the'),
        ]
        
        for pattern, replacement in capital_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Pattern 8: Fix special characters and accents
        # Handle various OCR artifacts for special characters
        text = text.replace('´c', 'ć')
        text = text.replace('c´', 'ć')
        text = text.replace('´s', 'ś')
        text = text.replace('s´', 'ś')
        text = text.replace('n´', 'ń')
        text = text.replace('´n', 'ń')
        text = text.replace('z´', 'ź')
        text = text.replace('´z', 'ź')
        text = text.replace('Kry´sci ´nski', 'Kryściński')
        text = text.replace('c⃝', '')  # Remove copy symbol artifact
        
        # Pattern 9: Fix common name/location patterns
        text = text.replace('Tangera', 'Tangerang')  # Common OCR error for Tangerang
        
        return text
    
    def summarize(self, 
                  text: str,
                  max_length: int = 1000,
                  min_length: int = 400,
                  format_as_points: bool = True) -> str:
        """
        Generate summary of input text with improved formatting.
        
        Args:
            text: Text to summarize (may contain OCR errors)
            max_length: Maximum summary length in tokens
            min_length: Minimum summary length in tokens
            format_as_points: Whether to format output as bullet points
            
        Returns:
            Generated summary formatted with correct spacing
        """
        if not text or len(text.split()) < 50:
            return self.preprocess_text(text)
        
        try:
            if self.model is None or self.tokenizer is None:
                print("[WARN] BART model not loaded, using fallback simple_summarize")
                # Import from utils to avoid circular imports
                from src.utils import simple_summarize
                return simple_summarize(text, max_length, min_length)
            
            # Step 1: Preprocess input to fix OCR errors
            text = self.preprocess_text(text)
            
            # Step 2: Tokenize cleaned text
            inputs = self.tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Step 3: Generate summary
            summary_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                length_penalty=1.5,
                num_beams=5,
                early_stopping=True,
                temperature=0.8,
            )
            
            # Step 4: Decode and clean summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summary = self._fix_spacing(summary)  # Extra spacing cleanup
            
            # Step 5: Format as bullet points if requested
            if format_as_points:
                summary = self._format_as_points(summary)
            
            return summary
            
        except Exception as e:
            print(f"[ERROR] Error in summarization: {e}")
            return self.preprocess_text(text[:500])
    
    def _fix_spacing(self, text: str) -> str:
        """Fix spacing issues in text - handles periods without spaces, concatenated words, etc."""
        import re
        
        # PHASE 1: Fix acronyms with internal spaces (e.g., "C ONS ISTENT" -> "CONSISTENT")
        text = re.sub(r'\b([A-Z])\s+([A-Z])\s+([A-Z]+)\b', r'\1\2\3', text)
        text = re.sub(r'\b([A-Z])\s+([A-Z])\b', r'\1\2', text)
        
        # PHASE 2: Insert spaces before capital letters that follow lowercase letters (main boundary detection)
        # This is the most reliable indicator of word boundaries in concatenated text
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # PHASE 3: Handle obvious low-confidence patterns
        # Fix patterns like "andthe", "ofthe", "inthe" that are clearly two words
        text = re.sub(r'\b(and)(the)([A-Z])', r'\1 \2 \3', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(of)(the)([A-Z])', r'\1 \2 \3', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(in)(the)([A-Z])', r'\1 \2 \3', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(for)(the)([A-Z])', r'\1 \2 \3', text, flags=re.IGNORECASE)
        
        # PHASE 4: Fix missing spaces after periods, commas, and other punctuation followed by capitals
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        text = re.sub(r',([A-Z])', r', \1', text)
        text = re.sub(r':([A-Z])', r': \1', text)
        text = re.sub(r'([!?])([A-Z])', r'\1 \2', text)
        
        # PHASE 5: Clean up multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # PHASE 6: Fix spaces before punctuation (standard English rules)
        text = re.sub(r' ([,.!?;:\)])', r'\1', text)
        
        # PHASE 7: Normalize spacing around brackets
        text = re.sub(r'\(\s+', r'(', text)
        text = re.sub(r'\s+\)', r')', text)
        
        # PHASE 8: Capitalize first letter of sentences
        text = re.sub(r'(^|\.\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text, flags=re.MULTILINE)
        
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
                       max_length: int = 1000,
                       min_length: int = 400) -> List[str]:
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
        
        # Load tokenizer and model for direct use (modern API)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"[OK] QA Model loaded: {model_name}")
        except Exception as e:
            print(f"[ERROR] Failed to load QA model: {e}")
            # Fallback to pipeline if direct loading fails
            try:
                self.pipeline = pipeline(
                    "question-answering",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
                self.tokenizer = None
                self.model = None
                print(f"[WARN] Using fallback pipeline for QA")
            except Exception as e2:
                print(f"[ERROR] Failed to load QA pipeline: {e2}")
                self.pipeline = None
                self.tokenizer = None
                self.model = None
    
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
            
            # Try direct model first if available
            if self.model is not None and self.tokenizer is not None:
                return self._answer_with_model(question, relevant_context, confidence_threshold)
            elif hasattr(self, 'pipeline') and self.pipeline is not None:
                return self._answer_with_pipeline(question, relevant_context, confidence_threshold)
            else:
                print("[ERROR] No QA model or pipeline available")
                return {
                    'answer': 'QA model not available',
                    'score': 0.0,
                    'start': -1,
                    'end': -1
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
    
    def _answer_with_model(self, question: str, context: str, confidence_threshold: float) -> Dict:
        """Answer using direct model inference with modern tokenizer API."""
        # Use the modern tokenizer call syntax (no encode_plus)
        inputs = self.tokenizer(
            question,
            context,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=512,
            truncation="only_second",
            padding="max_length"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        print(f"[QA] Using direct model inference")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get start and end logits
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]
        
        # Get the best answer span
        start_idx = torch.argmax(start_logits)
        end_idx = torch.argmax(end_logits) + 1
        
        # Ensure end_idx > start_idx
        if end_idx <= start_idx:
            end_idx = start_idx + 1
        
        # Decode the answer
        input_ids = inputs["input_ids"][0]
        answer_ids = input_ids[start_idx:end_idx]
        answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True)
        
        # Calculate confidence score
        start_score = torch.softmax(start_logits, dim=0)[start_idx].item()
        end_score = torch.softmax(end_logits, dim=0)[end_idx - 1].item()
        score = (start_score + end_score) / 2
        
        print(f"[QA] Got result with score: {score:.4f}")
        print(f"[QA] Answer: {answer}")
        
        result = {
            'answer': answer if answer.strip() else 'No answer found',
            'score': float(score),
            'start': int(start_idx),
            'end': int(end_idx)
        }
        
        if score >= confidence_threshold:
            return result
        else:
            print(f"[QA] Score below threshold ({score:.4f} < {confidence_threshold})")
            result['confidence'] = 'low'
            return result
    
    def _answer_with_pipeline(self, question: str, context: str, confidence_threshold: float) -> Dict:
        """Answer using the HuggingFace pipeline as fallback."""
        result = self.pipeline(
            question=question,
            context=context,
            top_k=1
        )
        
        print(f"[QA] Got result with score: {result[0]['score']:.4f}")
        print(f"[QA] Answer: {result[0]['answer']}")
        
        if result[0]['score'] >= confidence_threshold:
            return result[0]
        else:
            print(f"[QA] Score below threshold ({result[0]['score']:.4f} < {confidence_threshold})")
            return {
                'answer': result[0]['answer'],
                'score': result[0]['score'],
                'start': result[0].get('start', -1),
                'end': result[0].get('end', -1),
                'confidence': 'low'
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




class StructuredSummarizer:
    """
    Generates structured summaries of research papers organized by sections:
    Introduction, Methods, Results, Conclusion, Key Findings (all in paragraph form).
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn", summarizer_instance=None):
        """
        Initialize the structured summarizer with BART model.
        
        Args:
            model_name: HuggingFace model identifier
            summarizer_instance: SummarizationModel instance for preprocessing
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.summarizer_instance = summarizer_instance  # Reference for preprocessing
        print(f"[OK] Structured Summarizer loaded: {model_name}")
    
    def _segment_text_into_sections(self, text: str) -> Dict[str, str]:
        """
        Attempt to segment text into sections: Introduction, Methods, Results, Conclusion.
        If sections don't exist explicitly, creates them from content chunks.
        """
        import re
        
        sections = {
            'introduction': '',
            'methods': '',
            'results': '',
            'conclusion': '',
            'general': ''
        }
        
        # Common section headers in research papers
        section_patterns = {
            'introduction': r'(introduction|background|motivation|overview)',
            'methods': r'(method|approach|methodology|technical approach|system design)',
            'results': r'(results?|findings?|evaluation|experiments?|experimental results)',
            'conclusion': r'(conclusion|future work|discussion|implications)'
        }
        
        # Try to find explicit sections
        current_section = 'general'
        lines = text.split('\n')
        section_lines = {k: [] for k in sections.keys()}
        
        for line in lines:
            # Check if this line starts a new section
            found_section = False
            for section_name, pattern in section_patterns.items():
                if re.search(pattern, line.lower()) and len(line.split()) < 10:  # Section headers are usually short
                    current_section = section_name
                    found_section = True
                    break
            
            if not found_section and line.strip():
                section_lines[current_section].append(line)
        
        # Join sections
        for section, lines_list in section_lines.items():
            sections[section] = '\n'.join(lines_list).strip()
        
        # If sections are mostly empty, segment by logical chunks instead
        if not sections['introduction'] and not sections['methods']:
            # Divide text into roughly equal parts
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            if len(paragraphs) > 0:
                # Distribute paragraphs across sections
                intro_end = max(1, len(paragraphs) // 5)
                method_end = max(intro_end + 1, (len(paragraphs) * 2) // 5)
                result_end = max(method_end + 1, (len(paragraphs) * 4) // 5)
                
                sections['introduction'] = '\n\n'.join(paragraphs[:intro_end])
                sections['methods'] = '\n\n'.join(paragraphs[intro_end:method_end])
                sections['results'] = '\n\n'.join(paragraphs[method_end:result_end])
                sections['conclusion'] = '\n\n'.join(paragraphs[result_end:])
        
        return sections
    
    def _summarize_section(self, text: str, max_length: int = 200) -> str:
        """Summarize a single section using BART."""
        if not text or len(text.split()) < 20:
            return text[:500] if text else ""
        
        try:
            inputs = self.tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                summary_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=int(max_length * 0.4),
                    length_penalty=1.2,
                    num_beams=5,
                    early_stopping=True,
                    temperature=0.8,
                )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return self._fix_spacing(summary)
        except Exception as e:
            print(f"[ERROR] Section summarization failed: {e}")
            return text[:max_length]
    
    def _fix_spacing(self, text: str) -> str:
        """Fix spacing issues in generated text."""
        import re
        
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        text = re.sub(r'([!?])([A-Z])', r'\1 \2', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r' ([,.!?;:])', r'\1', text)
        
        return text.strip()
    
    def _extract_key_findings(self, results_section: str, conclusion_section: str) -> str:
        """Extract key findings from results and conclusion sections."""
        import re
        
        # Combine results and conclusion
        combined = results_section + ' ' + conclusion_section
        
        # Summarize combined text
        if len(combined.split()) < 30:
            return combined
        
        try:
            inputs = self.tokenizer(combined, max_length=1024, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                summary_ids = self.model.generate(
                    **inputs,
                    max_length=120,
                    min_length=40,
                    length_penalty=1.1,
                    num_beams=4,
                    early_stopping=True,
                )
            
            findings = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return self._fix_spacing(findings)
        except Exception as e:
            print(f"[ERROR] Key findings extraction failed: {e}")
            return combined[:300]
    
    def summarize_structured(self, 
                            text: str,
                            total_length: int = 1000) -> str:
        """
        Generate a structured summary with sections as paragraphs.
        
        Args:
            text: Full research paper text (may contain OCR errors)
            total_length: Target total summary length in tokens
            
        Returns:
            Formatted structured summary
        """
        # Step 1: Preprocess input to fix OCR errors
        if self.summarizer_instance:
            text = self.summarizer_instance.preprocess_text(text)
        
        # Step 2: Allocate tokens per section
        section_tokens = int(total_length / 5)  # 5 main sections
        
        # Step 3: Segment cleaned text
        sections = self._segment_text_into_sections(text)
        
        # Step 4: Summarize each section
        summaries = {}
        for section_name in ['introduction', 'methods', 'results', 'conclusion']:
            if sections[section_name]:
                print(f"[SUMMARY] Summarizing {section_name} section...")
                summaries[section_name] = self._summarize_section(
                    sections[section_name],
                    max_length=section_tokens
                )
            else:
                summaries[section_name] = ""
        
        # Step 5: Extract key findings
        key_findings = ""
        if sections['results'] or sections['conclusion']:
            print("[SUMMARY] Extracting key findings...")
            key_findings = self._extract_key_findings(
                sections['results'],
                sections['conclusion']
            )
        
        # Step 6: Format as structured summary
        formatted_summary = self._format_structured_output(summaries, key_findings)
        
        return formatted_summary
    
    def _format_structured_output(self, summaries: Dict[str, str], key_findings: str) -> str:
        """Format summaries into structured paragraph form."""
        import re
        
        output_parts = []
        
        # Add sections with proper headers
        if summaries.get('introduction'):
            output_parts.append(f"**Introduction**\n\n{summaries['introduction']}\n")
        
        if summaries.get('methods'):
            output_parts.append(f"**Methods**\n\n{summaries['methods']}\n")
        
        if summaries.get('results'):
            output_parts.append(f"**Results**\n\n{summaries['results']}\n")
        
        if summaries.get('conclusion'):
            output_parts.append(f"**Conclusion**\n\n{summaries['conclusion']}\n")
        
        if key_findings:
            output_parts.append(f"**Key Findings**\n\n{key_findings}\n")
        
        # Join with clear separation
        full_summary = '\n---\n\n'.join(output_parts)
        
        # Ensure proper paragraph spacing
        full_summary = re.sub(r'\n{3,}', '\n\n', full_summary)
        
        return full_summary.strip()



class ResearchPaperQASystem:
    """Complete QA system for research papers combining all components."""
    
    def __init__(self,
                 summarization_model: str = "facebook/bart-large-cnn",
                 qa_model: str = "distilbert-base-uncased-distilled-squad"):
        """
        Initialize complete QA system.
        
        Args:
            summarization_model: Model for summarization
            qa_model: Model for question answering
        """
        # Initialize summarizer (required)
        try:
            self.summarizer = SummarizationModel(summarization_model)
            print("[OK] Summarizer initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize summarizer: {e}")
            self.summarizer = None
        
        # Initialize structured summarizer (new)
        try:
            self.structured_summarizer = StructuredSummarizer(
                summarization_model, 
                summarizer_instance=self.summarizer
            )
            print("[OK] Structured Summarizer initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize structured summarizer: {e}")
            self.structured_summarizer = None
        
        # Initialize QA model (optional)
        try:
            self.qa_model = QuestionAnsweringModel(qa_model)
            print("[OK] QA model initialized")
        except Exception as e:
            print(f"[WARN] Failed to initialize QA model (optional): {e}")
            self.qa_model = None
    
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
