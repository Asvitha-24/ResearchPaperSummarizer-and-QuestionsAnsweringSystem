"""
Data preprocessing module for research paper summarization and QA system.
Handles text cleaning, tokenization, and data preparation.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Set
from difflib import SequenceMatcher
import hashlib

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class DataPreprocessor:
    """Handles all data preprocessing operations."""
    
    # Academic stopwords to filter
    ACADEMIC_STOPWORDS = {
        'abstract', 'introduction', 'conclusion', 'acknowledgements',
        'references', 'methodology', 'results', 'discussion',
        'author', 'paper', 'study', 'research', 'proposed',
        'et al', 'et al.', 'fig', 'figure', 'table', 'equation',
        'respectively', 'herein', 'thereof', 'thereof'
    }
    
    def __init__(self, remove_stopwords: bool = False):
        """
        Initialize preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stopwords (default: False for QA)
        """
        self.remove_stopwords = remove_stopwords
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        self.academic_stopwords = self.ACADEMIC_STOPWORDS
        self._text_hashes: Set[str] = set()  # For duplicate detection
    
    @staticmethod
    def _hash_text(text: str) -> str:
        """Generate hash of normalized text for duplicate detection."""
        normalized = ' '.join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters, extra whitespace, and academic noise.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove citations and references [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\(\d+\)', '', text)
        
        # Remove arXiv IDs and paper identifiers
        text = re.sub(r'arXiv:\d+\.\d+|arxiv:\d+\.\d+', '', text)
        text = re.sub(r'doi:\s*\S+', '', text, flags=re.IGNORECASE)
        
        # Remove extra spaces before citations (et al., etc.)
        text = re.sub(r'\s+et\s+al\.?', ' et al.', text)
        
        # Normalize common abbreviations
        text = re.sub(r'\be\.g\.\s+', 'e.g. ', text)
        text = re.sub(r'\bi\.e\.\s+', 'i.e. ', text)
        
        # Remove multiple spaces and special math symbols
        text = re.sub(r'[^\w\s.!?,-α-ω∈∉∅∪∩]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_boilerplate(self, text: str) -> str:
        """
        Remove common boilerplate text patterns.
        
        Args:
            text: Text to clean
            
        Returns:
            Text with boilerplate removed
        """
        # Remove funding/author information lines
        text = re.sub(r'(?i)(funding|acknowledgement|supported by|grant\s+\w+).*?(?=\n|$)', '', text)
        text = re.sub(r'(?i)(author|correspondence).*?email.*?(?=\n|$)', '', text)
        
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize all whitespace variations."""
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def is_duplicate(self, text: str, similarity_threshold: float = 0.95) -> bool:
        """
        Check if text is a duplicate or near-duplicate of previously seen text.
        
        Args:
            text: Text to check
            similarity_threshold: Threshold for fuzzy matching (0-1)
            
        Returns:
            True if duplicate detected, False otherwise
        """
        text_hash = self._hash_text(text)
        
        if text_hash in self._text_hashes:
            return True
        
        self._text_hashes.add(text_hash)
        return False
    
    def meets_quality_criteria(self, text: str, 
                              min_words: int = 50, 
                              max_words: int = 5000,
                              min_sentences: int = 2) -> bool:
        """
        Check if text meets minimum quality criteria.
        
        Args:
            text: Text to validate
            min_words: Minimum word count
            max_words: Maximum word count
            min_sentences: Minimum sentence count
            
        Returns:
            True if text meets criteria, False otherwise
        """
        words = len(self.tokenize_words(text))
        sentences = len(self.tokenize_sentences(text))
        
        if words < min_words or words > max_words:
            return False
        if sentences < min_sentences:
            return False
        
        # Check for mostly numbers or special characters
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text) if text else 0
        if alpha_ratio < 0.5:  # Less than 50% alphabetic
            return False
        
        return True
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenize text into sentences.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of sentences
        """
        try:
            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except:
            return [text]
    
    def tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize text into words with optional stopword removal.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of words (tokens)
        """
        words = word_tokenize(text.lower())
        
        if self.remove_stopwords:
            words = [w for w in words if w not in self.stop_words and w not in string.punctuation]
        
        return words
    
    def preprocess(self, text: str) -> str:
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        text = self.clean_text(text)
        text = self.remove_boilerplate(text)
        text = self.normalize_whitespace(text)
        return text
    
    def preprocess_dataframe(self, df: pd.DataFrame, 
                            text_column: str = 'summary',
                            remove_duplicates: bool = True,
                            remove_low_quality: bool = True) -> pd.DataFrame:
        """
        Preprocess an entire dataframe with quality filtering.
        
        Args:
            df: Input dataframe
            text_column: Column name to preprocess
            remove_duplicates: Whether to filter duplicates
            remove_low_quality: Whether to filter low-quality texts
            
        Returns:
            Dataframe with preprocessed text and quality indicators
        """
        df = df.copy()
        
        # Clean text
        df['cleaned_' + text_column] = df[text_column].apply(self.preprocess)
        
        # Add quality metrics
        df['sentence_count'] = df['cleaned_' + text_column].apply(
            lambda x: len(self.tokenize_sentences(x))
        )
        df['word_count'] = df['cleaned_' + text_column].apply(
            lambda x: len(self.tokenize_words(x))
        )
        
        # Check for duplicates
        if remove_duplicates:
            df['is_duplicate'] = df['cleaned_' + text_column].apply(self.is_duplicate)
        else:
            df['is_duplicate'] = False
        
        # Check quality criteria
        if remove_low_quality:
            df['passes_quality'] = df['cleaned_' + text_column].apply(self.meets_quality_criteria)
        else:
            df['passes_quality'] = True
        
        # Filter out bad records
        if remove_duplicates or remove_low_quality:
            original_count = len(df)
            df = df[~df['is_duplicate'] & df['passes_quality']].reset_index(drop=True)
            removed_count = original_count - len(df)
            if removed_count > 0:
                print(f"Removed {removed_count} records ({removed_count/original_count*100:.1f}%) due to quality filters")
        
        return df


class DataSplitter:
    """Handles train-test-validation splitting."""
    
    @staticmethod
    def split_data(df: pd.DataFrame, 
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15,
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input dataframe
            train_ratio: Training set ratio (default: 0.7)
            val_ratio: Validation set ratio (default: 0.15)
            test_ratio: Test set ratio (default: 0.15)
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
        
        np.random.seed(random_state)
        
        # Shuffle indices
        indices = np.random.permutation(len(df))
        
        train_end = int(len(df) * train_ratio)
        val_end = train_end + int(len(df) * val_ratio)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        train_df = df.iloc[train_indices].reset_index(drop=True)
        val_df = df.iloc[val_indices].reset_index(drop=True)
        test_df = df.iloc[test_indices].reset_index(drop=True)
        
        return train_df, val_df, test_df
    
    @staticmethod
    def stratified_split(df: pd.DataFrame,
                        category_column: str = 'category_code',
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15,
                        test_ratio: float = 0.15,
                        random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data while maintaining category distribution.
        
        Args:
            df: Input dataframe
            category_column: Column for stratification
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            df,
            test_size=test_ratio,
            stratify=df[category_column],
            random_state=random_state
        )
        
        # Second split: train vs val
        val_size = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(
            train_val,
            test_size=val_size,
            stratify=train_val[category_column],
            random_state=random_state
        )
        
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
