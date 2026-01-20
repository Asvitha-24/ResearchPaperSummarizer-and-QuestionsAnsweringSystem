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
from typing import List, Tuple, Dict

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
    
    def __init__(self, remove_stopwords: bool = False):
        """
        Initialize preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stopwords (default: False for QA)
        """
        self.remove_stopwords = remove_stopwords
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters, extra whitespace.
        
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
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.!?,-]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
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
        Tokenize text into words.
        
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
        return text
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'summary') -> pd.DataFrame:
        """
        Preprocess an entire dataframe.
        
        Args:
            df: Input dataframe
            text_column: Column name to preprocess
            
        Returns:
            Dataframe with preprocessed text
        """
        df = df.copy()
        df['cleaned_' + text_column] = df[text_column].apply(self.preprocess)
        df['sentence_count'] = df['cleaned_' + text_column].apply(
            lambda x: len(self.tokenize_sentences(x))
        )
        df['word_count'] = df['cleaned_' + text_column].apply(
            lambda x: len(self.tokenize_words(x))
        )
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
