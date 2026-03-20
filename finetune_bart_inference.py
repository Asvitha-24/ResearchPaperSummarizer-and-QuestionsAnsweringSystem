"""
Utility module for loading and using the fine-tuned BART model.
Provides easy inference capabilities after fine-tuning.
"""

import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from typing import List, Optional
import warnings

warnings.filterwarnings('ignore')


class FinetuedBartSummarizer:
    """
    Wrapper for using the fine-tuned BART model for summarization.
    """
    
    def __init__(self, 
                 model_dir: str = "./checkpoints/bart_finetuned",
                 device: Optional[str] = None):
        """
        Initialize the fine-tuned BART summarizer.
        
        Args:
            model_dir: Directory containing the fine-tuned model
            device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading fine-tuned BART model from {model_dir}...")
        print(f"Using device: {self.device}")
        
        try:
            self.tokenizer = BartTokenizer.from_pretrained(model_dir)
            self.model = BartForConditionalGeneration.from_pretrained(model_dir)
            self.model.to(self.device)
            self.model.eval()
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def summarize(self,
                  text: str,
                  max_length: int = 150,
                  min_length: int = 50,
                  num_beams: int = 4,
                  length_penalty: float = 2.0,
                  no_repeat_ngram_size: int = 3) -> str:
        """
        Generate summary for input text.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length in tokens
            min_length: Minimum summary length in tokens
            num_beams: Number of beams for beam search
            length_penalty: Higher values encourage longer summaries
            no_repeat_ngram_size: Prevent repeating n-grams
            
        Returns:
            Generated summary
        """
        if not text or len(text.strip()) < 10:
            return text
        
        try:
            inputs = self.tokenizer.encode(
                text,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            )
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    early_stopping=True,
                    no_repeat_ngram_size=no_repeat_ngram_size
                )
            
            summary = self.tokenizer.batch_decode(
                summary_ids,
                skip_special_tokens=True
            )[0]
            
            return summary
        
        except Exception as e:
            print(f"Error during summarization: {e}")
            return text[:500]
    
    def batch_summarize(self,
                       texts: List[str],
                       max_length: int = 150,
                       min_length: int = 50,
                       **kwargs) -> List[str]:
        """
        Summarize multiple texts.
        
        Args:
            texts: List of texts to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length
            **kwargs: Additional arguments for summarization
            
        Returns:
            List of summaries
        """
        summaries = []
        for text in texts:
            summary = self.summarize(text, max_length, min_length, **kwargs)
            summaries.append(summary)
        return summaries
    
    def compare_with_pretrained(self,
                               text: str,
                               pretrained_model: str = "facebook/bart-large-cnn",
                               max_length: int = 150) -> dict:
        """
        Compare fine-tuned vs pre-trained model outputs.
        
        Args:
            text: Text to summarize
            pretrained_model: Pre-trained model to compare against
            max_length: Maximum summary length
            
        Returns:
            Dictionary with both summaries
        """
        # Get summary from fine-tuned model
        finetuned_summary = self.summarize(text, max_length=max_length)
        
        # Load and use pre-trained model
        from transformers import pipeline
        pretrained_summarizer = pipeline(
            "summarization",
            model=pretrained_model,
            device=0 if self.device == "cuda" else -1
        )
        
        pretrained_summary = pretrained_summarizer(
            text,
            max_length=max_length,
            min_length=50,
            do_sample=False
        )[0]['summary_text']
        
        return {
            'finetuned': finetuned_summary,
            'pretrained': pretrained_summary,
            'input_length': len(text.split()),
            'finetuned_length': len(finetuned_summary.split()),
            'pretrained_length': len(pretrained_summary.split()),
        }


def load_and_test():
    """Test the fine-tuned model with sample texts."""
    
    # Initialize summarizer
    summarizer = FinetuedBartSummarizer()
    
    # Sample test texts
    test_texts = [
        """
        Deep learning has revolutionized the field of artificial intelligence.
        Neural networks with multiple layers can learn hierarchical representations
        of data, enabling breakthrough performance in computer vision, natural language
        processing, and other domains. This work presents a novel architecture combining
        convolutional and attention mechanisms for improved image classification.
        """,
        """
        Transformer models have become the dominant architecture in modern NLP.
        The self-attention mechanism allows these models to capture long-range
        dependencies in text more effectively than RNNs. We introduce a new variant
        with improved efficiency and propose novel pre-training objectives.
        """
    ]
    
    print("\n" + "=" * 80)
    print("Testing Fine-tuned BART Model")
    print("=" * 80)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*80}")
        print(f"Sample {i}:")
        print(f"{'='*80}")
        print(f"\nInput: {text.strip()[:150]}...")
        
        summary = summarizer.summarize(text)
        print(f"\nSummary: {summary}")
        print(f"Length: {len(summary.split())} words")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    load_and_test()
