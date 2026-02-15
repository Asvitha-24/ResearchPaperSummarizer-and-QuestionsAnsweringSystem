"""
QA Model Evaluation using Hugging Face Transformers - Google Colab Version
===========================================================================

This script evaluates multiple pre-trained Question Answering models using
the Hugging Face Transformers library and SQuAD metrics.

Models evaluated:
1. distilbert-base-uncased-distilled-squad (DistilBERT - Lighter, Faster)
2. bert-base-uncased (BERT - Standard)
3. roberta-base (RoBERTa - Improved)
4. albert-base-v2 (ALBERT - Parameter-efficient)

Metrics computed:
- Exact Match (EM): Whether the prediction exactly matches ground truth
- F1-Score: Harmonic mean of precision and recall at token level

Usage in Google Colab:
1. Copy all cells below into a Colab notebook
2. Run cells sequentially
3. Results will be displayed and saved
"""

# ============================================================================
# CELL 1: Install Required Libraries (Colab Only)
# ============================================================================
"""
!pip install -q transformers torch datasets evaluate
"""

# ============================================================================
# CELL 2: Import Libraries
# ============================================================================

import warnings
import json
import re
from typing import Dict, List, Tuple
from collections import defaultdict

from transformers import pipeline
from datasets import load_metric
import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# ============================================================================
# CELL 3: Define QA Model Evaluator Class
# ============================================================================

class QAModelEvaluator:
    """
    Evaluates multiple QA models using SQuAD metrics (Exact Match and F1-Score).
    """
    
    def __init__(self):
        """Initialize the evaluator with SQuAD metric."""
        self.metric = load_metric('squad')
        self.models = {
            'DistilBERT': 'distilbert-base-uncased-distilled-squad',
            'BERT': 'bert-base-uncased',
            'RoBERTa': 'roberta-base',
            'ALBERT': 'albert-base-v2'
        }
        self.pipelines = {}
        self.results = {}
        
    def _cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def load_models(self):
        """Load all QA models using the pipeline."""
        print("=" * 80)
        print("LOADING QUESTION ANSWERING MODELS")
        print("=" * 80)
        
        for model_name, model_id in self.models.items():
            print(f"\nLoading {model_name:<15} ({model_id})...", end=" ", flush=True)
            try:
                qa_pipeline = pipeline(
                    'question-answering',
                    model=model_id,
                    device=0 if self._cuda_available() else -1
                )
                self.pipelines[model_name] = qa_pipeline
                print("✓ Loaded successfully")
            except Exception as e:
                print(f"✗ Error: {str(e)[:50]}")
    
    def prepare_samples(self) -> Tuple[str, str, List[Dict], List[Dict]]:
        """
        Prepare multiple sample contexts and questions for evaluation.
        
        Returns:
            Tuple of (samples_list, ground_truth_list)
        """
        samples = [
            {
                "context": (
                    "Machine learning is a subset of artificial intelligence (AI) that focuses on "
                    "enabling computers to learn from and make decisions based on data, without being "
                    "explicitly programmed. Deep learning, a subset of machine learning, uses neural "
                    "networks with multiple layers to learn representations of data. The field has seen "
                    "remarkable progress in recent years, with applications ranging from natural language "
                    "processing to computer vision."
                ),
                "question": "What is machine learning a subset of?",
                "ground_truths": [
                    {"text": "artificial intelligence", "answer_start": 45},
                    {"text": "AI", "answer_start": 56}
                ]
            },
            {
                "context": (
                    "Transformers are deep learning models that rely entirely on an attention mechanism "
                    "to draw global dependencies between input and output. They were introduced in the paper "
                    "'Attention is All You Need' by Vaswani et al. in 2017. Transformers have since become "
                    "the foundation for many state-of-the-art models in natural language processing, including "
                    "BERT, GPT, and T5. This architecture has revolutionized the field by enabling parallel "
                    "processing and improved performance on various NLP tasks."
                ),
                "question": "In what year were Transformers introduced?",
                "ground_truths": [
                    {"text": "2017", "answer_start": 168}
                ]
            },
            {
                "context": (
                    "Python is a high-level, interpreted programming language known for its simplicity "
                    "and readability. Created by Guido van Rossum and first released in 1991, Python has "
                    "become one of the most popular programming languages in the world. It is widely used in "
                    "web development, data science, artificial intelligence, scientific computing, and automation. "
                    "The Python Software Foundation maintains and promotes the language and its ecosystem."
                ),
                "question": "Who created Python?",
                "ground_truths": [
                    {"text": "Guido van Rossum", "answer_start": 111}
                ]
            }
        ]
        
        return samples
    
    def normalize_answer(self, answer: str) -> str:
        """
        Normalize answer text for fair evaluation.
        Removes articles, punctuation, and performs lowercasing and whitespace normalization.
        
        Args:
            answer: Raw answer text
            
        Returns:
            Normalized answer
        """
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(answer))))
    
    def compute_exact_match(self, prediction: str, ground_truth: str) -> float:
        """
        Compute Exact Match (EM) score.
        
        Args:
            prediction: Model's predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            EM score (1.0 if exact match, 0.0 otherwise)
        """
        return float(
            self.normalize_answer(prediction) == self.normalize_answer(ground_truth)
        )
    
    def compute_f1(self, prediction: str, ground_truth: str) -> float:
        """
        Compute F1-Score based on token-level overlap.
        
        Args:
            prediction: Model's predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            F1 score (0.0 to 1.0)
        """
        pred_tokens = self.normalize_answer(prediction).split()
        truth_tokens = self.normalize_answer(ground_truth).split()
        
        common = set(pred_tokens) & set(truth_tokens)
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return float(pred_tokens == truth_tokens)
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall) \
            if (precision + recall) > 0 else 0
        
        return f1
    
    def metric_max_over_ground_truths(
        self,
        metric_fn,
        prediction: str,
        ground_truths: List[Dict]
    ) -> float:
        """
        Compute maximum metric score over all ground truth answers.
        Takes the best score if multiple valid answers exist.
        
        Args:
            metric_fn: Metric function to apply
            prediction: Model's predicted answer
            ground_truths: List of ground truth answers (dict with 'text' key)
            
        Returns:
            Maximum metric score
        """
        scores = [
            metric_fn(prediction, gt['text'])
            for gt in ground_truths
        ]
        return max(scores) if scores else 0
    
    def evaluate_models(self):
        """Evaluate all loaded models on sample data."""
        print("\n" + "=" * 80)
        print("EVALUATING MODELS ON SAMPLE DATA")
        print("=" * 80)
        
        samples = self.prepare_samples()
        all_results = defaultdict(lambda: {'em': [], 'f1': [], 'score': []})
        
        for sample_idx, sample in enumerate(samples, 1):
            context = sample['context']
            question = sample['question']
            ground_truths = sample['ground_truths']
            
            print(f"\n{'─' * 80}")
            print(f"Sample {sample_idx}/{len(samples)}")
            print(f"{'─' * 80}")
            print(f"Context: {context[:100]}...")
            print(f"Question: {question}\n")
            print("Ground Truth Answers:")
            for gt in ground_truths:
                print(f"  - {gt['text']}")
            
            print(f"\n{' ' * 40}Results:")
            print("-" * 80)
            
            for model_name, pipeline_model in self.pipelines.items():
                try:
                    # Get prediction from model
                    prediction = pipeline_model(
                        question=question,
                        context=context,
                        max_answer_len=20
                    )
                    
                    pred_answer = prediction['answer']
                    pred_score = prediction['score']
                    
                    # Compute metrics
                    em_score = self.metric_max_over_ground_truths(
                        self.compute_exact_match,
                        pred_answer,
                        ground_truths
                    )
                    
                    f1_score = self.metric_max_over_ground_truths(
                        self.compute_f1,
                        pred_answer,
                        ground_truths
                    )
                    
                    # Store results
                    all_results[model_name]['em'].append(em_score)
                    all_results[model_name]['f1'].append(f1_score)
                    all_results[model_name]['score'].append(pred_score)
                    
                    print(f"{model_name:<15} | Pred: {pred_answer:<20} | "
                          f"EM: {em_score:.2f} | F1: {f1_score:.4f} | "
                          f"Conf: {pred_score:.4f}")
                    
                except Exception as e:
                    print(f"{model_name:<15} | Error: {str(e)[:40]}")
        
        # Store aggregated results
        self.results = all_results
    
    def print_summary(self):
        """Print comprehensive summary of evaluation results."""
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY - AGGREGATED METRICS")
        print("=" * 80)
        
        if not self.results:
            print("No results to display.")
            return
        
        # Create results dataframe
        summary_data = []
        for model_name in sorted(self.results.keys()):
            metrics = self.results[model_name]
            
            if 'em' not in metrics or len(metrics['em']) == 0:
                continue
                
            summary_data.append({
                'Model': model_name,
                'EM (%)': np.mean(metrics['em']) * 100,
                'F1-Score': np.mean(metrics['f1']),
                'Confidence': np.mean(metrics['score']),
                'Samples': len(metrics['em'])
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values('F1-Score', ascending=False)
        
        print("\n" + df_summary.to_string(index=False))
        
        # Print detailed statistics
        print("\n" + "=" * 80)
        print("DETAILED STATISTICS")
        print("=" * 80)
        
        for model_name in sorted(self.results.keys()):
            metrics = self.results[model_name]
            if 'em' not in metrics or len(metrics['em']) == 0:
                continue
            
            print(f"\n{model_name}:")
            print(f"  Exact Match:  {np.mean(metrics['em']):.4f} "
                  f"(±{np.std(metrics['em']):.4f})")
            print(f"  F1-Score:     {np.mean(metrics['f1']):.4f} "
                  f"(±{np.std(metrics['f1']):.4f})")
            print(f"  Confidence:   {np.mean(metrics['score']):.4f} "
                  f"(±{np.std(metrics['score']):.4f})")
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get results as a pandas DataFrame for easy analysis.
        
        Returns:
            DataFrame with model performance metrics
        """
        data = []
        for model_name in sorted(self.results.keys()):
            metrics = self.results[model_name]
            if 'em' not in metrics or len(metrics['em']) == 0:
                continue
            
            data.append({
                'Model': model_name,
                'EM_Mean': np.mean(metrics['em']),
                'EM_Std': np.std(metrics['em']),
                'F1_Mean': np.mean(metrics['f1']),
                'F1_Std': np.std(metrics['f1']),
                'Confidence_Mean': np.mean(metrics['score']),
                'Confidence_Std': np.std(metrics['score']),
                'Num_Samples': len(metrics['em'])
            })
        
        return pd.DataFrame(data)
    
    def run_evaluation(self):
        """
        Run the complete evaluation pipeline.
        
        Returns:
            Dictionary with evaluation results
        """
        self.load_models()
        if not self.pipelines:
            print("Error: No models loaded successfully!")
            return None
        
        self.evaluate_models()
        self.print_summary()
        
        return self.results


# ============================================================================
# CELL 4: Run Evaluation
# ============================================================================

def main():
    """Main entry point for the QA model evaluation."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + "QA MODEL EVALUATION - SQuAD METRICS".center(78) + "║")
    print("║" + "(Exact Match & F1-Score)".center(78) + "║")
    print("╚" + "=" * 78 + "╝\n")
    
    # Initialize evaluator
    evaluator = QAModelEvaluator()
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    if results:
        # Get results as DataFrame
        df_results = evaluator.get_results_dataframe()
        print("\n" + "=" * 80)
        print("RESULTS DATAFRAME (for further analysis)")
        print("=" * 80)
        print(df_results.to_string(index=False))
        
        print("\n" + "=" * 80)
        print("✓ Evaluation completed successfully!")
        print("=" * 80 + "\n")
    
    return evaluator, results


# ============================================================================
# CELL 5: Visualization (Optional)
# ============================================================================

def plot_results(evaluator):
    """Create visualizations of evaluation results."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        df_results = evaluator.get_results_dataframe()
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('QA Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Exact Match Scores
        axes[0].barh(df_results['Model'], df_results['EM_Mean'] * 100, 
                     xerr=df_results['EM_Std'] * 100, capsize=5, color='skyblue')
        axes[0].set_xlabel('Exact Match (%)', fontweight='bold')
        axes[0].set_title('Exact Match Scores')
        axes[0].set_xlim(0, 100)
        
        # Plot 2: F1-Scores
        axes[1].barh(df_results['Model'], df_results['F1_Mean'], 
                     xerr=df_results['F1_Std'], capsize=5, color='lightgreen')
        axes[1].set_xlabel('F1-Score', fontweight='bold')
        axes[1].set_title('F1-Scores')
        axes[1].set_xlim(0, 1.0)
        
        # Plot 3: Confidence Scores
        axes[2].barh(df_results['Model'], df_results['Confidence_Mean'], 
                     xerr=df_results['Confidence_Std'], capsize=5, color='salmon')
        axes[2].set_xlabel('Confidence Score', fontweight='bold')
        axes[2].set_title('Model Confidence')
        axes[2].set_xlim(0, 1.0)
        
        plt.tight_layout()
        plt.show()
        
        print("✓ Visualization created successfully!")
        
    except ImportError:
        print("Note: Matplotlib/Seaborn not available for visualization")


# ============================================================================
# CELL 6: Execute Evaluation
# ============================================================================

if __name__ == "__main__":
    evaluator, results = main()
    
    # Optional: Create visualizations
    plot_results(evaluator)
