"""
Question Answering Model Evaluation using SQuAD Metrics
========================================================
This script evaluates multiple pre-trained QA models from Hugging Face using
the SQuAD metric (Exact Match and F1-Score).

Models evaluated:
1. distilbert-base-uncased-distilled-squad (DistilBERT)
2. bert-base-uncased (BERT)
3. roberta-base (RoBERTa)
4. albert-base-v2 (ALBERT)
"""

import warnings
import json
from typing import Dict, List, Tuple
from collections import defaultdict

from transformers import pipeline
from datasets import load_metric
import numpy as np
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class QAModelEvaluator:
    """Evaluates QA models using SQuAD metrics."""
    
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
        
    def load_models(self):
        """Load all QA models using the pipeline."""
        print("=" * 80)
        print("LOADING QUESTION ANSWERING MODELS")
        print("=" * 80)
        
        for model_name, model_id in self.models.items():
            print(f"\nLoading {model_name} ({model_id})...", end=" ")
            try:
                qa_pipeline = pipeline(
                    'question-answering',
                    model=model_id,
                    device=0 if self._cuda_available() else -1
                )
                self.pipelines[model_name] = qa_pipeline
                print("✓ Loaded successfully")
            except Exception as e:
                print(f"✗ Error: {e}")
                
    @staticmethod
    def _cuda_available() -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def prepare_samples(self) -> Tuple[str, str, List[str]]:
        """
        Prepare sample data for evaluation.
        
        Returns:
            Tuple of (context, question, ground_truth_answers)
        """
        # Sample context and question pair
        context = (
            "Machine learning is a subset of artificial intelligence (AI) that focuses on "
            "enabling computers to learn from and make decisions based on data, without being "
            "explicitly programmed. Deep learning, a subset of machine learning, uses neural "
            "networks with multiple layers to learn representations of data. The field has seen "
            "remarkable progress in recent years, with applications ranging from natural language "
            "processing to computer vision."
        )
        
        question = "What is machine learning a subset of?"
        
        # Ground truth answers (multiple valid answers)
        ground_truth_answers = [
            {"text": "artificial intelligence", "answer_start": 45},
            {"text": "artificial intelligence (AI)", "answer_start": 45},
            {"text": "AI", "answer_start": 56}
        ]
        
        return context, question, ground_truth_answers
    
    def normalize_answer(self, answer: str) -> str:
        """
        Normalize answer text for evaluation.
        
        Args:
            answer: Raw answer text
            
        Returns:
            Normalized answer
        """
        import re
        
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
        Compute Exact Match score.
        
        Args:
            prediction: Model's predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            EM score (0 or 1)
        """
        return float(
            self.normalize_answer(prediction) == self.normalize_answer(ground_truth)
        )
    
    def compute_f1(self, prediction: str, ground_truth: str) -> float:
        """
        Compute F1-Score.
        
        Args:
            prediction: Model's predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            F1 score
        """
        pred_tokens = self.normalize_answer(prediction).split()
        truth_tokens = self.normalize_answer(ground_truth).split()
        
        common = set(pred_tokens) & set(truth_tokens)
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return float(pred_tokens == truth_tokens)
        
        precision = len(common) / len(pred_tokens) if len(pred_tokens) > 0 else 0
        recall = len(common) / len(truth_tokens) if len(truth_tokens) > 0 else 0
        
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
        
        context, question, ground_truth_answers = self.prepare_samples()
        
        print(f"\nContext: {context[:100]}...")
        print(f"Question: {question}\n")
        print("Ground Truth Answers:")
        for gt in ground_truth_answers:
            print(f"  - {gt['text']}")
        
        print("\n" + "-" * 80)
        
        for model_name, pipeline_model in self.pipelines.items():
            print(f"\n{model_name.upper()}")
            print("-" * 40)
            
            try:
                # Get prediction from model
                prediction = pipeline_model(
                    question=question,
                    context=context,
                    max_answer_len=20
                )
                
                pred_answer = prediction['answer']
                pred_score = prediction['score']
                
                print(f"Predicted Answer: {pred_answer}")
                print(f"Confidence Score: {pred_score:.4f}")
                
                # Compute metrics
                em_score = self.metric_max_over_ground_truths(
                    self.compute_exact_match,
                    pred_answer,
                    ground_truth_answers
                )
                
                f1_score = self.metric_max_over_ground_truths(
                    self.compute_f1,
                    pred_answer,
                    ground_truth_answers
                )
                
                # Store results
                self.results[model_name] = {
                    'prediction': pred_answer,
                    'confidence': pred_score,
                    'exact_match': em_score,
                    'f1_score': f1_score,
                    'start': prediction['start'],
                    'end': prediction['end']
                }
                
                print(f"Exact Match (EM): {em_score:.4f}")
                print(f"F1-Score: {f1_score:.4f}")
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                self.results[model_name] = {
                    'error': str(e)
                }
    
    def print_summary(self):
        """Print summary of evaluation results."""
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        if not self.results:
            print("No results to display.")
            return
        
        # Create summary table
        print(f"\n{'Model':<15} {'Prediction':<30} {'EM':<8} {'F1':<8} {'Confidence':<12}")
        print("-" * 75)
        
        for model_name in sorted(self.results.keys()):
            result = self.results[model_name]
            
            if 'error' in result:
                print(f"{model_name:<15} {'[ERROR]':<30} {'-':<8} {'-':<8} {'-':<12}")
            else:
                pred = result['prediction'][:25] + "..." if len(result['prediction']) > 25 \
                    else result['prediction']
                em = result['exact_match']
                f1 = result['f1_score']
                conf = result['confidence']
                
                print(f"{model_name:<15} {pred:<30} {em:<8.4f} {f1:<8.4f} {conf:<12.4f}")
        
        # Compute averages (excluding errors)
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if valid_results:
            avg_em = np.mean([v['exact_match'] for v in valid_results.values()])
            avg_f1 = np.mean([v['f1_score'] for v in valid_results.values()])
            avg_conf = np.mean([v['confidence'] for v in valid_results.values()])
            
            print("-" * 75)
            print(f"{'AVERAGE':<15} {'':<30} {avg_em:<8.4f} {avg_f1:<8.4f} {avg_conf:<12.4f}")
        
        # Performance ranking
        print("\n" + "=" * 80)
        print("PERFORMANCE RANKING (by F1-Score)")
        print("=" * 80)
        
        ranked = sorted(
            valid_results.items(),
            key=lambda x: x[1]['f1_score'],
            reverse=True
        )
        
        for rank, (model_name, result) in enumerate(ranked, 1):
            print(f"{rank}. {model_name}: F1={result['f1_score']:.4f}, "
                  f"EM={result['exact_match']:.4f}, "
                  f"Confidence={result['confidence']:.4f}")
    
    def save_results(self, filename: str = 'qa_evaluation_results.json'):
        """
        Save evaluation results to JSON file.
        
        Args:
            filename: Output filename
        """
        output_path = f"c:\\Users\\user\\Desktop\\Researcher\\backend\\{filename}"
        
        # Convert numpy types to Python types for JSON serialization
        results_serializable = {}
        for model, data in self.results.items():
            results_serializable[model] = {
                k: float(v) if isinstance(v, np.floating) else v
                for k, v in data.items()
            }
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    def run_evaluation(self, save_results_flag: bool = True):
        """
        Run the complete evaluation pipeline.
        
        Args:
            save_results_flag: Whether to save results to JSON file
        """
        self.load_models()
        self.evaluate_models()
        self.print_summary()
        
        if save_results_flag:
            self.save_results()
        
        return self.results


def main():
    """Main entry point for the evaluation script."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + "QUESTION ANSWERING MODEL EVALUATION - SQuAD METRICS".center(78) + "║")
    print("╚" + "=" * 78 + "╝")
    
    evaluator = QAModelEvaluator()
    results = evaluator.run_evaluation(save_results_flag=True)
    
    print("\n" + "=" * 80)
    print("Evaluation completed successfully!")
    print("=" * 80 + "\n")
    
    return results


if __name__ == "__main__":
    main()
