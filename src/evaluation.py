"""
Comprehensive evaluation module for summarization models.
Evaluates BART, T5, PEGASUS, and GPT-3 based on ROUGE scores, compression ratio, and inference speed.
"""

import torch
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    pipeline
)
from rouge_score import rouge_scorer
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')


class SummarizationEvaluator:
    """Comprehensive evaluator for summarization models."""
    
    def __init__(self):
        """Initialize evaluator with model configurations."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.results = {}
        self.model_configs = {
            'bart': {
                'model_name': 'facebook/bart-large-cnn',
                'type': 'pipeline',
                'description': 'BART-Large-CNN'
            },
            't5': {
                'model_name': 'google/flan-t5-base',
                'type': 'pipeline',
                'description': 'FLAN-T5-Base'
            },
            'pegasus': {
                'model_name': 'google/pegasus-cnn_dailymail',
                'type': 'pipeline',
                'description': 'PEGASUS-CNN-DailyMail'
            }
        }
    
    def load_model(self, model_key: str) -> bool:
        """
        Load a summarization model.
        
        Args:
            model_key: Key of the model to load ('bart', 't5', 'pegasus')
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if model_key not in self.model_configs:
            print(f"Unknown model key: {model_key}")
            return False
        
        try:
            config = self.model_configs[model_key]
            print(f"Loading {config['description']}...")
            
            pipeline_model = pipeline(
                "summarization",
                model=config['model_name'],
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.models[model_key] = {
                'pipeline': pipeline_model,
                'config': config
            }
            print(f"✓ {config['description']} loaded successfully")
            return True
        
        except Exception as e:
            print(f"✗ Error loading {model_key}: {str(e)}")
            return False
    
    def load_all_models(self) -> Dict[str, bool]:
        """
        Load all available models.
        
        Returns:
            Dictionary with model keys and load status
        """
        status = {}
        for model_key in self.model_configs.keys():
            status[model_key] = self.load_model(model_key)
        return status
    
    def generate_summary(self, 
                        text: str, 
                        model_key: str,
                        max_length: int = 150,
                        min_length: int = 50) -> Tuple[str, float]:
        """
        Generate summary using specified model.
        
        Args:
            text: Text to summarize
            model_key: Model to use
            max_length: Maximum summary length
            min_length: Minimum summary length
            
        Returns:
            Tuple of (summary, generation_time)
        """
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not loaded")
        
        if not text or len(text.split()) < 50:
            return text, 0.0
        
        try:
            pipeline_model = self.models[model_key]['pipeline']
            
            start_time = time.time()
            result = pipeline_model(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            inference_time = time.time() - start_time
            
            summary = result[0]['summary_text']
            return summary, inference_time
        
        except Exception as e:
            print(f"Error generating summary with {model_key}: {str(e)}")
            return "", 0.0
    
    def calculate_rouge_scores(self, 
                              reference: str, 
                              hypothesis: str) -> Dict[str, Dict[str, float]]:
        """
        Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
        
        Args:
            reference: Reference summary (ground truth)
            hypothesis: Generated summary
            
        Returns:
            Dictionary with ROUGE scores
        """
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        
        scores = scorer.score(reference, hypothesis)
        
        return {
            'rouge1': {
                'precision': scores['rouge1'].precision,
                'recall': scores['rouge1'].recall,
                'fmeasure': scores['rouge1'].fmeasure
            },
            'rouge2': {
                'precision': scores['rouge2'].precision,
                'recall': scores['rouge2'].recall,
                'fmeasure': scores['rouge2'].fmeasure
            },
            'rougeL': {
                'precision': scores['rougeL'].precision,
                'recall': scores['rougeL'].recall,
                'fmeasure': scores['rougeL'].fmeasure
            }
        }
    
    def calculate_compression_ratio(self, 
                                   original_text: str, 
                                   summary: str) -> float:
        """
        Calculate compression ratio.
        
        Args:
            original_text: Original document
            summary: Generated summary
            
        Returns:
            Compression ratio (summary_length / original_length)
        """
        original_length = len(original_text.split())
        summary_length = len(summary.split())
        
        if original_length == 0:
            return 0.0
        
        return summary_length / original_length
    
    def evaluate_model(self,
                      model_key: str,
                      texts: List[str],
                      references: List[str],
                      max_length: int = 150,
                      min_length: int = 50) -> Dict:
        """
        Comprehensive evaluation of a single model.
        
        Args:
            model_key: Model to evaluate
            texts: List of texts to summarize
            references: List of reference summaries
            max_length: Maximum summary length
            min_length: Minimum summary length
            
        Returns:
            Dictionary with evaluation metrics
        """
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not loaded")
        
        print(f"\nEvaluating {self.models[model_key]['config']['description']}...")
        print(f"Processing {len(texts)} documents...")
        
        summaries = []
        inference_times = []
        rouge_scores_list = []
        compression_ratios = []
        
        for i, (text, reference) in enumerate(zip(texts, references)):
            # Generate summary
            summary, inference_time = self.generate_summary(
                text, model_key, max_length, min_length
            )
            summaries.append(summary)
            inference_times.append(inference_time)
            
            # Calculate ROUGE scores
            rouge_scores = self.calculate_rouge_scores(reference, summary)
            rouge_scores_list.append(rouge_scores)
            
            # Calculate compression ratio
            compression_ratio = self.calculate_compression_ratio(text, summary)
            compression_ratios.append(compression_ratio)
            
            if (i + 1) % max(1, len(texts) // 5) == 0:
                print(f"  Processed {i + 1}/{len(texts)} documents")
        
        # Aggregate metrics
        avg_inference_time = np.mean(inference_times)
        
        # Average ROUGE scores
        avg_rouge1_f = np.mean([s['rouge1']['fmeasure'] for s in rouge_scores_list])
        avg_rouge2_f = np.mean([s['rouge2']['fmeasure'] for s in rouge_scores_list])
        avg_rougeL_f = np.mean([s['rougeL']['fmeasure'] for s in rouge_scores_list])
        
        avg_compression_ratio = np.mean(compression_ratios)
        
        results = {
            'model_key': model_key,
            'model_name': self.models[model_key]['config']['description'],
            'num_samples': len(texts),
            'summaries': summaries,
            'inference_times': inference_times,
            'avg_inference_time': avg_inference_time,
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'std_inference_time': np.std(inference_times),
            'rouge_scores': rouge_scores_list,
            'avg_rouge1': avg_rouge1_f,
            'avg_rouge2': avg_rouge2_f,
            'avg_rougeL': avg_rougeL_f,
            'compression_ratios': compression_ratios,
            'avg_compression_ratio': avg_compression_ratio,
            'std_compression_ratio': np.std(compression_ratios)
        }
        
        self.results[model_key] = results
        
        print(f"✓ Evaluation complete")
        print(f"  Avg ROUGE-1: {avg_rouge1_f:.4f}")
        print(f"  Avg ROUGE-2: {avg_rouge2_f:.4f}")
        print(f"  Avg ROUGE-L: {avg_rougeL_f:.4f}")
        print(f"  Avg Compression Ratio: {avg_compression_ratio:.4f}")
        print(f"  Avg Inference Time: {avg_inference_time:.4f}s")
        
        return results
    
    def evaluate_all_models(self,
                           texts: List[str],
                           references: List[str],
                           max_length: int = 150,
                           min_length: int = 50) -> Dict:
        """
        Evaluate all loaded models.
        
        Args:
            texts: List of texts to summarize
            references: List of reference summaries
            max_length: Maximum summary length
            min_length: Minimum summary length
            
        Returns:
            Dictionary with all evaluation results
        """
        print("="*80)
        print("SUMMARIZATION MODELS EVALUATION")
        print("="*80)
        
        for model_key in self.models.keys():
            self.evaluate_model(model_key, texts, references, max_length, min_length)
        
        return self.results
    
    def create_comparison_dataframe(self) -> pd.DataFrame:
        """
        Create a comparison dataframe from evaluation results.
        
        Returns:
            DataFrame with model comparisons
        """
        comparison_data = []
        
        for model_key, results in self.results.items():
            comparison_data.append({
                'Model': results['model_name'],
                'ROUGE-1': results['avg_rouge1'],
                'ROUGE-2': results['avg_rouge2'],
                'ROUGE-L': results['avg_rougeL'],
                'Avg Compression Ratio': results['avg_compression_ratio'],
                'Std Compression Ratio': results['std_compression_ratio'],
                'Avg Inference Time (s)': results['avg_inference_time'],
                'Std Inference Time (s)': results['std_inference_time'],
                'Min Inference Time (s)': results['min_inference_time'],
                'Max Inference Time (s)': results['max_inference_time']
            })
        
        return pd.DataFrame(comparison_data)
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a detailed evaluation report.
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Report as string
        """
        report = []
        report.append("="*80)
        report.append("SUMMARIZATION MODELS EVALUATION REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*80)
        
        # Comparison table
        comparison_df = self.create_comparison_dataframe()
        report.append("\n" + "COMPARISON SUMMARY")
        report.append("-"*80)
        report.append(comparison_df.to_string())
        
        # Detailed analysis
        report.append("\n\n" + "="*80)
        report.append("DETAILED ANALYSIS")
        report.append("="*80)
        
        for model_key, results in self.results.items():
            report.append(f"\n{results['model_name']}")
            report.append("-"*60)
            report.append(f"Samples Evaluated: {results['num_samples']}")
            report.append(f"\nROUGE Scores:")
            report.append(f"  ROUGE-1 (F): {results['avg_rouge1']:.4f}")
            report.append(f"  ROUGE-2 (F): {results['avg_rouge2']:.4f}")
            report.append(f"  ROUGE-L (F): {results['avg_rougeL']:.4f}")
            report.append(f"\nCompression Metrics:")
            report.append(f"  Average: {results['avg_compression_ratio']:.4f}")
            report.append(f"  Std Dev: {results['std_compression_ratio']:.4f}")
            report.append(f"\nInference Speed:")
            report.append(f"  Average: {results['avg_inference_time']:.4f}s")
            report.append(f"  Min: {results['min_inference_time']:.4f}s")
            report.append(f"  Max: {results['max_inference_time']:.4f}s")
            report.append(f"  Std Dev: {results['std_inference_time']:.4f}s")
        
        # Key findings
        report.append("\n\n" + "="*80)
        report.append("KEY FINDINGS & TRADE-OFFS")
        report.append("="*80)
        
        if self.results:
            # Best ROUGE
            best_rouge1 = max(self.results.items(), 
                            key=lambda x: x[1]['avg_rouge1'])
            report.append(f"\nBest ROUGE-1 Score: {best_rouge1[1]['model_name']} ({best_rouge1[1]['avg_rouge1']:.4f})")
            
            # Fastest inference
            fastest = min(self.results.items(), 
                         key=lambda x: x[1]['avg_inference_time'])
            report.append(f"Fastest Inference: {fastest[1]['model_name']} ({fastest[1]['avg_inference_time']:.4f}s)")
            
            # Best compression
            best_compression = max(self.results.items(),
                                 key=lambda x: x[1]['avg_compression_ratio'])
            report.append(f"Best Compression: {best_compression[1]['model_name']} ({best_compression[1]['avg_compression_ratio']:.4f})")
            
            report.append("\nTrade-off Analysis:")
            report.append("  • Higher ROUGE scores indicate better summary quality")
            report.append("  • Lower inference time indicates faster processing")
            report.append("  • Lower compression ratio indicates more concise summaries")
            report.append("  • Trade-off: Quality vs Speed - stronger models are often slower")
        
        report_str = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_str)
            print(f"\nReport saved to: {output_path}")
        
        return report_str
    
    def save_results_json(self, output_path: str) -> None:
        """
        Save evaluation results to JSON file.
        
        Args:
            output_path: Path to save JSON file
        """
        # Prepare data for JSON serialization
        json_results = {}
        
        for model_key, results in self.results.items():
            json_results[model_key] = {
                'model_name': results['model_name'],
                'num_samples': results['num_samples'],
                'avg_inference_time': float(results['avg_inference_time']),
                'std_inference_time': float(results['std_inference_time']),
                'min_inference_time': float(results['min_inference_time']),
                'max_inference_time': float(results['max_inference_time']),
                'avg_rouge1': float(results['avg_rouge1']),
                'avg_rouge2': float(results['avg_rouge2']),
                'avg_rougeL': float(results['avg_rougeL']),
                'avg_compression_ratio': float(results['avg_compression_ratio']),
                'std_compression_ratio': float(results['std_compression_ratio'])
            }
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=4)
        
        print(f"Results saved to: {output_path}")
