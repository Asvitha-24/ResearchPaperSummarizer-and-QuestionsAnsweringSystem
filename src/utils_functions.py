"""
Utility functions for evaluation metrics and analysis.
"""

import numpy as np
from typing import List, Dict, Tuple
from rouge_score import rouge_scorer
import torch
from bert_score import score as bert_score


class EvaluationMetrics:
    """Compute various NLP evaluation metrics."""
    
    @staticmethod
    def rouge_scores(reference: str, hypothesis: str, rouge_types: List[str] = ['rouge1', 'rouge2', 'rougeL']) -> Dict[str, float]:
        """
        Calculate ROUGE scores for summarization.
        
        Args:
            reference: Reference text (gold standard)
            hypothesis: Hypothesis text (generated summary)
            rouge_types: Types of ROUGE scores to calculate
            
        Returns:
            Dictionary of ROUGE scores
        """
        scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        
        result = {}
        for rouge_type in rouge_types:
            result[rouge_type] = {
                'precision': scores[rouge_type].precision,
                'recall': scores[rouge_type].recall,
                'fmeasure': scores[rouge_type].fmeasure
            }
        
        return result
    
    @staticmethod
    def batch_rouge_scores(references: List[str], 
                          hypotheses: List[str],
                          rouge_types: List[str] = ['rouge1', 'rouge2', 'rougeL']) -> Dict[str, Dict[str, float]]:
        """
        Calculate ROUGE scores for multiple pairs.
        
        Args:
            references: List of reference texts
            hypotheses: List of hypothesis texts
            rouge_types: Types of ROUGE scores
            
        Returns:
            Dictionary of averaged ROUGE scores
        """
        all_scores = {rt: {'precision': [], 'recall': [], 'fmeasure': []} for rt in rouge_types}
        
        for ref, hyp in zip(references, hypotheses):
            scores = EvaluationMetrics.rouge_scores(ref, hyp, rouge_types)
            for rouge_type in rouge_types:
                all_scores[rouge_type]['precision'].append(scores[rouge_type]['precision'])
                all_scores[rouge_type]['recall'].append(scores[rouge_type]['recall'])
                all_scores[rouge_type]['fmeasure'].append(scores[rouge_type]['fmeasure'])
        
        # Average the scores
        averaged_scores = {}
        for rouge_type in rouge_types:
            averaged_scores[rouge_type] = {
                'precision': np.mean(all_scores[rouge_type]['precision']),
                'recall': np.mean(all_scores[rouge_type]['recall']),
                'fmeasure': np.mean(all_scores[rouge_type]['fmeasure'])
            }
        
        return averaged_scores
    
    @staticmethod
    def bert_similarity(references: List[str], 
                       hypotheses: List[str],
                       model_type: str = "distilbert-base-uncased") -> Dict[str, float]:
        """
        Calculate BERTScore for evaluating summaries.
        
        Args:
            references: List of reference texts
            hypotheses: List of hypothesis texts
            model_type: BERT model to use
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        P, R, F1 = bert_score(
            hypotheses,
            references,
            model_type=model_type,
            device=device,
            batch_size=8,
            num_layers=8
        )
        
        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item()
        }
    
    @staticmethod
    def bleu_score(reference: List[str], hypothesis: List[str]) -> float:
        """
        Calculate BLEU score.
        
        Args:
            reference: Reference tokens
            hypothesis: Hypothesis tokens
            
        Returns:
            BLEU score (0-1)
        """
        from nltk.translate.bleu_score import sentence_bleu
        
        return sentence_bleu([reference], hypothesis)
    
    @staticmethod
    def exact_match(reference: str, hypothesis: str) -> bool:
        """
        Check exact match between reference and hypothesis.
        
        Args:
            reference: Reference text
            hypothesis: Hypothesis text
            
        Returns:
            True if texts match exactly
        """
        return reference.strip().lower() == hypothesis.strip().lower()
    
    @staticmethod
    def f1_score(prediction: str, ground_truth: str) -> float:
        """
        Calculate token-level F1 score (useful for QA).
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            F1 score (0-1)
        """
        pred_tokens = set(prediction.lower().split())
        truth_tokens = set(ground_truth.lower().split())
        
        common = len(pred_tokens & truth_tokens)
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)
        
        precision = common / len(pred_tokens) if len(pred_tokens) > 0 else 0
        recall = common / len(truth_tokens) if len(truth_tokens) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1


class RetrievalMetrics:
    """Evaluate information retrieval systems."""
    
    @staticmethod
    def precision_at_k(relevant_docs: List[int], retrieved_docs: List[int], k: int = 5) -> float:
        """
        Calculate Precision@K.
        
        Args:
            relevant_docs: List of relevant document indices
            retrieved_docs: List of retrieved document indices
            k: Cutoff for top-k
            
        Returns:
            Precision@K score
        """
        retrieved_k = retrieved_docs[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant_docs))
        return relevant_retrieved / k if k > 0 else 0.0
    
    @staticmethod
    def recall_at_k(relevant_docs: List[int], retrieved_docs: List[int], k: int = 5) -> float:
        """
        Calculate Recall@K.
        
        Args:
            relevant_docs: List of relevant document indices
            retrieved_docs: List of retrieved document indices
            k: Cutoff for top-k
            
        Returns:
            Recall@K score
        """
        retrieved_k = retrieved_docs[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant_docs))
        return relevant_retrieved / len(relevant_docs) if len(relevant_docs) > 0 else 0.0
    
    @staticmethod
    def mean_reciprocal_rank(relevant_docs: List[int], retrieved_docs: List[int]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            relevant_docs: List of relevant document indices
            retrieved_docs: List of retrieved document indices
            
        Returns:
            MRR score
        """
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def ndcg_at_k(relevant_docs: List[int], retrieved_docs: List[int], k: int = 5) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).
        
        Args:
            relevant_docs: List of relevant document indices
            retrieved_docs: List of retrieved document indices
            k: Cutoff for top-k
            
        Returns:
            NDCG@K score
        """
        # DCG calculation
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k]):
            if doc in relevant_docs:
                dcg += 1.0 / np.log2(i + 2)
        
        # IDCG calculation (ideal ranking)
        idcg = 0.0
        for i in range(min(len(relevant_docs), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def average_precision(relevant_docs: List[int], retrieved_docs: List[int]) -> float:
        """
        Calculate Average Precision (AP).
        
        Args:
            relevant_docs: List of relevant document indices
            retrieved_docs: List of retrieved document indices
            
        Returns:
            AP score
        """
        score = 0.0
        num_hits = 0.0
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        return score / len(relevant_docs) if len(relevant_docs) > 0 else 0.0


class AnswerQualityMetrics:
    """Metrics for evaluating QA system answers."""
    
    @staticmethod
    def confidence_distribution(scores: List[float]) -> Dict[str, float]:
        """
        Analyze confidence score distribution.
        
        Args:
            scores: List of confidence scores
            
        Returns:
            Statistics of score distribution
        """
        scores_array = np.array(scores)
        
        return {
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'median': float(np.median(scores_array)),
            'q25': float(np.percentile(scores_array, 25)),
            'q75': float(np.percentile(scores_array, 75))
        }
    
    @staticmethod
    def answer_length_stats(answers: List[str]) -> Dict[str, float]:
        """
        Analyze answer length statistics.
        
        Args:
            answers: List of answers
            
        Returns:
            Length statistics (tokens and characters)
        """
        token_lengths = [len(ans.split()) for ans in answers]
        char_lengths = [len(ans) for ans in answers]
        
        return {
            'avg_tokens': np.mean(token_lengths),
            'std_tokens': np.std(token_lengths),
            'min_tokens': np.min(token_lengths),
            'max_tokens': np.max(token_lengths),
            'avg_chars': np.mean(char_lengths),
            'std_chars': np.std(char_lengths),
            'min_chars': np.min(char_lengths),
            'max_chars': np.max(char_lengths)
        }


class MetricsReporter:
    """Generate comprehensive evaluation reports."""
    
    @staticmethod
    def summarization_report(references: List[str], 
                            hypotheses: List[str],
                            detailed: bool = True) -> Dict:
        """
        Generate comprehensive summarization evaluation report.
        
        Args:
            references: Reference summaries
            hypotheses: Generated summaries
            detailed: Include detailed metrics
            
        Returns:
            Dictionary with all evaluation metrics
        """
        print("Computing ROUGE scores...")
        rouge_scores = EvaluationMetrics.batch_rouge_scores(references, hypotheses)
        
        report = {
            'rouge': rouge_scores,
            'sample_count': len(references)
        }
        
        if detailed and len(references) <= 100:
            print("Computing BERTScore...")
            try:
                bert_scores = EvaluationMetrics.bert_similarity(references, hypotheses)
                report['bert_score'] = bert_scores
            except Exception as e:
                print(f"BERTScore computation skipped: {e}")
        
        return report
    
    @staticmethod
    def qa_report(predictions: List[str],
                 ground_truths: List[str],
                 confidence_scores: List[float] = None) -> Dict:
        """
        Generate comprehensive QA evaluation report.
        
        Args:
            predictions: Predicted answers
            ground_truths: Ground truth answers
            confidence_scores: Confidence scores (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        f1_scores = [EvaluationMetrics.f1_score(pred, truth) 
                     for pred, truth in zip(predictions, ground_truths)]
        exact_matches = [EvaluationMetrics.exact_match(pred, truth)
                        for pred, truth in zip(predictions, ground_truths)]
        
        report = {
            'f1_score': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'exact_match_rate': np.mean(exact_matches),
            'total_samples': len(predictions)
        }
        
        if confidence_scores:
            report['confidence_stats'] = AnswerQualityMetrics.confidence_distribution(confidence_scores)
        
        report['answer_length_stats'] = AnswerQualityMetrics.answer_length_stats(predictions)
        
        return report
