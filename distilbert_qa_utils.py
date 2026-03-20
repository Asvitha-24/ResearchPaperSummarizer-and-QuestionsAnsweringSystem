"""
Utility functions for DistilBERT QA model - loading, inference, and evaluation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import json
from pathlib import Path


class DistilBertQAModel:
    """Wrapper for DistilBERT QA model."""
    
    def __init__(self, model_path="distilbert-base-uncased", device=None):
        """Initialize model and tokenizer."""
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, question, context, top_k=1):
        """
        Predict answer to a question given context.
        
        Args:
            question: Question string
            context: Context/passage string
            top_k: Return top k predictions
            
        Returns:
            List of predictions with scores
        """
        inputs = self.tokenizer(
            question,
            context,
            return_tensors='pt',
            max_length=512,
            truncation=True
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        start_logits = outputs.start_logits[0].cpu()
        end_logits = outputs.end_logits[0].cpu()
        
        # Get all possible answers
        answers = []
        for start_idx in torch.argsort(start_logits, descending=True)[:top_k]:
            for end_idx in torch.argsort(end_logits, descending=True)[:top_k]:
                if end_idx >= start_idx:
                    score = (start_logits[start_idx] + end_logits[end_idx]).item()
                    answer_text = self.tokenizer.decode(
                        input_ids[0][start_idx:end_idx+1],
                        skip_special_tokens=True
                    )
                    
                    # Skip empty answers
                    if answer_text.strip():
                        answers.append({
                            'answer': answer_text,
                            'score': score,
                            'start': start_idx.item(),
                            'end': end_idx.item()
                        })
        
        # Sort by score and return top
        answers = sorted(answers, key=lambda x: x['score'], reverse=True)
        return answers[:top_k]
    
    def batch_predict(self, qa_pairs):
        """
        Predict answers for multiple QA pairs.
        
        Args:
            qa_pairs: List of {'question': str, 'context': str} dicts
            
        Returns:
            List of predictions
        """
        results = []
        for qa in qa_pairs:
            pred = self.predict(qa['question'], qa['context'])
            results.append({
                'question': qa['question'],
                'context': qa['context'],
                'predictions': pred
            })
        return results
    
    def save(self, output_path):
        """Save model and tokenizer."""
        Path(output_path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"Model saved to: {output_path}")
    
    @classmethod
    def from_finetuned(cls, checkpoint_dir):
        """Load a fine-tuned model."""
        return cls(model_path=checkpoint_dir)


def evaluate_qa_model(model, test_data, top_k=1):
    """
    Simple evaluation on test set.
    Computes exact match and F1 scores if reference answers available.
    """
    from collections import Counter
    import re
    
    def normalize_answer(s):
        """Normalize answer for evaluation."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
            return ''.join(ch if ch not in exclude else ' ' for ch in text)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def f1_score(prediction, ground_truth):
        """Calculate F1 score."""
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0
        
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    def exact_match(prediction, ground_truth):
        return float(normalize_answer(prediction) == normalize_answer(ground_truth))
    
    exact_matches = []
    f1_scores = []
    
    for item in test_data:
        predictions = model.predict(
            item['question'],
            item['context'],
            top_k=1
        )
        
        if predictions:
            pred_answer = predictions[0]['answer']
        else:
            pred_answer = ""
        
        if 'answers' in item:
            # If multiple reference answers, use best score
            reference_answers = item['answers']
            if isinstance(reference_answers, str):
                reference_answers = [reference_answers]
            
            em = max(exact_match(pred_answer, ref) for ref in reference_answers)
            f1 = max(f1_score(pred_answer, ref) for ref in reference_answers)
        else:
            em = 0
            f1 = 0
        
        exact_matches.append(em)
        f1_scores.append(f1)
    
    avg_em = sum(exact_matches) / len(exact_matches) if exact_matches else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    return {
        'exact_match': avg_em,
        'f1_score': avg_f1,
        'num_samples': len(test_data)
    }


def interactive_qa(model):
    """Interactive QA session."""
    print("\n" + "="*70)
    print("INTERACTIVE QA MODE")
    print("Type 'quit' to exit")
    print("="*70)
    
    while True:
        context = input("\nEnter context (or 'quit'): ").strip()
        if context.lower() == 'quit':
            break
        
        if not context:
            print("Context cannot be empty!")
            continue
        
        question = input("Enter question: ").strip()
        if not question:
            print("Question cannot be empty!")
            continue
        
        predictions = model.predict(question, context, top_k=3)
        
        print("\nPredictions:")
        for i, pred in enumerate(predictions, 1):
            print(f"  {i}. {pred['answer']} (score: {pred['score']:.4f})")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "./checkpoints/distilbert_qa_finetuned"
    
    # Load model
    model = DistilBertQAModel(model_path=model_path)
    
    # Example QA
    context = """
    Artificial Intelligence (AI) is the simulation of human intelligence 
    processes by machines, especially computer systems. These processes 
    include learning, reasoning, and self-correction.
    """
    
    question = "What is Artificial Intelligence?"
    
    print("\nExample QA:")
    print(f"Question: {question}")
    print(f"Context: {context}")
    
    predictions = model.predict(question, context, top_k=3)
    
    print("\nPredictions:")
    for i, pred in enumerate(predictions, 1):
        print(f"  {i}. {pred['answer']} (score: {pred['score']:.4f})")
    
    # Interactive mode
    response = input("\nEnter interactive mode? (y/n): ").strip().lower()
    if response == 'y':
        interactive_qa(model)
