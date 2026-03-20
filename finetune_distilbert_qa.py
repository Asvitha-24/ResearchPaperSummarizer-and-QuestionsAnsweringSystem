"""
Fine-tune DistilBERT for Question Answering task using HuggingFace Transformers.
Can work with SQuAD-style datasets or synthetic QA pairs.
"""

import os
import sys
import torch
import pandas as pd
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from datasets import Dataset, load_dataset
import random

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seed(42)

# Configuration
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 3
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./checkpoints/distilbert_qa_finetuned"
TRAIN_DATASET_SIZE = 5000  # Use SQuAD subset or generate synthetic data


def generate_synthetic_qa_pairs(csv_path: str, num_pairs: int = 5000):
    """
    Generate synthetic QA pairs from paper titles and summaries.
    This creates question-answer pairs where the answer comes from the text.
    """
    print(f"Generating {num_pairs} synthetic QA pairs...")
    
    try:
        df = pd.read_csv(csv_path, nrows=num_pairs * 2)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []
    
    qa_pairs = []
    
    # Question templates
    question_templates = [
        "What is the main topic of this paper?",
        "What does this paper discuss about {topic}?",
        "What are the key findings in this paper?",
        "What is described in the following text?",
        "According to the text, what is the focus?",
        "Can you explain what this paper is about?",
        "What information is provided about research?",
        "Summarize the main points mentioned:",
    ]
    
    for idx, row in df.iterrows():
        if len(qa_pairs) >= num_pairs:
            break
        
        try:
            context = str(row.get('summary', '')).strip()
            title = str(row.get('title', '')).strip()
            
            if not context or len(context) < 50:
                continue
            
            # Create QA pair
            question = random.choice(question_templates)
            if "{topic}" in question:
                # Extract first noun from title
                words = title.split()[:3]
                topic = " ".join(words) if words else "research"
                question = question.format(topic=topic)
            
            # Answer is from context
            answer_start = 0
            answer_text = context[:100] if len(context) > 100 else context
            
            qa_pairs.append({
                'question': question,
                'context': context,
                'answers': {
                    'text': [answer_text],
                    'answer_start': [answer_start]
                }
            })
            
        except Exception as e:
            continue
    
    print(f"Generated {len(qa_pairs)} QA pairs")
    return qa_pairs


def load_squad_dataset(dataset_name="squad", split="train", num_samples=5000):
    """Load SQuAD or similar dataset from HuggingFace."""
    print(f"Loading {dataset_name} dataset ({split} split)...")
    try:
        dataset = load_dataset(dataset_name, split=split[:5000])
        if num_samples and len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))
        print(f"Loaded {len(dataset)} samples")
        return dataset
    except Exception as e:
        print(f"Could not load {dataset_name}: {e}")
        print("Falling back to synthetic QA generation...")
        return None


def prepare_qa_dataset(use_squad=True, csv_path=None, num_samples=5000):
    """Prepare QA dataset for training."""
    
    # Try to load SQuAD first
    if use_squad:
        dataset = load_squad_dataset(num_samples=num_samples)
        if dataset:
            return dataset
    
    # Fallback: generate synthetic QA pairs
    if csv_path is None:
        csv_path = "data/raw/arXiv Scientific Research Papers Dataset.csv"
    
    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}")
        print("Creating minimal synthetic dataset...")
        qa_pairs = [
            {
                'question': f'What is the main topic?',
                'context': f'This is a research paper about machine learning and AI topics. Sample {i}.',
                'answers': {'text': ['machine learning and AI'], 'answer_start': [33]}
            }
            for i in range(100)
        ]
    else:
        qa_pairs = generate_synthetic_qa_pairs(csv_path, num_pairs=num_samples)
    
    if not qa_pairs:
        raise ValueError("Failed to load or generate QA dataset")
    
    # Split into train and validation
    n_train = int(len(qa_pairs) * 0.9)
    train_qa = qa_pairs[:n_train]
    val_qa = qa_pairs[n_train:]
    
    train_dataset = Dataset.from_dict({
        'question': [qa['question'] for qa in train_qa],
        'context': [qa['context'] for qa in train_qa],
        'answers': [qa['answers'] for qa in train_qa]
    })
    
    val_dataset = Dataset.from_dict({
        'question': [qa['question'] for qa in val_qa],
        'context': [qa['context'] for qa in val_qa],
        'answers': [qa['answers'] for qa in val_qa]
    })
    
    return train_dataset, val_dataset


def preprocess_qa_data(examples, tokenizer, max_length=384, doc_stride=128):
    """Preprocess QA examples for DistilBERT."""
    
    questions = [q.strip() for q in examples['question']]
    contexts = [c.strip() for c in examples['context']]
    answers = examples['answers']
    
    # Tokenize
    tokenized_examples = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )
    
    # Map answer positions
    sample_map = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    
    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        
        sequence_ids = tokenized_examples.sequence_ids(i)
        
        # Find start and end token positions
        start_pos = 0
        end_pos = 0
        for j, (offset_start, offset_end) in enumerate(offsets):
            if offset_start <= start_char < offset_end:
                start_pos = j
            if offset_start < end_char <= offset_end:
                end_pos = j
        
        tokenized_examples["start_positions"].append(start_pos)
        tokenized_examples["end_positions"].append(end_pos)
    
    return tokenized_examples


def fine_tune_distilbert(
    train_dataset=None,
    val_dataset=None,
    model_name=MODEL_NAME,
    output_dir=OUTPUT_DIR,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
    max_seq_length=MAX_SEQ_LENGTH,
    doc_stride=DOC_STRIDE,
):
    """Fine-tune DistilBERT for QA."""
    
    print("\n" + "="*60)
    print("STARTING DISTILBERT QA FINE-TUNING")
    print("="*60)
    
    # Load tokenizer and model
    print(f"\nLoading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    # Prepare datasets if not provided
    if train_dataset is None or val_dataset is None:
        print("\nPreparing QA dataset...")
        train_dataset, val_dataset = prepare_qa_dataset(
            use_squad=False,  # Set to True to use SQuAD
            num_samples=TRAIN_DATASET_SIZE
        )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Preprocess datasets
    print("\nPreprocessing datasets...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_qa_data(
            x, tokenizer, max_seq_length, doc_stride
        ),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Processing train dataset"
    )
    
    val_dataset = val_dataset.map(
        lambda x: preprocess_qa_data(
            x, tokenizer, max_seq_length, doc_stride
        ),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Processing val dataset"
    )
    
    # Set format for PyTorch
    train_dataset.set_format(type='torch')
    val_dataset.set_format(type='torch')
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        logging_steps=100,
        warmup_steps=500,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save model and tokenizer
    print(f"\nSaving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n" + "="*60)
    print("FINE-TUNING COMPLETE!")
    print("="*60)
    print(f"Model saved to: {output_dir}")
    
    return model, tokenizer


def predict_qa(model, tokenizer, question, context):
    """
    Use fine-tuned model to predict answer to a question.
    """
    inputs = tokenizer.encode_plus(
        question,
        context,
        return_tensors='pt',
        max_length=512,
        truncation=True
    )
    
    output = model(**inputs)
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits) + 1
    
    answer = tokenizer.decode(
        inputs['input_ids'][0][answer_start:answer_end]
    )
    
    return answer


if __name__ == "__main__":
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Fine-tune model
    model, tokenizer = fine_tune_distilbert()
    
    # Test on sample QA
    print("\n" + "="*60)
    print("TESTING MODEL ON SAMPLE QUESTIONS")
    print("="*60)
    
    sample_context = """
    Machine learning is a subset of artificial intelligence that enables 
    systems to learn and improve from experience. Deep learning is a 
    specialized field of machine learning using neural networks with 
    multiple layers.
    """
    
    sample_questions = [
        "What is machine learning?",
        "What is deep learning?",
    ]
    
    model.eval()
    with torch.no_grad():
        for question in sample_questions:
            answer = predict_qa(model, tokenizer, question, sample_context)
            print(f"\nQ: {question}")
            print(f"A: {answer}")
    
    print("\n" + "="*60)
    print("Fine-tuning script completed!")
    print("="*60)
