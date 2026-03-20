"""
Configuration settings for BART fine-tuning.
Modify these values to adjust training parameters.
"""

# Model Configuration
MODEL_CONFIG = {
    'model_name': 'facebook/bart-large-cnn',
    'max_input_length': 1024,
    'max_target_length': 256,
    'max_position_embeddings': 1024,
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 8,  # Reduce if GPU runs out of memory
    'learning_rate': 5e-5,
    'num_epochs': 3,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
}

# Data Configuration
DATA_CONFIG = {
    'data_path': 'data/raw/arXiv Scientific Research Papers Dataset.csv',
    'output_dir': './checkpoints/bart_finetuned',
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'sample_size': 5000,  # Set to None to use entire dataset
    'seed': 42,
}

# Generation Configuration
GENERATION_CONFIG = {
    'max_length': 150,
    'min_length': 50,
    'num_beams': 4,
    'length_penalty': 2.0,
    'early_stopping': True,
    'no_repeat_ngram_size': 3,
}

# Evaluation Configuration
EVAL_CONFIG = {
    'compute_rouge': True,
    'compute_bertscore': False,  # Slow, disable if needed
    'rouge_types': ['rouge1', 'rouge2', 'rougeL'],
}

# Hardware Configuration
HARDWARE_CONFIG = {
    'use_cuda': True,
    'use_amp': True,  # Automatic Mixed Precision
    'mixed_precision': 'fp16',  # 'no', 'fp16', 'bf16'
}

# Logging Configuration
LOGGING_CONFIG = {
    'logging_steps': 100,
    'save_steps': 500,
    'eval_steps': 500,
    'save_total_limit': 3,
}


def get_training_args_dict():
    """Get all training arguments as a single dictionary."""
    return {
        **MODEL_CONFIG,
        **TRAINING_CONFIG,
        **DATA_CONFIG,
        **GENERATION_CONFIG,
        **EVAL_CONFIG,
        **HARDWARE_CONFIG,
        **LOGGING_CONFIG,
    }


if __name__ == "__main__":
    import json
    print("Current Configuration:")
    print(json.dumps(get_training_args_dict(), indent=2))
