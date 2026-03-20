"""
paraphrase-distilroberta-base-v1 Checkpoint Preparation
Prepares the model checkpoint for use in semantic search
"""

from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
import os

print("\n" + "=" * 80)
print("PREPARING paraphrase-distilroberta-base-v1 CHECKPOINT")
print("=" * 80)

# Configuration
MODEL_NAME = 'paraphrase-distilroberta-base-v1'
CHECKPOINT_DIR = 'checkpoints/distilroberta_paraphrase_finetuned'

Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

print(f"\nLoading pre-trained model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
print(f"✓ Model loaded successfully")
print(f"  Architecture: {type(model).__name__}")
print(f"  Max sequence length: {model.max_seq_length}")

# Save the model
print(f"\nSaving model to: {CHECKPOINT_DIR}")
model.save(CHECKPOINT_DIR)
print(f"✓ Model saved")

# Create metadata file
metadata = {
    "model_name": MODEL_NAME,
    "checkpoint_type": "paraphrase",
    "use_case": "semantic_search",
    "description": "Distilled RoBERTa-based model for paraphrase detection and semantic search",
    "max_seq_length": model.max_seq_length,
    "embedding_dimension": model.get_sentence_embedding_dimension(),
}

metadata_path = os.path.join(CHECKPOINT_DIR, "checkpoint_metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Metadata saved to: {metadata_path}")

# Verify checkpoint
print(f"\nVerifying checkpoint...")
print(f"  Files in checkpoint: {os.listdir(CHECKPOINT_DIR)}")

# Test the checkpoint
print(f"\nTesting checkpoint model...")
test_model = SentenceTransformer(CHECKPOINT_DIR)
print(f"✓ Checkpoint loaded successfully")

# Test encoding
test_sentences = ["This is a test", "Here is another test"]
embeddings = test_model.encode(test_sentences)
print(f"✓ Encoding works: {embeddings.shape}")
print(f"  Embedding dimension: {embeddings.shape[1]}")

# Calculate similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
print(f"  Similarity between test sentences: {similarity:.4f}")

print("\n" + "=" * 80)
print("✓ CHECKPOINT READY FOR USE")
print("=" * 80)

print(f"\nTo use this model in your code:")
print(f"  from sentence_transformers import SentenceTransformer")
print(f"  model = SentenceTransformer('{CHECKPOINT_DIR}')")

print(f"\nTo use in main.py:")
print(f"  Update src/retrieval.py SemanticRetriever init:")
print(f"    retriever = SemanticRetriever(model_name='{CHECKPOINT_DIR}')")

print(f"\nModel Details:")
print(f"  - Name: {MODEL_NAME}")
print(f"  - Embedding DIM: {metadata['embedding_dimension']}")
print(f"  - Max Seq Length: {metadata['max_seq_length']}")
print(f"  - Use Case: {metadata['use_case']}")
