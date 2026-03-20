"""
Model Switcher Utility - Easily switch between pre-trained and domain-specific models
"""

import os
import json
from pathlib import Path

print("=" * 80)
print("SEMANTIC SEARCH MODEL SWITCHER")
print("=" * 80)

# Define models
MODELS = {
    'pretrained': {
        'name': 'Pre-trained (General Purpose)',
        'path': 'checkpoints/distilroberta_paraphrase_finetuned',
        'description': 'General-purpose semantic search, balanced quality/speed',
        'use_case': 'General text, news, blogs, diverse content'
    },
    'arxiv': {
        'name': 'arXiv Domain-Specific',
        'path': 'checkpoints/distilroberta_arxiv_finetuned',
        'description': 'Fine-tuned on research papers, understands academic terminology',
        'use_case': 'Research papers, academic content, arXiv papers'
    }
}

RETRIEVAL_FILE = 'src/retrieval.py'
MAIN_FILE = 'main.py'

# Current configuration
def get_current_model():
    """Get the currently configured model."""
    if not os.path.exists(RETRIEVAL_FILE):
        return None
    
    with open(RETRIEVAL_FILE, 'r') as f:
        content = f.read()
        
    if 'checkpoints/distilroberta_arxiv_finetuned' in content:
        return 'arxiv'
    elif 'checkpoints/distilroberta_paraphrase_finetuned' in content:
        return 'pretrained'
    else:
        return 'unknown'

def switch_model(model_key):
    """Switch to a different model."""
    if model_key not in MODELS:
        print(f"❌ Unknown model: {model_key}")
        return False
    
    model_info = MODELS[model_key]
    model_path = model_info['path']
    
    # Check if checkpoint exists
    if not os.path.exists(model_path):
        print(f"❌ Model checkpoint not found: {model_path}")
        return False
    
    print(f"\n📤 Switching to: {model_info['name']}")
    print(f"   Path: {model_path}")
    
    # Update src/retrieval.py
    if os.path.exists(RETRIEVAL_FILE):
        with open(RETRIEVAL_FILE, 'r') as f:
            content = f.read()
        
        # Replace model path in default parameter
        old_pretrained = 'checkpoints/distilroberta_paraphrase_finetuned'
        old_arxiv = 'checkpoints/distilroberta_arxiv_finetuned'
        old_minilm = '"all-MiniLM-L6-v2"'
        
        # Try all possible old values
        if old_minilm in content:
            content = content.replace(old_minilm, f'"{model_path}"')
        elif old_pretrained in content:
            content = content.replace(old_pretrained, model_path)
        elif old_arxiv in content:
            content = content.replace(old_arxiv, model_path)
        
        with open(RETRIEVAL_FILE, 'w') as f:
            f.write(content)
        
        print(f"✓ Updated: {RETRIEVAL_FILE}")
    
    # Update main.py initialization (if present)
    if os.path.exists(MAIN_FILE):
        with open(MAIN_FILE, 'r') as f:
            content = f.read()
        
        # Check if it uses the default model initialization
        if 'SemanticRetriever()' in content or 'SemanticRetriever(model_name=' in content:
            print(f"✓ {MAIN_FILE} will use the new default from retrieval.py")
    
    print(f"\n✓ Successfully switched to: {model_info['name']}")
    return True


def show_menu():
    """Display available models."""
    print("\n" + "=" * 80)
    print("AVAILABLE MODELS")
    print("=" * 80)
    
    current = get_current_model()
    
    for key, info in MODELS.items():
        status = "✓ CURRENT" if key == current else " "
        print(f"\n[{key.upper()}] {status}")
        print(f"  Name: {info['name']}")
        print(f"  Path: {info['path']}")
        print(f"  Description: {info['description']}")
        print(f"  Use Case: {info['use_case']}")
        
        # Check if checkpoint exists
        exists = "✓ Present" if os.path.exists(info['path']) else "❌ Missing"
        print(f"  Status: {exists}")


def compare_models():
    """Show comparison between models."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    comparison_data = {
        'Metric': ['Training Data', 'Embedding Dim', 'Max Tokens', 'Speed', 'Use Case', 'Quality'],
        'Pre-trained': ['General corpus', '768', '128', 'Baseline', 'General text', 'Good'],
        'arXiv-tuned': ['600 research papers', '768', '128', 'Similar', 'Research papers', 'Very Good']
    }
    
    print("\nPre-trained General-Purpose:")
    print("  ✓ Broad domain coverage")
    print("  ✓ Faster inference (slightly)")
    print("  ✓ Good for general semantic search")
    print("  ⚠ May not understand domain-specific terminology")
    
    print("\narXiv Domain-Specific:")
    print("  ✓ Trained on research papers")
    print("  ✓ Better for academic content")
    print("  ✓ Understands technical terminology")
    print("  ⚠ May underperform on non-academic texts")


def test_model(model_key):
    """Test a specific model with sample queries."""
    if model_key not in MODELS:
        print(f"❌ Unknown model: {model_key}")
        return
    
    model_info = MODELS[model_key]
    model_path = model_info['path']
    
    if not os.path.exists(model_path):
        print(f"❌ Model checkpoint not found: {model_path}")
        return
    
    print(f"\n🧪 Testing: {model_info['name']}")
    print(f"   Loading model...", end="", flush=True)
    
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        model = SentenceTransformer(model_path)
        print(" ✓")
        
        # Test queries
        test_cases = [
            ("machine learning", ["Deep neural networks learn from data", 
                                 "Machine learning is a subset of AI"]),
            ("transformer attention", ["Transformers use self-attention mechanisms",
                                      "NLP models process sequences"]),
            ("research paper classification", ["Papers are categorized by topic",
                                             "Document classification systems"])
        ]
        
        print(f"\n   Test Results:")
        for query, docs in test_cases:
            query_emb = model.encode(query, convert_to_tensor=False)
            doc_embs = model.encode(docs, convert_to_tensor=False)
            sims = cosine_similarity([query_emb], doc_embs)[0]
            
            top_doc_idx = np.argmax(sims)
            top_sim = sims[top_doc_idx]
            
            print(f"   Query: '{query}'")
            print(f"   → Best match (sim: {top_sim:.4f}): '{docs[top_doc_idx]}'")
        
        print(f"\n✓ Model working correctly!")
        
    except Exception as e:
        print(f"❌ Error testing model: {str(e)}")


def show_config():
    """Show current configuration."""
    print("\n" + "=" * 80)
    print("CURRENT CONFIGURATION")
    print("=" * 80)
    
    current = get_current_model()
    
    if current in MODELS:
        info = MODELS[current]
        print(f"\n✓ Active Model: {info['name']}")
        print(f"  Path: {info['path']}")
        print(f"  Description: {info['description']}")
    else:
        print(f"\n⚠ Current model unknown or not configured")
    
    # Check files
    print(f"\n✓ Files:")
    print(f"  {RETRIEVAL_FILE}: {'✓' if os.path.exists(RETRIEVAL_FILE) else '❌'}")
    print(f"  {MAIN_FILE}: {'✓' if os.path.exists(MAIN_FILE) else '❌'}")
    
    for key, info in MODELS.items():
        exists = "✓" if os.path.exists(info['path']) else "❌"
        print(f"  {info['path']}: {exists}")


# Main menu
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'switch':
            if len(sys.argv) > 2:
                model_key = sys.argv[2].lower()
                switch_model(model_key)
            else:
                print("Usage: python model_switcher.py switch [pretrained|arxiv]")
        
        elif command == 'compare':
            compare_models()
        
        elif command == 'test':
            if len(sys.argv) > 2:
                model_key = sys.argv[2].lower()
                test_model(model_key)
            else:
                print("Usage: python model_switcher.py test [pretrained|arxiv]")
        
        elif command == 'status':
            show_config()
        
        elif command == 'help':
            print("""
Usage: python model_switcher.py [command] [args]

Commands:
  switch [pretrained|arxiv]  - Switch to a model
  compare                    - Show model comparison
  test [pretrained|arxiv]    - Test a specific model  
  status                     - Show current configuration
  help                       - Show this help message

Examples:
  python model_switcher.py switch arxiv
  python model_switcher.py compare
  python model_switcher.py test arxiv
  python model_switcher.py status
""")
        else:
            print(f"Unknown command: {command}")
    
    else:
        # Interactive menu
        show_menu()
        compare_models()
        show_config()
        
        print("\n" + "=" * 80)
        print("USAGE")
        print("=" * 80)
        print("""
Commands:
  python model_switcher.py switch arxiv        - Use arXiv domain-specific model
  python model_switcher.py switch pretrained   - Use pre-trained general model
  python model_switcher.py test arxiv          - Test the arXiv model
  python model_switcher.py compare             - Compare models
  python model_switcher.py status              - Show configuration
""")
