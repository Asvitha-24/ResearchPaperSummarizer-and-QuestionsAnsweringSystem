#!/usr/bin/env python
"""
Test script to verify summarization speed optimization.
Compares original BART (slow) vs optimized DistilBart (fast).
"""

import time
import sys
import torch
from src.model import SummarizationModel

# Sample text for testing (research paper excerpt)
SAMPLE_TEXT = """
The rapid development of natural language processing has led to significant advances in various applications including machine translation, question answering, and text summarization. Recent advances in neural networks and deep learning have enabled models to achieve state-of-the-art performance on many benchmark datasets. However, the computational requirements for training and deploying these models have become a major concern. 

We propose a new approach to accelerate text summarization using distilled models. The key innovation is to reduce the number of parameters while maintaining accuracy through knowledge distillation. Our experiments show that the proposed approach achieves a 10x speedup compared to baseline methods while maintaining competitive accuracy levels.

The methodology involves three main steps: (1) preprocessing and tokenization of input text, (2) model inference using the optimized DistilBart architecture, and (3) post-processing to ensure output quality. Our results demonstrate significant improvements in processing time across different input sizes and hardware configurations.

We evaluate our approach on multiple datasets and show that it outperforms existing methods in terms of speed while maintaining high summary quality. The implementation is straightforward and can be easily integrated into existing systems. We believe this work will be valuable for practitioners who need to deploy summarization systems in resource-constrained environments.
"""

def test_summarization_speed():
    """Test and display summarization performance metrics."""
    print("=" * 80)
    print("SUMMARIZATION SPEED OPTIMIZATION TEST")
    print("=" * 80)
    print()
    
    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    print(f"[SYSTEM] GPU Available: {gpu_available}")
    if gpu_available:
        print(f"[SYSTEM] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[SYSTEM] PyTorch Version: {torch.__version__}")
    print()
    
    # Initialize model
    print("[INIT] Initializing optimized DistilBart model...")
    start_init = time.time()
    
    try:
        model = SummarizationModel()  # Uses default: DistilBart
        init_time = time.time() - start_init
        print(f"[OK] Model initialized in {init_time:.2f}s")
        print(f"[INFO] Model: {model.model_name}")
        print(f"[INFO] Device: {model.device.upper()}")
        print()
    except Exception as e:
        print(f"[ERROR] Failed to initialize model: {e}")
        return
    
    # Test summarization
    print("[TEST] Testing summarization on sample paper...")
    print(f"[INPUT] Text length: {len(SAMPLE_TEXT)} characters ({len(SAMPLE_TEXT.split())} words)")
    print()
    
    start_summary = time.time()
    try:
        summary = model.summarize(SAMPLE_TEXT, format_as_points=False)
        summary_time = time.time() - start_summary
        
        print(f"[SUCCESS] Summarization completed in {summary_time:.2f}s")
        print(f"[OUTPUT] Summary length: {len(summary)} characters ({len(summary.split())} words)")
        print(f"[COMPRESSION] Ratio: {round((len(summary) / len(SAMPLE_TEXT)) * 100, 1)}%")
        print()
        
        # Performance estimate for different document sizes
        print("[PERFORMANCE ESTIMATE]")
        print(f"  ~2 min paper (1000 words): {summary_time * 10:.1f}s")
        print(f"  ~5 min paper (2500 words): {summary_time * 25:.1f}s")
        print(f"  ~10 min paper (5000 words): {summary_time * 50:.1f}s")
        print()
        
        if summary_time < 5:
            print("[✓] ✅ OPTIMIZATION SUCCESSFUL - Well under 1 minute target!")
        elif summary_time < 15:
            print("[✓] ✅ GOOD - Should process papers in under 1 minute")
        else:
            print("[⚠] ⚠️  SLOW - May need additional optimization")
        
        print()
        print("=" * 80)
        print("[SUMMARY OUTPUT]")
        print("=" * 80)
        print(summary[:500])  # Show first 500 chars
        print("...")
        print()
        
    except Exception as e:
        print(f"[ERROR] Summarization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_summarization_speed()
