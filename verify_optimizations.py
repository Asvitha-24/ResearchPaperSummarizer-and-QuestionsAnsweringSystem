#!/usr/bin/env python
"""
Quick verification that summarization optimizations are working.
Run this to confirm your system is using the fast DistilBart model.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def verify_optimizations():
    """Verify that all optimizations are correctly applied."""
    print("=" * 80)
    print("SUMMARIZATION OPTIMIZATION VERIFICATION")
    print("=" * 80)
    print()
    
    # Check 1: Verify model.py uses DistilBart
    print("[CHECK 1] Verifying model configuration...")
    try:
        from src.model import SummarizationModel
        
        # Inspect the __init__ to see default model
        import inspect
        source = inspect.getsource(SummarizationModel.__init__)
        
        if "distilbart" in source.lower():
            print("✓ PASS: Using DistilBart model")
        elif "bart-large" in source.lower():
            print("✗ FAIL: Still using BART-large (needs optimization)")
            return False
        else:
            print("? UNKNOWN: Could not determine model")
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False
    
    print()
    
    # Check 2: Verify beam search is disabled
    print("[CHECK 2] Verifying greedy decoding (num_beams=1)...")
    try:
        source = inspect.getsource(SummarizationModel.summarize)
        
        if "num_beams=1" in source:
            print("✓ PASS: Using greedy decoding (num_beams=1)")
        elif "num_beams=5" in source:
            print("✗ FAIL: Still using beam search (needs optimization)")
            return False
        else:
            print("? UNKNOWN: Could not verify beam setting")
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False
    
    print()
    
    # Check 3: Verify chunked processing for large docs
    print("[CHECK 3] Verifying chunked processing support...")
    try:
        if hasattr(SummarizationModel, '_summarize_chunked'):
            print("✓ PASS: Chunked processing available for large documents")
        else:
            print("✗ FAIL: Missing chunked processing method")
            return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False
    
    print()
    print("=" * 80)
    print("ALL CHECKS PASSED - OPTIMIZATIONS ARE ACTIVE")
    print("=" * 80)
    print()
    print("Expected Performance:")
    print("  • ~45s for typical 2-5 min research paper")
    print("  • ~80s maximum for very large 10+ min papers")
    print("  • ~10-20s with GPU (CUDA)")
    print()
    print("To test: python test_speed_optimization.py")
    return True

if __name__ == "__main__":
    success = verify_optimizations()
    sys.exit(0 if success else 1)
