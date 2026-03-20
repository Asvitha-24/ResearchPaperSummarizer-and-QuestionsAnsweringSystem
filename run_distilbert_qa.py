"""
Quick start script to run DistilBERT QA fine-tuning.
Choose which version to run: full or lightweight.
"""

import argparse
import subprocess
import sys
import os


def run_lightweight():
    """Run lightweight fine-tuning."""
    print("\n" + "="*70)
    print("RUNNING LIGHTWEIGHT DISTILBERT QA FINE-TUNING")
    print("="*70)
    print("\nThis version:")
    print("  - Uses simple PyTorch training loop")
    print("  - Generates synthetic QA pairs from your dataset")
    print("  - Faster training for quick testing")
    print("  - Smaller batch size (4)")
    print("  - 2 epochs by default")
    
    result = subprocess.run(
        [sys.executable, "finetune_distilbert_qa_simple.py"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    return result.returncode


def run_full():
    """Run full fine-tuning with HuggingFace Trainer."""
    print("\n" + "="*70)
    print("RUNNING FULL DISTILBERT QA FINE-TUNING")
    print("="*70)
    print("\nThis version:")
    print("  - Uses HuggingFace Trainer API")
    print("  - Can load SQuAD or generate synthetic QA pairs")
    print("  - More optimization features (learning rate scheduling, etc.)")
    print("  - Better for production use")
    print("  - Larger batch size (8)")
    print("  - 3 epochs by default")
    
    result = subprocess.run(
        [sys.executable, "finetune_distilbert_qa.py"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    return result.returncode


def run_interactive():
    """Run interactive QA testing."""
    print("\n" + "="*70)
    print("RUNNING INTERACTIVE QA MODE")
    print("="*70)
    
    model_path = input(
        "Enter model path "
        "(default: ./checkpoints/distilbert_qa_lightweight): "
    ).strip()
    
    if not model_path:
        model_path = "./checkpoints/distilbert_qa_lightweight"
    
    result = subprocess.run(
        [sys.executable, "distilbert_qa_utils.py", model_path],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="DistilBERT QA Fine-tuning Quick Start"
    )
    parser.add_argument(
        "--mode",
        choices=["lightweight", "full", "interactive"],
        help="Which mode to run"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of training samples"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size"
    )
    
    args = parser.parse_args()
    
    if not args.mode:
        print("\n" + "="*70)
        print("DISTILBERT QA FINE-TUNING - QUICK START")
        print("="*70)
        print("\nChoose which mode to run:")
        print("\n1. Lightweight (recommended for testing)")
        print("   - Simple PyTorch training loop")
        print("   - Fast training")
        print("   - Good for quick experimentation")
        print("\n2. Full (recommended for production)")
        print("   - HuggingFace Trainer with advanced features")
        print("   - Better optimization")
        print("   - More control and monitoring")
        print("\n3. Interactive")
        print("   - Test your fine-tuned model")
        print("   - Ask questions on custom context")
        
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice == "1":
            args.mode = "lightweight"
        elif choice == "2":
            args.mode = "full"
        elif choice == "3":
            args.mode = "interactive"
        else:
            print("Invalid choice!")
            return 1
    
    # Run selected mode
    if args.mode == "lightweight":
        return run_lightweight()
    elif args.mode == "full":
        return run_full()
    elif args.mode == "interactive":
        return run_interactive()


if __name__ == "__main__":
    sys.exit(main())
