"""
Verification script to check if all dependencies are installed 
and the system is ready for BART fine-tuning.
"""

import sys
import torch
import importlib
from pathlib import Path


def check_import(package_name, import_name=None):
    """Check if package is installed and can be imported."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name}")
        return True
    except ImportError as e:
        print(f"❌ {package_name} - {str(e)}")
        return False


def check_system():
    """Check system requirements."""
    print("\n" + "=" * 80)
    print("BART FINE-TUNING SYSTEM CHECK")
    print("=" * 80 + "\n")
    
    all_good = True
    
    # Python version
    print("1. Python Environment:")
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"   Python: {py_version}")
    if sys.version_info >= (3, 8):
        print("   ✅ Python version OK")
    else:
        print("   ❌ Python 3.8+ required")
        all_good = False
    
    # PyTorch & CUDA
    print("\n2. PyTorch & GPU:")
    print(f"   PyTorch: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA Available: {'✅ Yes' if cuda_available else '⚠️  No (will use CPU, slower)'}")
    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU Memory: {gpu_memory:.2f} GB")
        if gpu_memory < 4:
            print("   ⚠️  Warning: GPU has < 4GB memory, training may be slow")
        elif gpu_memory >= 8:
            print("   ✅ GPU memory sufficient")
    
    # Required packages
    print("\n3. Required Packages:")
    packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('datasets', 'datasets'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('nltk', 'nltk'),
        ('tqdm', 'tqdm'),
        ('scipy', 'scipy'),
    ]
    
    for package, import_name in packages:
        if not check_import(package, import_name):
            all_good = False
    
    # Optional packages
    print("\n4. Optional Packages:")
    optional = [
        ('rouge-score', 'rouge_score'),
        ('bert-score', 'bert_score'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
    ]
    
    for package, import_name in optional:
        check_import(package, import_name)
    
    # Dataset files
    print("\n5. Dataset Files:")
    dataset_path = Path('data/raw/arXiv Scientific Research Papers Dataset.csv')
    if dataset_path.exists():
        size_gb = dataset_path.stat().st_size / 1e9
        print(f"   ✅ Found arXiv dataset ({size_gb:.2f} GB)")
    else:
        print(f"   ⚠️  Dataset not found at {dataset_path}")
        print("      The fine-tuning scripts will attempt synthetic data if needed")
    
    # Output directory
    print("\n6. Output Directories:")
    output_dir = Path('checkpoints')
    if output_dir.exists():
        print(f"   ✅ Checkpoints directory exists")
    else:
        print(f"   📁 Creating checkpoints directory...")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created")
    
    # Fine-tuning scripts
    print("\n7. Fine-tuning Scripts:")
    scripts = [
        'finetune_bart_lite.py',
        'finetune_bart.py',
        'finetune_bart_inference.py',
        'finetune_config.py',
    ]
    
    for script in scripts:
        script_path = Path(script)
        if script_path.exists():
            print(f"   ✅ {script}")
        else:
            print(f"   ❌ {script} - NOT FOUND")
            all_good = False
    
    # Summary
    print("\n" + "=" * 80)
    if all_good:
        print("✅ ALL CHECKS PASSED - Ready for fine-tuning!")
    else:
        print("⚠️  Some issues found - see above")
    print("=" * 80 + "\n")
    
    # Quick recommendations
    print("Next Steps:")
    print("1. Start with quick test:")
    print("   python finetune_bart_lite.py --samples 1000 --epochs 2")
    print()
    print("2. Or use the menu script:")
    if sys.platform == "win32":
        print("   run_finetune.bat")
    else:
        print("   bash run_finetune.sh")
    print()
    print("3. See FINETUNE_README.md for detailed guide")
    print()
    
    return all_good


if __name__ == "__main__":
    success = check_system()
    sys.exit(0 if success else 1)
