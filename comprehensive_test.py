#!/usr/bin/env python
"""
Comprehensive test of the preprocessing pipeline
"""
from src.model import SummarizationModel

# Example text with heavy metadata (like the user's PDF)
test_paper = """Automated Literature Review Using NLP Techniquesand LLM-Based Retrieval-Augmented Generation

Nurshat Fateh Ali
Departmentof Computer Scienceand Engineering
Military Instituteof Scienceand Technology
Dhaka, Bangladesh
nurshatfateh@gmail.com

Shakil Mosharrof
Departmentof Computer Scienceand Engineering
Military Instituteof Scienceand Technology
Dhaka, Bangladesh
shakilmrf@gmail.com

Md. Mahdi Mohtasim
Departmentof Computer Scienceand Engineering
Military Instituteof Scienceand Technology
Dhaka, Bangladesh
mohtasim@gmail.com

Vol. 15, No. 2, 2025, pp. 123-145
© 2025 ACM

Abstract
This paper presents a comprehensive literature review on automated systems for analyzing research papers using NLP and large language models. We examine recent advances in retrieval-augmented generation systems.

Introduction
Scientific publication has grown exponentially. Automated systems are critical for processing literature. This study surveys developments in NLP-based literature analysis, covering retrieval methods and language models. We identify key trends in RAG systems.

Methods
We conducted a systematic literature review from 2020-2024. We identified 487 papers through database searches. Based on inclusion criteria, 156 papers underwent detailed analysis. Papers were assessed for methodological quality and relevance.

Results
Our analysis shows that transformer models outperform traditional methods by 15-30%. Hybrid retrieval combining BM25 and dense methods is most effective. Fine-tuning on domain data improves specialized task performance. Recent models show exceptional generalization.

Conclusion
This review demonstrates significant progress in literature analysis through NLP. Future work should focus on interpretability and computational efficiency. RAG systems show particular promise for specialized domains.

References
[1] Author et al. 2024. Paper Title. Journal Name, Vol 10."""

print("=" * 80)
print("PREPROCESSING PIPELINE TEST")
print("=" * 80)

summarizer = SummarizationModel()

print("\n[STEP 1] INPUT TEXT")
print("-" * 80)
print(f"Length: {len(test_paper)} characters")
print(f"Sample:\n{test_paper[:300]}...")

print("\n[STEP 2] APPLY PREPROCESSING")
print("-" * 80)
preprocessed = summarizer.preprocess_text(test_paper)
removed_chars = len(test_paper) - len(preprocessed)
removed_pct = round((removed_chars / len(test_paper)) * 100, 1)

print(f"✓ Preprocessing complete")
print(f"  Original length: {len(test_paper)} chars")
print(f"  After preprocessing: {len(preprocessed)} chars")
print(f"  Removed: {removed_chars} chars ({removed_pct}%)")

print("\n[STEP 3] PREPROCESSED OUTPUT")
print("-" * 80)
print(preprocessed)

print("\n[STEP 4] VERIFICATION - What was removed?")
print("-" * 80)
checks = [
    ("Author names (Nurshat, Shakil, Mahdi)", "Nurshat" not in preprocessed and "Md." not in preprocessed),
    ("Affiliations (Department, Military Institute)", "Departmentof" not in preprocessed and "Military" not in preprocessed),
    ("Emails (nurshatfateh@gmail.com, etc)", "@gmail" not in preprocessed and "mohtasim@" not in preprocessed),
    ("Locations (Dhaka, Bangladesh)", "Dhaka" not in preprocessed and "Bangladesh" not in preprocessed),
    ("Journal metadata (Vol, No, pp)", "Vol." not in preprocessed and "pp." not in preprocessed),
    ("Copyright notice (© 2025)", "©" not in preprocessed),
    ("Page numbers (123-145)", "123-145" not in preprocessed),
]

all_passed = True
for check_name, result in checks:
    status = "✓" if result else "✗"
    print(f"  {status} {check_name}: {'REMOVED' if result else 'STILL PRESENT'}")
    if not result:
        all_passed = False

print("\n[STEP 5] SUMMARY")
print("-" * 80)
if all_passed:
    print("✅ ALL METADATA SUCCESSFULLY REMOVED!")
    print(f"\n   The summarizer now provides CLEAN OUTPUT with only research content.")
    print(f"   Author names, affiliations, emails, locations are all filtered out.")
else:
    print("⚠️  Some metadata items were not removed - manual review needed")

print("\n" + "=" * 80)
