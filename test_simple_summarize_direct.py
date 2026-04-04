#!/usr/bin/env python3
"""Direct test of simple_summarize function"""
import sys
sys.path.insert(0, '.')

# Import the function directly from src.utils
from src.utils import simple_summarize

# Sample research paper text to test
test_text = """
Automated Literature Review Using NLP Techniques and Machine Learning Models

Abstract: This paper presents a comprehensive system for automated literature review using natural language processing and machine learning. Our approach combines multiple techniques including document retrieval, semantic analysis, and information extraction to provide researchers with meaningful summaries and insights from large document collections. The system has been tested on over 10,000 academic papers from the arXiv database with promising results.

Introduction: Literature reviews are fundamental to research methodology, yet they are increasingly challenging as the volume of published research grows exponentially. Manual review of hundreds or thousands of papers becomes impractical for most researchers. This work presents an automated system that can analyze documents, extract key information, generate summaries, and identify relationships between papers. Our implementation leverages state-of-the-art transformer models and combines multiple NLP techniques to achieve high-quality results.

Methodology: We employ a multi-stage pipeline consisting of document preprocessing, semantic analysis using transformer models, entity recognition, and summary generation. The preprocessing stage handles PDF extraction, OCR correction, and text normalization. The semantic analysis stage uses pre-trained models to understand document content. Entity recognition identifies key concepts, authors, and methodologies. Finally, the summary generation stage creates concise yet comprehensive abstracts of the original documents.

Results: Our system achieved a compression ratio of 15-20% while maintaining 85% of the original information content according to human evaluation. Processing speed averages 2-3 seconds per document. The system correctly identifies relationships between papers with 78% accuracy when compared to manual expert review. User testing indicates that automatically generated summaries save researchers approximately 60% of the time typically spent on literature review tasks.

Discussion: The results demonstrate that automated literature review systems can significantly enhance researcher productivity. The high accuracy of relationship identification suggests that semantic analysis methods are effective for this task. We identify several limitations including handling of highly technical terminology and maintaining context across documents. Future work should address these limitations through domain-specific model training and improved context preservation mechanisms.

Conclusion: This paper successfully demonstrates an automated literature review system that achieves high-quality results while significantly reducing the time required for literature review tasks. The system is practical for real-world deployment and can handle large volumes of academic papers efficiently. This work opens new possibilities for AI-assisted research methodologies.

References: [1] Smith et al., 2023. Document Analysis Review... [2] Johnson et al., 2022. Neural Text Processing... [3] Williams et al., 2021. Transformer Applications...
"""

# Extend test text for larger document
extended_text = test_text * 3  # Triple the size

print("=" * 80)
print("DIRECT TEST OF simple_summarize() FUNCTION")
print("=" * 80)
print(f"\nInput document:")
print(f"  Length: {len(extended_text)} characters")
print(f"  Words: {len(extended_text.split())} words")

print(f"\nCalling simple_summarize with:")
print(f"  max_length: 5000 characters")
print(f"  min_length: 2000 characters")

summary = simple_summarize(extended_text, max_length=5000, min_length=2000)

print(f"\nGenerated Summary:")
print("-" * 80)
print(summary)
print("-" * 80)

print(f"\nResults:")
print(f"  Summary length: {len(summary)} characters")
print(f"  Word count: {len(summary.split())} words")
print(f"  Compression ratio: {round((len(summary) / len(extended_text)) * 100, 2)}%")

if len(summary) >= 2000:
    print(f"\n✅ SUCCESS: Summary meets minimum length ({len(summary)} >= 2000)")
else:
    print(f"\n❌ ISSUE: Summary is only {len(summary)} chars (target: 2000+)")
    print(f"  This suggests the function is not selecting enough sentences.")
