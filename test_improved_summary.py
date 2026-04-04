#!/usr/bin/env python3
"""Test the improved summary generation with fixed sentence calculation"""
import requests
import json

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
extended_text = test_text * 3  # Triple the size to test with larger content

print("=" * 80)
print("TESTING IMPROVED SUMMARY GENERATION")
print("=" * 80)
print(f"\nOriginal document length: {len(extended_text)} characters")
print(f"Word count: {len(extended_text.split())} words\n")

# Test the API endpoint
url = "http://localhost:5000/api/summarize"
payload = {
    'text': extended_text,
}

try:
    print("Sending request to API...")
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json={
        'text': extended_text,
        'use_structured': False,  # Use simple summarizer for more predictable length
        'max_length': 5000,
        'min_length': 2000
    }, headers=headers, timeout=30)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ API Response Received")
        print(f"  Summary length: {data.get('summary_length')} characters")
        print(f"  Expected minimum: 2000 characters")
        print(f"  Compression ratio: {data.get('compression_ratio')}%")
        print(f"\nGenerated Summary:\n")
        print("-" * 80)
        summary = data.get('summary', '')
        print(summary)
        print("-" * 80)
        print(f"\nWord count: {len(summary.split())} words")
        
        # Validate summary length
        if len(summary) >= 2000:
            print(f"\n✅ SUCCESS: Summary meets minimum length requirement ({len(summary)} >= 2000 chars)")
        elif len(summary) >= 1000:
            print(f"\n⚠️  PARTIAL: Summary is {len(summary)} chars (target was 2000+)")
        else:
            print(f"\n❌ ISSUE: Summary is only {len(summary)} chars (target was 2000+)")
    else:
        print(f"❌ API Error: Status code {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"❌ Error: {e}")
