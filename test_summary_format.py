#!/usr/bin/env python
"""Test script for new BART summary formatting"""

import requests
import json

# Test text (sample research paper abstract)
test_text = """
The development of effective machine learning models requires careful consideration of multiple factors. 
This includes data quality, model architecture, hyperparameter tuning, and evaluation metrics. 
Neural networks have shown remarkable success in various domains including computer vision, natural language processing, and quantitative analysis.
Recent advances in transformer-based models have revolutionized the field of deep learning. 
These models can understand context better and maintain information over long sequences. 
The application of these techniques to real-world problems has resulted in significant improvements in performance metrics.
Transfer learning has become a fundamental technique in machine learning. 
By leveraging pre-trained models, researchers can achieve better results with less training data. 
Fine-tuning these models on specific tasks allows for optimal performance in domain-specific applications.
Data augmentation and regularization techniques help prevent overfitting and improve generalization. 
Early stopping and dropout are commonly used methods to achieve this. 
Cross-validation provides a more reliable estimate of model performance on unseen data.
The computational requirements for training large models have increased significantly. 
GPUs and TPUs are now essential tools for deep learning research and development.
Cloud computing platforms have made these resources more accessible to researchers worldwide.
"""

def test_summarization():
    """Test the new summarization with point-form formatting"""
    
    # Endpoint
    url = "http://localhost:5000/api/summarize"
    
    # Request payload
    payload = {
        "text": test_text,
        "max_length": 250,
        "min_length": 100,
        "format_as_points": True
    }
    
    print("="*80)
    print("Testing BART Summarization with Point-Form Formatting")
    print("="*80)
    print(f"\nOriginal text length: {len(test_text)} characters")
    print(f"\nSending request to {url}")
    print(f"Payload: max_length={payload['max_length']}, min_length={payload['min_length']}, format_as_points={payload['format_as_points']}")
    print("\n" + "-"*80)
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        print("\n📋 SUMMARY (Point-Form Format):")
        print("-"*80)
        print(result.get('summary', 'No summary generated'))
        print("-"*80)
        print(f"\nSummary length: {result.get('summary_length', 0)} characters")
        print(f"Original length: {result.get('original_length', 0)} characters")
        print(f"Compression ratio: {result.get('compression_ratio', 0):.2f}%")
        print("\n✅ Test completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to server at http://localhost:5000")
        print("Make sure the Flask server is running.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_summarization()
