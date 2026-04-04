#!/usr/bin/env python
"""Quick test of the summarize endpoint"""

import requests
import json

def test_summarize():
    text = """
    Machine learning is a subset of artificial intelligence (AI) that enables systems to learn and improve from experience without being explicitly programmed. 
    Deep learning is a specialized branch of machine learning that uses artificial neural networks with multiple layers to process data. 
    These technologies are transforming various industries by enabling computers to perform tasks that typically require human intelligence, 
    such as visual perception, speech recognition, decision-making, and language translation.
    """
    
    print("Testing /api/summarize endpoint...")
    print("=" * 60)
    
    try:
        response = requests.post(
            'http://localhost:5000/api/summarize',
            json={'text': text},
            timeout=60
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ SUCCESS")
            print(f"\nOriginal length: {result['original_length']} chars")
            print(f"Summary length: {result['summary_length']} chars")
            print(f"Compression ratio: {result['compression_ratio']}%")
            print(f"\nSummary:\n{result['summary']}")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Connection error: {e}")

if __name__ == '__main__':
    test_summarize()
