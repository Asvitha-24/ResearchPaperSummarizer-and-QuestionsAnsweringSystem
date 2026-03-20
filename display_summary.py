#!/usr/bin/env python
"""
Display Summary Script
Test document upload and show the BART-generated summary
"""

import requests
import tempfile
import os

# Sample test document
test_text = '''
Artificial intelligence (AI) is the intelligence of computer systems, in contrast to the intelligence of humans or animals.
Machine learning is a subset of artificial intelligence that focuses on the development of computer programs that can access data.
Deep learning is part of a broader family of machine learning methods based on artificial neural networks.
Natural language processing is a subfield of linguistics, computer science, and artificial intelligence.
Computer vision is an interdisciplinary scientific field that deals with how digital cameras and computers can gain high-level understanding.
These technologies are transforming how we work, learn, and communicate with each other globally.
'''

print("=" * 80)
print("📄 DOCUMENT SUMMARIZATION WITH BART MODEL")
print("=" * 80)
print(f"\n📝 Original Text ({len(test_text)} characters):\n")
print(test_text)

try:
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_text)
        temp_file_path = f.name
    
    # Upload and summarize
    print("\n⏳ Sending to backend... (this may take 30-60 seconds)")
    with open(temp_file_path, 'rb') as f:
        files = {'file': ('test.txt', f)}
        response = requests.post(
            'http://localhost:5000/api/summarize-file',
            files=files,
            timeout=300
        )
    
    os.unlink(temp_file_path)
    
    if response.status_code == 200:
        result = response.json()
        
        summary = result.get('summary', 'N/A')
        original_length = result.get('original_length', 0)
        summary_length = result.get('summary_length', 0)
        compression = result.get('compression_ratio', 0)
        
        # Create output content
        output_lines = []
        output_lines.append("\n" + "=" * 80)
        output_lines.append("✅ BART GENERATED SUMMARY")
        output_lines.append("=" * 80)
        output_lines.append(f"\n{summary}\n")
        output_lines.append("=" * 80)
        output_lines.append("📊 STATISTICS")
        output_lines.append("=" * 80)
        output_lines.append(f"Original Length:    {original_length:,} characters")
        output_lines.append(f"Summary Length:     {summary_length:,} characters")
        output_lines.append(f"Compression Ratio:  {compression:.2f}%")
        output_lines.append("=" * 80 + "\n")
        
        output_text = "\n".join(output_lines)
        
        # Print to console
        print(output_text)
        
        # Save to file
        output_file = 'summary_output.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        
        print(f"✅ Output saved to: {output_file}")
        
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"\n❌ Error: {e}")
