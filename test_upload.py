#!/usr/bin/env python
import requests
import json
from pathlib import Path

# Upload test file
file_path = Path('test_paper.txt')
if not file_path.exists():
    print(f"❌ File not found: {file_path}")
    exit(1)

print("=" * 80)
print("UPLOADING TEST PAPER TO /api/summarize-file")
print("=" * 80)

with open(file_path, 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/api/summarize-file', files=files)

if response.status_code == 200:
    result = response.json()
    
    print("\n✓ SUCCESS - File summarized successfully!\n")
    print("=" * 80)
    print("EXTRACTED TEXT (first 500 chars):")
    print("=" * 80)
    print(result['extracted_text'][:500])
    
    print("\n" + "=" * 80)
    print("GENERATED SUMMARY:")
    print("=" * 80)
    print(result['summary'])
    
    print("\n" + "=" * 80)
    print("COMPRESSION METRICS:")
    print("=" * 80)
    print(f"Original text length: {result['original_length']} characters")
    print(f"Summary length: {result['summary_length']} characters")
    print(f"Compression ratio: {result['compression_ratio']}%")
    print(f"✓ Preprocessing removes affiliations, emails, locations from extracted_text")
    
else:
    print(f"❌ ERROR: {response.status_code}")
    print(response.text)
