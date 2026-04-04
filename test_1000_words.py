#!/usr/bin/env python
"""
Test the increased summary length (1000 words)
"""
import requests
from pathlib import Path

file_path = Path('test_paper.txt')

print("=" * 80)
print("TESTING 1000-WORD SUMMARY LENGTH")
print("=" * 80)

with open(file_path, 'rb') as f:
    files = {'file': f}
    # Note: Not specifying max_length, should use new default of 1000
    response = requests.post('http://localhost:5000/api/summarize-file', files=files)

if response.status_code == 200:
    result = response.json()
    
    summary = result.get('summary', '')
    
    # Estimate word count (average 4.7 characters per word)
    char_count = len(summary)
    word_count = char_count / 4.7
    
    print("\n✓ SUCCESS - Summary generated!\n")
    print("=" * 80)
    print("SUMMARY STATISTICS:")
    print("=" * 80)
    print(f"Summary length: {char_count} characters")
    print(f"Estimated words: ~{int(word_count)} words")
    print(f"Compression ratio: {result.get('compression_ratio')}%")
    
    print("\n" + "=" * 80)
    print("SUMMARY (First 500 chars):")
    print("=" * 80)
    print(summary[:500])
    if len(summary) > 500:
        print(f"\n... ({len(summary) - 500} more characters)")
    
    print("\n" + "=" * 80)
    if word_count >= 900:
        print(f"✅ SUCCESS: Summary is ~{int(word_count)} words (target: 1000+)")
    else:
        print(f"⚠️  Summary is ~{int(word_count)} words (target: 1000+)")
    print("=" * 80)
    
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)
