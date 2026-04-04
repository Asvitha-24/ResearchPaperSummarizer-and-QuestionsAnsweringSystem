#!/usr/bin/env python
"""Test 1000-word summary with longer document"""
import requests

file_path = 'test_paper_long.txt'
print('Uploading longer paper for 1000-word summary test...')
print()

with open(file_path, 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/api/summarize-file', files=files)

if response.status_code == 200:
    result = response.json()
    
    summary = result.get('summary', '')
    char_count = len(summary)
    word_count = char_count / 4.7
    
    print('✓ SUCCESS')
    print()
    print(f'Summary characters: {char_count}')
    print(f'Estimated words: ~{int(word_count)}')
    print(f'Compression ratio: {result.get("compression_ratio")}%')
    print()
    print('SUMMARY:')
    print('-' * 80)
    print(summary)
    print('-' * 80)
else:
    print(f'Error: {response.status_code}')
