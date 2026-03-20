import requests
import tempfile
import os

test_text = '''
Artificial intelligence (AI) is the intelligence of computer systems, in contrast to the intelligence of humans or animals.
Machine learning is a subset of artificial intelligence that focuses on the development of computer programs that can access data.
Deep learning is part of a broader family of machine learning methods based on artificial neural networks.
Natural language processing is a subfield of linguistics, computer science, and artificial intelligence.
Computer vision is an interdisciplinary scientific field that deals with how digital cameras and computers can gain high-level understanding.
These technologies are transforming how we work, learn, and communicate with each other globally.
'''

print('Testing /api/summarize-file endpoint...')
print('Input text length:', len(test_text), 'characters')
print()

try:
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_text)
        temp_file_path = f.name
    
    # Open and send file
    with open(temp_file_path, 'rb') as f:
        files = {'file': ('test.txt', f)}
        response = requests.post(
            'http://localhost:5000/api/summarize-file',
            files=files,
            timeout=60
        )
    
    # Clean up
    os.unlink(temp_file_path)
    
    if response.status_code == 200:
        result = response.json()
        print('✅ SUCCESS!')
        print()
        print('EXTRACTED TEXT:')
        print(result.get('extracted_text', 'N/A')[:200])
        print()
        print('SUMMARY:')
        print(result.get('summary', 'N/A'))
        print()
        print('Original Length:', result.get('original_length', 'N/A'), 'chars')
        print('Summary Length:', result.get('summary_length', 'N/A'), 'chars')
        print('Compression Ratio:', result.get('compression_ratio', 'N/A'), '%')
    else:
        print('❌ Error:', response.status_code)
        print(response.text)
except Exception as e:
    print('❌ Error:', str(e))
    import traceback
    traceback.print_exc()
