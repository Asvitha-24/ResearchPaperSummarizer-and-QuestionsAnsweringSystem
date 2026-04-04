#!/usr/bin/env python
import sys
# Force reimport
if 'src.model' in sys.modules:
    del sys.modules['src.model']
if 'src' in sys.modules:
    del sys.modules['src']

from src.model import SummarizationModel

test_text = '''Automated Literature Review Using NLP Techniquesand LLM-Based Retrieval-Augmented Generation Nurshat Fateh Ali Departmentof Computer Scienceand Engineering Military Instituteof Scienceand Technology Dhaka, Bangladeshnurshatfateh@gmail.com Shakil Mosharrof Departmentof Computer Scienceand Engineering Military Instituteof Scienceand Technology Dhaka, Bangladesh'''

s = SummarizationModel()
result = s.preprocess_text(test_text)

print("=" * 80)
print("ORIGINAL TEXT:")
print("=" * 80)
print(test_text)
print("\n" + "=" * 80)
print("AFTER PREPROCESSING:")
print("=" * 80)
print(result)
print("\n" + "=" * 80)
print(f"Original length: {len(test_text)} chars")
print(f"Preprocessed length: {len(result)} chars")
print(f"Removed: {len(test_text) - len(result)} chars ({round((len(test_text) - len(result)) / len(test_text) * 100)}% reduction)")
print("=" * 80)
