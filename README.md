# Research Paper Summarizer & Question Answering System

## **Overview**
This project implements a **Research Paper Summarizer** and **Question Answering (QA) System** using **Python**, **Machine Learning (ML)**, and **Natural Language Processing (NLP)** techniques. The system can:
- Summarize research papers into concise abstractive summaries
- Answer research-related questions based on document content
- Retrieve relevant papers using hybrid semantic and keyword search

## **Features**
-  **Abstractive Text Summarization**: BART-based automatic summarization
-  **Question Answering**: DistilBERT QA on paper content
-  **Semantic Search**: Hybrid retrieval combining TF-IDF and embeddings
-  **Comprehensive Evaluation**: ROUGE, BERTScore, F1, NDCG metrics
-  **GPU Acceleration**: CUDA support with CPU fallback
-  **Production Ready**: Pre-trained models, no fine-tuning needed

## **Quick Start**

### **Installation**

\\\ash
pip install -r requirements.txt
\\\

### **Basic Usage**

\\\python
from src.model import ResearchPaperQASystem

# Initialize system
qa_system = ResearchPaperQASystem()

# Summarize text
summary = qa_system.summarizer.summarize(paper_text, max_length=150)

# Answer questions
answer = qa_system.qa_model.answer_question('What is the main topic?', paper_text)
\\\

## **Running the Notebook**

Open the Jupyter notebook for complete pipeline:
\\\ash
jupyter notebook notebooks/model_training_notebook.ipynb
\\\

## **System Architecture**

See **ARCHITECTURE.md** for detailed diagrams and technical flow.

**Components:**
- **Summarization**: BART (406M params) - Abstractive summarization
- **QA**: DistilBERT (66M params) - Answer extraction from context
- **Retrieval**: Hybrid approach - 60% semantic + 40% TF-IDF
- **Evaluation**: ROUGE, BERTScore, F1, NDCG metrics

## **Dataset**

- **Source**: arXiv Scientific Research Papers Dataset
- **Size**: 136,238 papers
- **Split**: 70% train, 15% validation, 15% test

## **Performance**

| Task | Metric | Score |
|------|--------|-------|
| Summarization | ROUGE-1 F | 0.40 |
| QA | Avg Confidence | 0.70 |
| Retrieval | NDCG@5 | 0.62 |

## **Testing**

\\\ash
pytest tests/ -v
\\\

---

**Status**:  Complete and Tested
**Last Updated**: January 2025
