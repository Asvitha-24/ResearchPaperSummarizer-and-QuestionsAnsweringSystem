#!/usr/bin/env python
"""
Flask API Server for Research Paper Summarizer & QA System
Connects the backend ML models with the frontend React application
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.model import ResearchPaperQASystem

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Initialize the QA System
try:
    qa_system = ResearchPaperQASystem()
    print("✅ QA System initialized successfully")
except Exception as e:
    print(f"❌ Error initializing QA System: {e}")
    qa_system = None


# ==================== HEALTH CHECK ====================
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Research Paper QA System API is running'
    }), 200


# ==================== SUMMARIZATION ====================
@app.route('/api/summarize', methods=['POST'])
def summarize():
    """
    Summarize research paper text
    Expected JSON: {"text": "paper content", "max_length": 150}
    """
    try:
        if not qa_system:
            return jsonify({'error': 'QA System not initialized'}), 500
        
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing required field: text'}), 400
        
        text = data['text']
        max_length = data.get('max_length', 150)
        min_length = data.get('min_length', 50)
        
        summary = qa_system.summarizer.summarize(
            text,
            max_length=max_length,
            min_length=min_length
        )
        
        return jsonify({
            'success': True,
            'summary': summary,
            'original_length': len(text),
            'summary_length': len(summary)
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== QUESTION ANSWERING ====================
@app.route('/api/answer', methods=['POST'])
def answer_question():
    """
    Answer questions about research paper content
    Expected JSON: {"question": "What is...", "context": "paper text"}
    """
    try:
        if not qa_system:
            return jsonify({'error': 'QA System not initialized'}), 500
        
        data = request.get_json()
        if not data or 'question' not in data or 'context' not in data:
            return jsonify({'error': 'Missing required fields: question, context'}), 400
        
        question = data['question']
        context = data['context']
        
        answer = qa_system.qa_model.answer_question(question, context)
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': answer
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== SEMANTIC SEARCH ====================
@app.route('/api/search', methods=['POST'])
def search_papers():
    """
    Search papers using semantic similarity
    Expected JSON: {"query": "search term", "papers": [...], "top_k": 5}
    """
    try:
        if not qa_system:
            return jsonify({'error': 'QA System not initialized'}), 500
        
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing required field: query'}), 400
        
        query = data['query']
        papers = data.get('papers', [])
        top_k = data.get('top_k', 5)
        
        results = qa_system.retriever.search(query, papers, top_k=top_k)
        
        return jsonify({
            'success': True,
            'query': query,
            'results': results,
            'count': len(results)
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== DOCUMENT UPLOAD ====================
@app.route('/api/documents/upload', methods=['POST'])
def upload_document():
    """
    Upload and process a document (PDF, TXT, DOCX)
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Save file temporarily
        upload_folder = 'data/uploads'
        Path(upload_folder).mkdir(parents=True, exist_ok=True)
        filepath = os.path.join(upload_folder, file.filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'filename': file.filename,
            'filepath': filepath
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== ERROR HANDLERS ====================
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("🚀 Starting Research Paper QA System API Server...")
    print("📍 Server running at: http://localhost:5000")
    print("📚 API Documentation available at: http://localhost:5000/api/health")
    app.run(debug=True, host='0.0.0.0', port=5000)
