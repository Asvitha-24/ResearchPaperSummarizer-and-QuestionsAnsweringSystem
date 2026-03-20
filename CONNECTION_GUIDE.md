# Backend & Frontend Connection Guide

## Overview
Your Flask backend is running on **port 5000** with multiple API endpoints. Your React frontend is configured to connect to it on **port 5000/api**.

---

## Setup Instructions

### **Step 1: Start the Backend (Flask)**

Navigate to your backend folder and run:

```bash
cd c:\Users\user\Desktop\Researcher\backend

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Start the Flask server
python app.py
```

✅ You should see:
```
🚀 Starting Research Paper QA System API Server...
📍 Server running at: http://localhost:5000
```

---

### **Step 2: Start the Frontend (React)**

In a new terminal, navigate to your frontend folder:

```bash
cd c:\Users\user\Desktop\Researcher\frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

✅ You should see:
```
Local:http://localhost:5173
```

---

### **Step 3: Verify Connection**

Test if frontend can reach backend by opening browser console and running:

```javascript
fetch('http://localhost:5000/api/health')
  .then(res => res.json())
  .then(data => console.log('✅ Connected:', data))
  .catch(err => console.error('❌ Error:', err))
```

---

## Available API Endpoints

### **1. Health Check**
- **Endpoint:** `GET /api/health`
- **Response:**
```json
{
  "status": "healthy",
  "message": "Research Paper QA System API is running"
}
```

### **2. Summarization**
- **Endpoint:** `POST /api/summarize`
- **Request:**
```json
{
  "text": "Your paper content here...",
  "max_length": 150,
  "min_length": 50
}
```
- **Response:**
```json
{
  "success": true,
  "summary": "Generated summary...",
  "original_length": 1000,
  "summary_length": 150
}
```

### **3. Question Answering**
- **Endpoint:** `POST /api/answer`
- **Request:**
```json
{
  "question": "What is machine learning?",
  "context": "Machine learning is a subset of AI that..."
}
```
- **Response:**
```json
{
  "success": true,
  "question": "What is machine learning?",
  "answer": "Machine learning is a subset of artificial intelligence..."
}
```

### **4. Semantic Search**
- **Endpoint:** `POST /api/search`
- **Request:**
```json
{
  "query": "neural networks",
  "papers": ["paper1", "paper2", "paper3"],
  "top_k": 5
}
```
- **Response:**
```json
{
  "success": true,
  "query": "neural networks",
  "results": [...matched papers...],
  "count": 3
}
```

### **5. Document Upload**
- **Endpoint:** `POST /api/documents/upload`
- **Headers:** `multipart/form-data`
- **Request:** Form data with `file` field containing the document
- **Response:**
```json
{
  "success": true,
  "message": "File uploaded successfully",
  "filename": "paper.pdf",
  "filepath": "data/uploads/paper.pdf"
}
```

---

## Frontend Implementation

### **Using API Service in React Components**

#### **Example 1: Summarization Component**

```jsx
import { generateSummary } from '@/services/api';
import { useState } from 'react';

export function SummarizeComponent() {
  const [text, setText] = useState('');
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSummarize = async () => {
    setLoading(true);
    setError('');
    try {
      const result = await generateSummary(text, {
        maxLength: 200,
        minLength: 50,
      });
      setSummary(result.summary);
    } catch (err) {
      setError(err.message || 'Failed to generate summary');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 border rounded">
      <h2 className="text-2xl font-bold mb-4">Summarize</h2>
      
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste your text here..."
        className="w-full h-32 p-2 border rounded mb-4"
      />
      
      <button
        onClick={handleSummarize}
        disabled={loading}
        className="bg-blue-500 text-white px-4 py-2 rounded disabled:opacity-50"
      >
        {loading ? 'Processing...' : 'Summarize'}
      </button>

      {error && <p className="text-red-500 mt-2">{error}</p>}
      
      {summary && (
        <div className="mt-4 p-3 bg-gray-100 rounded">
          <h3 className="font-semibold">Summary:</h3>
          <p>{summary}</p>
        </div>
      )}
    </div>
  );
}
```

#### **Example 2: Question Answering Component**

```jsx
import { askQuestion } from '@/services/api';
import { useState } from 'react';

export function QAComponent() {
  const [context, setContext] = useState('');
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleAsk = async () => {
    setLoading(true);
    setError('');
    try {
      const result = await askQuestion('doc1', question); // doc1 is example ID
      setAnswer(result.answer);
    } catch (err) {
      setError(err.message || 'Failed to get answer');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 border rounded">
      <h2 className="text-2xl font-bold mb-4">Question Answering</h2>
      
      <textarea
        value={context}
        onChange={(e) => setContext(e.target.value)}
        placeholder="Paste document context..."
        className="w-full h-24 p-2 border rounded mb-4"
      />
      
      <input
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Enter your question..."
        className="w-full p-2 border rounded mb-4"
      />
      
      <button
        onClick={handleAsk}
        disabled={loading}
        className="bg-green-500 text-white px-4 py-2 rounded disabled:opacity-50"
      >
        {loading ? 'Processing...' : 'Ask'}
      </button>

      {error && <p className="text-red-500 mt-2">{error}</p>}
      
      {answer && (
        <div className="mt-4 p-3 bg-gray-100 rounded">
          <h3 className="font-semibold">Answer:</h3>
          <p>{answer}</p>
        </div>
      )}
    </div>
  );
}
```

#### **Example 3: Semantic Search Component**

```jsx
import { searchPapers } from '@/services/api';
import { useState } from 'react';

export function SearchComponent() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSearch = async () => {
    setLoading(true);
    setError('');
    try {
      const result = await searchPapers(query);
      setResults(result.results || []);
    } catch (err) {
      setError(err.message || 'Failed to search');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 border rounded">
      <h2 className="text-2xl font-bold mb-4">Semantic Search</h2>
      
      <div className="flex gap-2 mb-4">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
          placeholder="Search papers..."
          className="flex-1 p-2 border rounded"
        />
        <button
          onClick={handleSearch}
          disabled={loading}
          className="bg-purple-500 text-white px-4 py-2 rounded disabled:opacity-50"
        >
          {loading ? 'Searching...' : 'Search'}
        </button>
      </div>

      {error && <p className="text-red-500">{error}</p>}
      
      {results.length > 0 && (
        <div className="space-y-2">
          {results.map((result, idx) => (
            <div key={idx} className="p-3 bg-gray-100 rounded">
              <p className="font-semibold">{result.title || `Result ${idx + 1}`}</p>
              <p className="text-sm text-gray-600">{result.excerpt || result.text}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

---

## Update API Service

Your current `api.js` needs to be aligned with Flask endpoints. Update these calls:

```javascript
// BEFORE (wrong endpoint)
export const askQuestion = async (documentId, question) => {
  const response = await apiClient.post(`/question-answer`, {...});
};

// AFTER (correct Flask endpoint)
export const askQuestion = async (question, context) => {
  const response = await apiClient.post(`/answer`, { question, context });
};
```

---

## Environment Configuration

Your `.env` is already set correctly:

```env
VITE_API_URL=http://localhost:5000/api
```

For production, update to your server URL:

```env
VITE_API_URL=https://your-backend-domain.com/api
```

---

## CORS Configuration

Your Flask backend already has CORS enabled in `app.py`:

```python
from flask_cors import CORS
CORS(app)  # Allow requests from any origin
```

For production, restrict CORS to your frontend domain:

```python
CORS(app, resources={r"/api/*": {"origins": "https://your-frontend-domain.com"}})
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Connection Refused** | Ensure Flask is running on port 5000 |
| **CORS Error** | Verify `flask_cors` is installed: `pip install flask-cors` |
| **404 Errors** | Check endpoint URLs match Flask router |
| **Timeout Errors** | Redux heavy models may need longer timeout (increase to 60s) |
| **File Upload Failed** | Ensure `data/uploads` folder exists in backend |

---

## File Structure Reference

```
Backend (Python)
├── app.py (Flask server with API endpoints)
├── src/
│   ├── model.py (ResearchPaperQASystem)
│   ├── retrieval.py (Semantic search)
│   └── utils_functions.py
└── requirements.txt

Frontend (React)
├── src/
│   ├── services/api.js (API calls)
│   ├── components/ (React components)
│   └── hooks/ (Custom hooks)
└── .env (API URL configuration)
```

---

## Quick Commands

```bash
# Backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py

# Frontend
npm install
npm run dev

# Check connection
curl -v http://localhost:5000/api/health
```
