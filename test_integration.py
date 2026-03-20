"""
Integration Test - Verify Backend & Frontend Connection
Tests all API endpoints to ensure full connectivity
"""

import requests
import json
import time

BACKEND_URL = "http://localhost:5000/api"
FRONTEND_URL = "http://localhost:3001"

print("\n" + "="*80)
print("🔗 BACKEND & FRONTEND INTEGRATION TEST")
print("="*80)

# Test 1: Backend Health Check
print("\n[1/5] Testing Backend Health Check...")
try:
    response = requests.get(f"{BACKEND_URL}/health", timeout=5)
    if response.status_code == 200:
        print(f"    ✅ Backend is HEALTHY")
        print(f"    Status: {response.json()['status']}")
    else:
        print(f"    ❌ Backend returned status {response.status_code}")
except Exception as e:
    print(f"    ❌ Failed to connect: {e}")

# Test 2: Summarization Endpoint
print("\n[2/5] Testing Summarization Endpoint...")
try:
    test_text = "Machine learning is a subset of artificial intelligence that enables systems to learn from data. Deep learning uses neural networks with multiple layers. These models can process vast amounts of unstructured data efficiently."
    payload = {
        "text": test_text,
        "max_length": 100,
        "min_length": 30
    }
    response = requests.post(f"{BACKEND_URL}/summarize", json=payload, timeout=10)
    if response.status_code == 200:
        result = response.json()
        print(f"    ✅ Summarization works!")
        print(f"    Summary: {result['summary'][:100]}...")
    else:
        print(f"    ❌ Status {response.status_code}: {response.text}")
except Exception as e:
    print(f"    ❌ Error: {e}")

# Test 3: Question Answering Endpoint
print("\n[3/5] Testing Question Answering Endpoint...")
try:
    payload = {
        "question": "What is machine learning?",
        "context": "Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed."
    }
    response = requests.post(f"{BACKEND_URL}/answer", json=payload, timeout=10)
    if response.status_code == 200:
        result = response.json()
        print(f"    ✅ Question Answering works!")
        print(f"    Answer: {result['answer'][:80]}...")
    else:
        print(f"    ❌ Status {response.status_code}: {response.text}")
except Exception as e:
    print(f"    ❌ Error: {e}")

# Test 4: Semantic Search Endpoint
print("\n[4/5] Testing Semantic Search Endpoint...")
try:
    payload = {
        "query": "neural networks",
        "papers": [],
        "top_k": 5
    }
    response = requests.post(f"{BACKEND_URL}/search", json=payload, timeout=10)
    if response.status_code == 200:
        result = response.json()
        print(f"    ✅ Semantic Search works!")
        print(f"    Query: {result['query']}")
        print(f"    Results found: {result['count']}")
    else:
        print(f"    ❌ Status {response.status_code}: {response.text}")
except Exception as e:
    print(f"    ❌ Error: {e}")

# Test 5: Frontend Connectivity
print("\n[5/5] Testing Frontend Connectivity...")
try:
    response = requests.get(FRONTEND_URL, timeout=5)
    if response.status_code == 200:
        print(f"    ✅ Frontend is running!")
        print(f"    URL: {FRONTEND_URL}")
    else:
        print(f"    ⚠️  Frontend status: {response.status_code}")
except Exception as e:
    print(f"    ❌ Frontend Error: {e}")

# Summary
print("\n" + "="*80)
print("✅ INTEGRATION TEST COMPLETE!")
print("="*80)
print("\n📌 NEXT STEPS:")
print("  1. Open browser: http://localhost:3001")
print("  2. Use the Integration Dashboard to test all features")
print("  3. Check browser console for any CORS or API errors")
print("\n🔗 URLS:")
print(f"  Frontend: http://localhost:3001")
print(f"  Backend:  http://localhost:5000")
print(f"  Backend API Docs: http://localhost:5000/api/health")
print("="*80 + "\n")
