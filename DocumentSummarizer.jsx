import React, { useState, useRef } from 'react';

/**
 * Document Summarizer Component
 * 
 * This component handles:
 * - File upload (DOCX, PDF, TXT)
 * - BART model summarization
 * - Display of extracted text and summary
 * - Statistics (compression ratio, character counts)
 * 
 * Uses the new /api/summarize-file endpoint
 */
export function DocumentSummarizer() {
  const [summary, setSummary] = useState('');
  const [extractedText, setExtractedText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [stats, setStats] = useState(null);
  const [fileName, setFileName] = useState('');
  const fileInputRef = useRef(null);

  /**
   * Handle file selection and summarization
   * ✅ CORRECT WAY - Uses /api/summarize-file endpoint
   */
  const handleFileUpload = async (file) => {
    if (!file) return;

    setLoading(true);
    setError('');
    setSuccess('');
    setSummary('');
    setExtractedText('');
    setStats(null);
    setFileName(file.name);

    try {
      // Step 1: Create FormData with the file
      const formData = new FormData();
      formData.append('file', file);

      console.log('📤 Uploading file:', file.name);

      // Step 2: Send to the new /api/summarize-file endpoint
      // ✅ THIS IS THE CORRECT ENDPOINT
      const response = await fetch('http://localhost:5000/api/summarize-file', {
        method: 'POST',
        body: formData
      });

      // Step 3: Check for errors
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to summarize file');
      }

      // Step 4: Parse the response
      const result = await response.json();

      console.log('✅ Successfully summarized!');
      console.log('Extracted text:', result.extracted_text.substring(0, 100) + '...');
      console.log('Summary:', result.summary.substring(0, 100) + '...');

      // Step 5: Display results
      setExtractedText(result.extracted_text);
      setSummary(result.summary);
      setStats({
        originalLength: result.original_length,
        summaryLength: result.summary_length,
        compressionRatio: result.compression_ratio
      });

      setSuccess(`✓ Successfully summarized "${file.name}"`);

    } catch (err) {
      console.error('❌ Error:', err);
      setError(`Error: ${err.message}`);
      setSummary('');
      setExtractedText('');
    } finally {
      setLoading(false);
    }
  };

  // Handle file input change
  const handleFileInputChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  // Handle drag and drop
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  return (
    <div className="document-summarizer">
      <style>{`
        .document-summarizer {
          max-width: 900px;
          margin: 0 auto;
          padding: 20px;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
        }

        .header {
          margin-bottom: 30px;
        }

        .header h1 {
          font-size: 32px;
          margin-bottom: 10px;
          color: #333;
        }

        .header p {
          color: #666;
          font-size: 16px;
        }

        .upload-section {
          background: #f8f9fa;
          border: 2px dashed #667eea;
          border-radius: 8px;
          padding: 40px;
          text-align: center;
          cursor: pointer;
          transition: all 0.3s ease;
          margin-bottom: 20px;
        }

        .upload-section:hover {
          border-color: #764ba2;
          background: #f0f2ff;
        }

        .upload-section.dragover {
          border-color: #764ba2;
          background: #e8ecff;
        }

        .upload-text {
          font-size: 18px;
          color: #333;
          margin-bottom: 10px;
          font-weight: 600;
        }

        .upload-hint {
          color: #999;
          font-size: 14px;
          margin-bottom: 20px;
        }

        input[type="file"] {
          display: none;
        }

        .upload-button {
          background: #667eea;
          color: white;
          border: none;
          padding: 12px 30px;
          border-radius: 6px;
          cursor: pointer;
          font-size: 16px;
          font-weight: 600;
          transition: all 0.3s ease;
        }

        .upload-button:hover {
          background: #764ba2;
          transform: translateY(-2px);
          box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }

        .upload-button:disabled {
          background: #ccc;
          cursor: not-allowed;
          transform: none;
        }

        .message {
          padding: 15px;
          border-radius: 6px;
          margin-bottom: 20px;
          display: none;
        }

        .message.show {
          display: block;
        }

        .error {
          background: #fff3cd;
          border: 1px solid #ffc107;
          color: #856404;
        }

        .success {
          background: #d4edda;
          border: 1px solid #28a745;
          color: #155724;
        }

        .loading {
          text-align: center;
          color: #667eea;
          font-weight: 600;
          padding: 20px;
        }

        .spinner {
          display: inline-block;
          width: 20px;
          height: 20px;
          border: 3px solid rgba(102, 126, 234, 0.3);
          border-radius: 50%;
          border-top-color: #667eea;
          animation: spin 0.8s linear infinite;
          margin-right: 10px;
          vertical-align: middle;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .results-section {
          display: none;
        }

        .results-section.show {
          display: block;
        }

        .text-box {
          background: #f0f7ff;
          border-left: 4px solid #667eea;
          padding: 20px;
          border-radius: 6px;
          margin-bottom: 20px;
        }

        .text-title {
          color: #667eea;
          font-weight: 600;
          margin-bottom: 10px;
          text-transform: uppercase;
          font-size: 12px;
          letter-spacing: 1px;
        }

        .text-content {
          color: #333;
          line-height: 1.6;
          font-size: 15px;
          max-height: 300px;
          overflow-y: auto;
          word-break: break-word;
        }

        .stats {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 15px;
          margin-top: 20px;
        }

        .stat-box {
          background: #fff;
          border: 1px solid #e0e0e0;
          padding: 15px;
          border-radius: 6px;
          text-align: center;
        }

        .stat-value {
          color: #667eea;
          font-size: 24px;
          font-weight: bold;
          margin-bottom: 5px;
        }

        .stat-label {
          color: #666;
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .tabs {
          display: flex;
          gap: 10px;
          margin-bottom: 20px;
          border-bottom: 2px solid #e0e0e0;
        }

        .tab {
          padding: 10px 20px;
          background: none;
          border: none;
          color: #999;
          cursor: pointer;
          font-weight: 600;
          border-bottom: 2px solid transparent;
          transition: all 0.3s ease;
        }

        .tab:hover {
          color: #667eea;
        }

        .tab.active {
          color: #667eea;
          border-bottom-color: #667eea;
        }

        .tab-content {
          display: none;
        }

        .tab-content.active {
          display: block;
        }
      `}</style>

      <div className="header">
        <h1>Upload & Summarize</h1>
        <p>Upload your research papers and get instant intelligent summaries using BART</p>
      </div>

      {error && (
        <div className={`message error show`}>
          {error}
        </div>
      )}

      {success && (
        <div className={`message success show`}>
          {success}
        </div>
      )}

      <div
        className="upload-section"
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".docx,.pdf,.txt"
          onChange={handleFileInputChange}
        />
        <div className="upload-text">📁 Click to upload or drag & drop</div>
        <div className="upload-hint">Supported formats: DOCX, PDF, TXT (Max 50MB)</div>
        <button className="upload-button" disabled={loading}>
          {loading ? '⏳ Processing...' : 'Choose File'}
        </button>
      </div>

      {loading && (
        <div className="loading">
          <span className="spinner"></span>
          Processing your document...
        </div>
      )}

      {summary && !loading && (
        <div className="results-section show">
          <div className="tabs">
            <button className="tab active" onClick={(e) => {
              e.target.classList.add('active');
              document.querySelector('.tab-content.summary').classList.add('active');
              document.querySelector('.tab-content.extracted').classList.remove('active');
              document.querySelectorAll('.tab').forEach((t, i) => {
                if (i !== 0) t.classList.remove('active');
              });
            }}>
              📄 Summary
            </button>
            <button className="tab" onClick={(e) => {
              e.target.classList.add('active');
              document.querySelector('.tab-content.extracted').classList.add('active');
              document.querySelector('.tab-content.summary').classList.remove('active');
              document.querySelectorAll('.tab').forEach((t, i) => {
                if (i !== 1) t.classList.remove('active');
              });
            }}>
              📋 Extracted Text
            </button>
          </div>

          <div className="tab-content summary active">
            <div className="text-box">
              <div className="text-title">✅ BART Generated Summary</div>
              <div className="text-content">
                {summary}
              </div>
            </div>
          </div>

          <div className="tab-content extracted">
            <div className="text-box">
              <div className="text-title">📋 Original Extracted Text</div>
              <div className="text-content">
                {extractedText}
              </div>
            </div>
          </div>

          {stats && (
            <div className="stats">
              <div className="stat-box">
                <div className="stat-value">{stats.originalLength.toLocaleString()}</div>
                <div className="stat-label">Original Characters</div>
              </div>
              <div className="stat-box">
                <div className="stat-value">{stats.summaryLength.toLocaleString()}</div>
                <div className="stat-label">Summary Characters</div>
              </div>
              <div className="stat-box">
                <div className="stat-value">{stats.compressionRatio.toFixed(2)}%</div>
                <div className="stat-label">Compression Rate</div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default DocumentSummarizer;
