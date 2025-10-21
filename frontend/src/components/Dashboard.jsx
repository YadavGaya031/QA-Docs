// frontend/src/components/Dashboard.jsx
import React, { useState } from 'react';
import { uploadFile, ingest, ask } from '../services/api';
import './Dashboard.css';

export default function Dashboard() {
  const [file, setFile] = useState(null);
  const [answer, setAnswer] = useState('');
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState({ upload: false, ingest: false, ask: false });
  const [message, setMessage] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState([]);

  async function handleUpload(e) {
    e.preventDefault();
    if (!file) return;
    setLoading(prev => ({ ...prev, upload: true }));
    setMessage('');

    try {
      const data = await uploadFile(file);
      setMessage(`✅ File "${file.name}" uploaded successfully!`);
      setUploadedFiles(prev => [...prev, file.name]);
      setFile(null);
      // Reset file input
      e.target.reset();
    } catch (err) {
      console.error(err);
      setMessage('❌ Upload failed. Please try again.');
    } finally {
      setLoading(prev => ({ ...prev, upload: false }));
    }
  }

  async function handleIngest() {
    setLoading(prev => ({ ...prev, ingest: true }));
    setMessage('');

    try {
      const data = await ingest();
      setMessage('✅ Vectorstore built successfully! You can now ask questions.');
    } catch (err) {
      console.error(err);
      setMessage('❌ Ingest failed. Make sure you have uploaded documents.');
    } finally {
      setLoading(prev => ({ ...prev, ingest: false }));
    }
  }

  async function handleAsk(e) {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(prev => ({ ...prev, ask: true }));
    setAnswer('');

    try {
      const res = await ask(query);
      setAnswer(res.answer || 'No answer received.');
    } catch (err) {
      console.error(err);
      setAnswer('❌ Failed to get answer. Make sure vectorstore is built.');
    } finally {
      setLoading(prev => ({ ...prev, ask: false }));
    }
  }

  return (
    <div className="dashboard">
      <div className="dashboard-section">
        <h2>Upload Documents</h2>
        <p>Upload PDF or TXT files to build your knowledge base.</p>
        <form onSubmit={handleUpload} className="upload-form">
          <input
            type="file"
            accept=".pdf,.txt"
            onChange={e => setFile(e.target.files[0])}
            disabled={loading.upload}
          />
          <button type="submit" disabled={loading.upload || !file}>
            {loading.upload ? 'Uploading...' : 'Upload'}
          </button>
        </form>
        {uploadedFiles.length > 0 && (
          <div className="uploaded-files">
            <h3>Uploaded Files:</h3>
            <ul>
              {uploadedFiles.map((fileName, index) => (
                <li key={index}>{fileName}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <div className="dashboard-section">
        <h2>Build Knowledge Base</h2>
        <p>Process your uploaded documents to create a searchable vectorstore.</p>
        <button
          onClick={handleIngest}
          disabled={loading.ingest}
          className="ingest-btn"
        >
          {loading.ingest ? 'Building...' : 'Build Vectorstore'}
        </button>
      </div>

      <div className="dashboard-section">
        <h2>Ask Questions</h2>
        <p>Ask questions about your documents.</p>
        <form onSubmit={handleAsk} className="ask-form">
          <input
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Enter your question..."
            disabled={loading.ask}
          />
          <button type="submit" disabled={loading.ask || !query.trim()}>
            {loading.ask ? 'Asking...' : 'Ask'}
          </button>
        </form>
        {answer && (
          <div className="answer-section">
            <h3>Answer:</h3>
            <div className="answer-content">{answer}</div>
          </div>
        )}
      </div>

      {message && (
        <div className={`message-banner ${message.includes('✅') ? 'success' : 'error'}`}>
          {message}
        </div>
      )}
    </div>
  );
}
