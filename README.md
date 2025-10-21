# LLM Document Processing System

A full-stack web application that uses Large Language Models (LLMs) to process natural language queries and retrieve relevant information from unstructured documents such as policy documents, contracts, and emails. Inspired by real-world challenges in insurance, legal compliance, and contract management.

## Objective

The system takes input queries like "46-year-old male, knee surgery in Pune, 3-month-old insurance policy" and:

- Parses and structures the query to identify key details (age, procedure, location, policy duration).
- Searches and retrieves relevant clauses using semantic understanding.
- Evaluates information to determine decisions (approval/rejection, payout amounts) based on document logic.
- Returns structured JSON responses with Decision, Amount, and Justification, including clause mappings.

## Features

- **User Authentication**: Secure registration and login with JWT tokens.
- **Document Upload**: Support for PDFs, Word files, and emails with user-specific storage.
- **Vectorstore Ingestion**: Automated processing of documents into searchable vector embeddings.
- **Semantic Q&A**: Intelligent question answering using LLMs for accurate, context-aware responses.
- **Structured Outputs**: JSON responses with decisions, amounts, and justifications referencing source clauses.
- **Scalable Architecture**: User-isolated data with MongoDB and vectorstores.
- **CORS Enabled**: Seamless frontend-backend communication.

## Tech Stack

- **Backend**: FastAPI, MongoDB, LangChain, Groq LLM
- **Frontend**: React, Vite
- **Authentication**: JWT
- **AI/ML**: Vector embeddings, semantic search
- **Deployment**: Ready for containerization (Docker not included)

## Prerequisites

- Python 3.8+
- Node.js 16+
- MongoDB (local or cloud instance)
- Groq API key (for LLM integration)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd qa-doc2
   ```

2. **Backend Setup**:
   - Navigate to backend directory:
     ```bash
     cd backend
     ```
   - Install Python dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Set up environment variables (create `.env` file):
     ```
     MONGO_URI=mongodb://localhost:27017
     DB_NAME=qa_app
     UPLOAD_ROOT=uploads
     VECTORSTORE_ROOT=vectorstore
     GROQ_API_KEY=your-groq-api-key
     COHERE_API_KEY=your-cohere-api-key
     ```

3. **Frontend Setup**:
   - Navigate to frontend directory:
     ```bash
     cd ../frontend
     ```
   - Install Node dependencies:
     ```bash
     npm install
     ```

## Running the Application

1. **Start MongoDB** (if running locally):
   ```bash
   mongod
   ```

2. **Start Backend**:
   - From backend directory:
     ```bash
     python app.py
     ```
   - Server runs on `http://localhost:8000`

3. **Start Frontend**:
   - From frontend directory:
     ```bash
     npm run dev
     ```
   - App runs on `http://localhost:5173`

## Usage

1. **Register/Login**: Create an account or log in.
2. **Upload Documents**: Upload policy PDFs or documents via the dashboard.
3. **Ingest Documents**: Click "Ingest" to process documents into a vectorstore.
4. **Ask Questions**: Enter queries like "46M, knee surgery, Pune, 3-month policy" and receive structured responses.

Example Query: "Is maternity coverage included for a 28-year-old female with a 2-year policy?"

Example Response:
```json
{
  "Decision": "Approved",
  "Amount": "Covered up to ₹2,00,000",
  "Justification": "Clause 3.1 provides maternity benefits for policies over 1 year."
}
```

## API Endpoints

### Authentication
- `POST /auth/register`: Register a new user.
- `POST /auth/login`: Login and receive JWT token.

### Document Operations
- `POST /upload`: Upload a file (requires auth).
- `POST /ingest`: Ingest uploaded documents into vectorstore (requires auth).
- `POST /ask`: Ask a question and get response (requires auth).

All endpoints require Bearer token in Authorization header.

## Project Structure

```
qa-doc2/
├── backend/
│   ├── app.py              # Main FastAPI app
│   ├── auth_utils.py       # JWT utilities
│   ├── qa_core.py          # Q&A logic
│   ├── ingest.py           # Document ingestion
│   ├── requirements.txt    # Python deps
│   ├── uploads/            # User uploads
│   └── vectorstore/        # Vectorstores
├── frontend/
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── services/       # API client
│   │   └── App.jsx         # Main app
│   ├── package.json        # Node deps
│   └── vite.config.js      # Vite config
└── README.md
```

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Make changes and test thoroughly.
4. Submit a pull request.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Inspired by hackathon challenges in AI-driven document processing for insurance and legal domains.
