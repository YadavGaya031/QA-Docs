# Document Question Answering

This project provides a Document Question Answering system that allows users to ask questions about documents and get answers using a combination of vector search and large language models. It includes both a command-line interface (CLI) and a web-based user interface built with Streamlit.

## Features

- Load and embed documents (PDF and TXT) into a FAISS vector store.
- Use Cohere embeddings and Groq LLM for question answering.
- Query documents via CLI or Streamlit web app.
- Clean output by removing internal tags.

## Installation

1. Clone the repository.

2. Create and activate a Python virtual environment (recommended):

```bash
python -m venv venv
.\venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Linux/macOS
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your API keys:

```
COHERE_API_KEY=your_cohere_api_key
GROQ_API_KEY=your_groq_api_key
```

## Usage

### Document Ingestion

Place your PDF and TXT documents in the `docs/` directory.

Run the ingestion script to build the vector store:

```bash
python ingest.py
```

### Command Line Interface (CLI)

Run the QA system in CLI mode:

```bash
python qa.py
```

Type your questions and get answers. Type `exit` to quit.

### Web Interface

Run the Streamlit app:

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually http://localhost:8501) to access the web UI.

## Project Structure

```
.
├── app.py              # Streamlit web app
├── ingest.py           # Document ingestion and vector store creation
├── qa.py               # Core QA logic and CLI interface
├── requirements.txt    # Python dependencies
├── docs/               # Directory for input documents (PDF, TXT)
└── vectorstore/        # Saved FAISS vector store files
```

## Environment Variables

- `COHERE_API_KEY`: API key for Cohere embeddings.
- `GROQ_API_KEY`: API key for Groq LLM.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
