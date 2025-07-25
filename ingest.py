import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv
import cohere
import numpy as np

load_dotenv()

DATA_DIR = "docs"
DB_DIR = "vectorstore"
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Custom wrapper for Cohere embeddings to be compatible with LangChain
class CohereEmbeddings(Embeddings):
    def __init__(self, api_key, model="embed-english-v3.0"):
        self.client = cohere.Client(api_key)
        self.model = model

    def embed_documents(self, texts):
        """
        Embed a list of texts (documents) and return list of vectors.
        """
        response = self.client.embed(texts=texts, model=self.model, input_type="search_document")
        return response.embeddings

    def embed_query(self, text):
        """
        Embed a single query string.
        """
        response = self.client.embed(texts=[text], model=self.model, input_type="search_query")
        return response.embeddings[0]


def load_documents():
    documents = []
    print("[*] Loading documents from directory:", DATA_DIR)
    if not os.path.exists(DATA_DIR):
        print(f"[!] Directory '{DATA_DIR}' does not exist.")
        return documents
    try:
        for file in os.listdir(DATA_DIR):
            file_path = os.path.join(DATA_DIR, file)
            if file.endswith(".pdf"):
                print(f"[*] Loading PDF file: {file}")
                loader = PyPDFLoader(file_path)
            elif file.endswith(".txt"):
                print(f"[*] Loading TXT file: {file}")
                loader = TextLoader(file_path)
            else:
                print(f"[!] Skipping unsupported file: {file}")
                continue
            loaded_docs = loader.load()
            print(f"    Loaded {len(loaded_docs)} documents from {file}")
            documents.extend(loaded_docs)
    except Exception as e:
        print(f"[!] Error loading documents: {e}")
    print(f"[*] Total documents loaded: {len(documents)}")
    return documents

def split_documents(documents):
    # Suggest experimenting with chunk_size and chunk_overlap here
    chunk_size = 4000
    chunk_overlap = 100
    print(f"[*] Splitting documents with chunk_size={chunk_size} and chunk_overlap={chunk_overlap}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = splitter.split_documents(documents)
    print(f"[*] Total chunks created: {len(chunks)}")
    return chunks

def ingest():
    print("[*] Starting ingestion process...")
    documents = load_documents()
    if not documents:
        print("[!] No documents found. Please check the 'docs' directory.")
        return

    chunks = split_documents(documents)

    print("[*] Embedding documents...")
    embeddings = CohereEmbeddings(api_key=COHERE_API_KEY)

    print("[*] Building vector store...")
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(DB_DIR)
    print(f"[✓] Vector store saved to '{DB_DIR}'.")

if __name__ == "__main__":
    ingest()
