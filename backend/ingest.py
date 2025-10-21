# backend/ingest.py
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv
import cohere

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

class CohereEmbeddings(Embeddings):
    def __init__(self, api_key, model="embed-english-v3.0"):
        self.client = cohere.Client(api_key)
        self.model = model

    def embed_documents(self, texts):
        response = self.client.embed(texts=texts, model=self.model, input_type="search_document")
        return response.embeddings

    def embed_query(self, text):
        response = self.client.embed(texts=[text], model=self.model, input_type="search_query")
        return response.embeddings[0]

def load_documents_from_path(path):
    documents = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            continue
        docs = loader.load()
        documents.extend(docs)
    return documents

def ingest_for_user(user_id: str, docs_path: str, db_dir_root="vectorstore"):
    """
    docs_path: folder containing files for the user (uploaded).
    creates vectorstore at: db_dir_root/<user_id>/
    """
    os.makedirs(db_dir_root, exist_ok=True)
    user_db_dir = os.path.join(db_dir_root, user_id)
    # create user folder
    os.makedirs(user_db_dir, exist_ok=True)

    documents = load_documents_from_path(docs_path)
    if not documents:
        raise ValueError("No documents to ingest in path: " + docs_path)

    # split
    chunk_size = 4000
    chunk_overlap = 100
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    chunks = splitter.split_documents(documents)
    embeddings = CohereEmbeddings(api_key=COHERE_API_KEY)
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(user_db_dir)
    return user_db_dir
