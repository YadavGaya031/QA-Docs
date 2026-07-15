
import os

from dotenv import load_dotenv
from pymongo import MongoClient

from google import genai
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import certifi

load_dotenv()

# -----------------------------
# Config
# -----------------------------
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB", "qa_app")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# client = MongoClient(MONGO_URI)
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client[DB_NAME]

chunks_collection = db["chunks"]
documents_collection = db["documents"]

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# -----------------------------
# Load Documents
# -----------------------------
def load_documents_from_path(path: str):
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

        for doc in docs:
            doc.metadata["filename"] = file

        documents.extend(docs)

    return documents


# -----------------------------
# Gemini Embedding
# -----------------------------
# def generate_embedding(text: str):

#     response = gemini_client.models.embed_content(
#         model="gemini-embedding-001",
#         contents=text,
#     )

#     return response.embeddings[0].values

def generate_embedding(text: str) -> list[float]:
    response = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )

    return response.embeddings[0].values


# -----------------------------
# Ingest
# -----------------------------
def ingest_for_user(user_id: str, docs_path: str):

    documents = load_documents_from_path(docs_path)

    if not documents:
        raise ValueError("No documents found.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)
    inserted = 0
    chunks_collection.delete_many({"userId": user_id})
    for idx, chunk in enumerate(chunks):

        embedding = generate_embedding(chunk.page_content)

        doc = {
            "userId": user_id,
            "filename": chunk.metadata.get("filename"),
            "page": chunk.metadata.get("page", 0),
            "chunkIndex": idx,
            "text": chunk.page_content,
            "embedding": embedding,
        }

        chunks_collection.insert_one(doc)

        inserted += 1

    return {
        "documents": len(documents),
        "chunks": inserted,
    }
