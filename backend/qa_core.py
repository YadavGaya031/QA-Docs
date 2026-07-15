import os

from dotenv import load_dotenv
from pymongo import MongoClient
from google import genai
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

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


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
# Retrieve Relevant Chunks
# -----------------------------
def retrieve_chunks(user_id: str, query: str, k: int = 5):

    query_embedding = generate_embedding(query)
    pipeline = [
        # {
        #     "$vectorSearch": {
        #         "index": "vector_index",
        #         "path": "embedding",
        #         "queryVector": query_embedding,
        #         "numCandidates": 100,
        #         "limit": k,
        #         # "filter": {"userId": user_id},
        #         "filter": {"userId": {"$eq": user_id}},
        #     }
        # },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "filename": 1,
                "page": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    
    return list(chunks_collection.aggregate(pipeline))


# -----------------------------
# Build Prompt
# -----------------------------
def build_prompt(query: str, docs):

    context = ""

    for doc in docs:

        context += (
            f"\nDocument: {doc['filename']}"
            f"\nPage: {doc['page']}"
            f"\nContent:\n{doc['text']}\n"
        )

    prompt = f"""
You are an AI assistant.

Answer ONLY using the information provided in the context below.

If the answer is not present in the context, reply:

"I couldn't find the answer in the uploaded documents."

Context:
{context}

Question:
{query}

Give a clear and concise answer.
"""

    return prompt


# -----------------------------
# Ask Question
# -----------------------------
def ask_question_for_user(query: str, user_id: str):

    docs = retrieve_chunks(user_id, query)
    
    if len(docs) == 0:
        return {"answer": "No relevant information found.", "sources": []}

    prompt = build_prompt(query, docs)
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    answer = response.candidates[0].content.parts[0].text

    
    return {"answer": answer}
