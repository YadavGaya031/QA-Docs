# backend/qa_core.py
import os, re
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain_core.embeddings import Embeddings
import cohere
from dotenv import load_dotenv

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
DB_DIR_ROOT = "vectorstore"

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

def remove_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def ask_question_for_user(query: str, llm: LLM, user_id: str):
    user_db_dir = os.path.join(DB_DIR_ROOT, user_id)
    if not os.path.isdir(user_db_dir):
        raise FileNotFoundError("Vectorstore for user not found; run ingest for this user.")

    embeddings = CohereEmbeddings(api_key=COHERE_API_KEY)
    vectordb = FAISS.load_local(user_db_dir, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    result = qa_chain.invoke({"query": query})
    raw = result["result"] if isinstance(result, dict) and "result" in result else str(result)
    return remove_think_tags(raw)
