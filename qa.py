from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
# from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_groq import ChatGroq
from langchain_core.embeddings import Embeddings
import cohere
from dotenv import load_dotenv
import os
import re
import sys

load_dotenv()

DB_DIR = "vectorstore"
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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
    return re.sub(r"<think>.*?</think>", "", text, flags= re.DOTALL).strip()


def load_llm():
    print("Loading qroq Model...")

    llm = ChatGroq(model_name='llama3-70b-8192')

    return llm

# Load vector store and run QA
def ask_question(query: str, llm: LLM) -> str:
    print("[*] Loading vector store...")
    embeddings = CohereEmbeddings(api_key=COHERE_API_KEY)
    vectordb = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    print("[*] Retrieving documents for the query...")
    docs = retriever.get_relevant_documents(query)
    print(f"[*] Retrieved {len(docs)} documents:")
    for i, doc in enumerate(docs):
        snippet = doc.page_content[:200].replace('\n', ' ')
        print(f"  Document {i+1} snippet: {snippet}...")

    print("[*] Running QA...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    result = qa_chain.invoke({"query": query})
    return result

# CLI Loop
if __name__ == "__main__":
    print("Loading groq model...")
    try:
        llm = load_llm()
    except Exception as e:
        print(f"[!] Error loading LLM: {e}")
        sys.exit(1)

    try:
        while True:
            query = input("Ask a question (or 'exit'): ")
            if query.lower() == "exit":
                break
            answer_dict = ask_question(query, llm)
            answer = answer_dict["result"]
            clean_output = remove_think_tags(answer)
            print(f"AI: {clean_output}\n")
    except KeyboardInterrupt:
        print("\n[!] Exiting on user interrupt.")
        sys.exit(0)
