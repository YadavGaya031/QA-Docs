# backend/app.py
import os
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from bson.objectid import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient
import shutil
from dotenv import load_dotenv
from auth_utils import hash_password, verify_password, create_access_token, decode_access_token
from ingest import ingest_for_user
from qa_core import ask_question_for_user
from langchain_groq import ChatGroq

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "qa_app")
UPLOAD_ROOT = os.getenv("UPLOAD_ROOT", "uploads")
VECTORSTORE_ROOT = os.getenv("VECTORSTORE_ROOT", "vectorstore")

# Ensure folders exist
os.makedirs(UPLOAD_ROOT, exist_ok=True)
os.makedirs(VECTORSTORE_ROOT, exist_ok=True)

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
users_col = db["users"]   # store {username, email, password_hash, _id}

app = FastAPI(title="QA App with Auth")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5173", "http://127.0.0.1:5174", "*"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# Load LLM once at startup
LLM = None
@app.on_event("startup")
def startup():
    global LLM
    try:
        LLM = ChatGroq(model_name="groq/compound")
    except Exception as e:
        # log but keep server up (endpoints will return 503)
        print("Failed to load LLM at startup:", e)
        LLM = None

# ------------------------
# Models
# ------------------------
class RegisterModel(BaseModel):
    username: str
    email: str
    password: str

class LoginModel(BaseModel):
    username: str
    password: str

# ------------------------
# Auth helpers
# ------------------------
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

    user_id_str = payload.get("user_id")
    if not user_id_str:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    try:
        user_obj_id = ObjectId(user_id_str)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=401, detail="Invalid user id in token")

    user = users_col.find_one({"_id": user_obj_id})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    print("token payload: ", payload)
    print("looking up user id: ", user_id_str )
    return user



# ------------------------
# Routes: auth
# ------------------------
@app.post("/auth/register")
def register(data: RegisterModel):
    if users_col.find_one({"username": data.username}):
        raise HTTPException(status_code=400, detail="username already exists")
    hashed = hash_password(data.password)
    result = users_col.insert_one({"username": data.username, "email": data.email, "password": hashed})
    user_id = str(result.inserted_id)
    # create user upload folder
    os.makedirs(os.path.join(UPLOAD_ROOT, user_id), exist_ok=True)
    return {"user_id": user_id, "username": data.username}

@app.post("/auth/login")
def login(data: LoginModel):
    user = users_col.find_one({"username": data.username})
    if not user:
        raise HTTPException(status_code=400, detail="invalid username or password")
    if not verify_password(data.password, user["password"]):
        raise HTTPException(status_code=400, detail="invalid username or password")
    token = create_access_token({"user_id": str(user["_id"]), "username": user["username"]})
    return {"access_token": token}

# ------------------------
# Routes: upload + ingest + ask
# ------------------------
@app.post("/upload")
def upload_file(file: UploadFile = File(...), user = Depends(get_current_user)):
    user_id = str(user["_id"])
    user_dir = os.path.join(UPLOAD_ROOT, user_id)
    os.makedirs(user_dir, exist_ok=True)
    dst_path = os.path.join(user_dir, file.filename)
    with open(dst_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # record metadata in DB
    users_col.update_one({"_id": user["_id"]}, {"$push": {"files": {"filename": file.filename}}})
    return {"status": "ok", "filename": file.filename}

@app.post("/ingest")
def ingest_endpoint(user = Depends(get_current_user)):
    user_id = str(user["_id"])
    user_docs_path = os.path.join(UPLOAD_ROOT, user_id)
    if not os.path.isdir(user_docs_path):
        raise HTTPException(status_code=400, detail="No uploaded docs for this user.")
    try:
        out_dir = ingest_for_user(user_id=user_id, docs_path=user_docs_path, db_dir_root=VECTORSTORE_ROOT)
        return {"status": "ingested", "vectorstore_dir": out_dir}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AskRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_endpoint(request: AskRequest, user = Depends(get_current_user)):
    if LLM is None:
        raise HTTPException(status_code=503, detail="LLM not ready")
    user_id = str(user["_id"])
    try:
        answer = ask_question_for_user(request.query, LLM, user_id)
        return {"query": request.query, "answer": answer}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Vectorstore not found for user. Run /ingest.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
