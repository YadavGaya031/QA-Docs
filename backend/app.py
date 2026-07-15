# backend/app.py
import os
import shutil

from bson.errors import InvalidId
from bson.objectid import ObjectId
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from pymongo import MongoClient

from auth_utils import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from ingest import ingest_for_user
from qa_core import ask_question_for_user
import certifi

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB", "qa_app")
UPLOAD_ROOT = os.getenv("UPLOAD_ROOT", "uploads")

os.makedirs(UPLOAD_ROOT, exist_ok=True)

# client = MongoClient(MONGO_URI)
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())

db = client[DB_NAME]
users_col = db["users"]

app = FastAPI(title="QA Docs API (Gemini + MongoDB Atlas)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()


class RegisterModel(BaseModel):
    username: str
    email: str
    password: str


class LoginModel(BaseModel):
    username: str
    password: str


class AskRequest(BaseModel):
    query: str


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    payload = decode_access_token(credentials.credentials)

    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        user_id = ObjectId(payload["user_id"])
    except (InvalidId, KeyError):
        raise HTTPException(status_code=401, detail="Invalid token")

    user = users_col.find_one({"_id": user_id})

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    return user


@app.get("/")
def root():
    return {"message": "QA Docs API is running"}


@app.post("/auth/register")
def register(data: RegisterModel):
    if users_col.find_one({"username": data.username}):
        raise HTTPException(status_code=400, detail="Username already exists")

    result = users_col.insert_one(
        {
            "username": data.username,
            "email": data.email,
            "password": hash_password(data.password),
        }
    )

    user_id = str(result.inserted_id)
    os.makedirs(os.path.join(UPLOAD_ROOT, user_id), exist_ok=True)

    return {
        "message": "Registration successful",
        "user_id": user_id,
    }


@app.post("/auth/login")
def login(data: LoginModel):
    user = users_col.find_one({"username": data.username})

    if user is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid username or password",
        )

    if not verify_password(data.password, user["password"]):
        raise HTTPException(
            status_code=400,
            detail="Invalid username or password",
        )

    token = create_access_token(
        {
            "user_id": str(user["_id"]),
            "username": user["username"],
        }
    )

    return {"access_token": token}


@app.post("/upload")
def upload_file(
    file: UploadFile = File(...),
    user=Depends(get_current_user),
):
    user_id = str(user["_id"])

    user_folder = os.path.join(UPLOAD_ROOT, user_id)
    os.makedirs(user_folder, exist_ok=True)

    file_path = os.path.join(user_folder, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    users_col.update_one(
        {"_id": user["_id"]},
        {
            "$push": {
                "files": {
                    "filename": file.filename,
                }
            }
        },
    )

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
    }


@app.post("/ingest")
def ingest(user=Depends(get_current_user)):
    user_id = str(user["_id"])

    docs_path = os.path.join(UPLOAD_ROOT, user_id)

    if not os.path.exists(docs_path):
        raise HTTPException(
            status_code=404,
            detail="Upload documents first.",
        )

    try:
        result = ingest_for_user(
            user_id=user_id,
            docs_path=docs_path,
        )

        return {
            "message": "Documents ingested successfully",
            **result,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@app.post("/ask")
def ask(
    request: AskRequest,
    user=Depends(get_current_user),
):
    user_id = str(user["_id"])

    try:
        result = ask_question_for_user(
            query=request.query,
            user_id=user_id,
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
