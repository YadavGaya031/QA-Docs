# backend/auth_utils.py
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta
import os

PWD_CTX = CryptContext(schemes=["bcrypt"], deprecated="auto")
JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
JWT_ALG = "HS256"
ACCESS_TOKEN_EXPIRES_MINUTES = 60*24  # 1 day

def hash_password(password: str) -> str:
    return PWD_CTX.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    return PWD_CTX.verify(password, hashed)

def create_access_token(data: dict, expires_delta: int = ACCESS_TOKEN_EXPIRES_MINUTES):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_delta)
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALG)
    return token

def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return payload
    except jwt.PyJWTError:
        return None
