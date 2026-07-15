
import os
from datetime import datetime, timedelta, timezone

import jwt
from pwdlib import PasswordHash

# -----------------------------
# Password Hashing
# -----------------------------
password_hash = PasswordHash.recommended()

# -----------------------------
# JWT Config
# -----------------------------
JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24


# -----------------------------
# Password Functions
# -----------------------------
def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    return password_hash.verify(password, hashed_password)


# -----------------------------
# JWT Functions
# -----------------------------
def create_access_token(
    data: dict,
    expires_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES,
):

    payload = data.copy()

    expire = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)

    payload["exp"] = expire

    token = jwt.encode(
        payload,
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )

    return token


def decode_access_token(token: str):

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
        )

        return payload

    except jwt.ExpiredSignatureError:
        return None

    except jwt.InvalidTokenError:
        return None
