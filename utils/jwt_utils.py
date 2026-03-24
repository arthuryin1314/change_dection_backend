import base64
import hashlib
import hmac
import json
import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from fastapi import HTTPException, status

load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY or len(SECRET_KEY) < 32:
    raise RuntimeError("JWT_SECRET_KEY must be at least 32 characters")

ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
if ALGORITHM != "HS256":
    raise RuntimeError("Only HS256 is supported")

ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_HOURS", "24"))
if ACCESS_TOKEN_EXPIRE_HOURS <= 0:
    raise RuntimeError("JWT_ACCESS_TOKEN_EXPIRE_HOURS must be greater than 0")


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )


def create_access_token(user_id: int, token_version: int = 0) -> str:
    now = datetime.now(timezone.utc)
    expire = now + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    header = {"alg": ALGORITHM, "typ": "JWT"}
    payload = {
        "sub": str(user_id),
        "ver": int(token_version),
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    header_part = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_part = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_part}.{payload_part}".encode("ascii")
    signature = hmac.new(SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header_part}.{payload_part}.{_b64url_encode(signature)}"


def verify_access_token(token: str) -> dict[str, int]:
    try:
        header_part, payload_part, signature_part = token.split(".")
        signing_input = f"{header_part}.{payload_part}".encode("ascii")
        expected_signature = hmac.new(
            SECRET_KEY.encode("utf-8"),
            signing_input,
            hashlib.sha256,
        ).digest()
        actual_signature = _b64url_decode(signature_part)
        if not hmac.compare_digest(actual_signature, expected_signature):
            raise _unauthorized()

        header = json.loads(_b64url_decode(header_part).decode("utf-8"))
        if header.get("alg") != ALGORITHM or header.get("typ") != "JWT":
            raise _unauthorized()

        payload = json.loads(_b64url_decode(payload_part).decode("utf-8"))
        if int(payload["exp"]) <= int(datetime.now(timezone.utc).timestamp()):
            raise _unauthorized()

        return {
            "user_id": int(payload["sub"]),
            "token_version": int(payload.get("ver", 0)),
        }
    except HTTPException:
        raise
    except (ValueError, KeyError, TypeError, json.JSONDecodeError, base64.binascii.Error):
        raise _unauthorized()
