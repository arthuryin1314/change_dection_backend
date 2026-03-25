import base64
import binascii
import hashlib
import hmac
import json
import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import HTTPException, status


@lru_cache(maxsize=1)
def _get_jwt_settings() -> tuple[str, str, int]:
    # Lazy-load env vars on first token operation, not at import time.
    load_dotenv()

    secret_key = os.getenv("JWT_SECRET_KEY")
    if not secret_key or len(secret_key) < 32:
        raise RuntimeError("JWT_SECRET_KEY must be at least 32 characters")

    algorithm = os.getenv("JWT_ALGORITHM", "HS256")
    if algorithm != "HS256":
        raise RuntimeError("Only HS256 is supported")

    expire_hours = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_HOURS", "24"))
    if expire_hours <= 0:
        raise RuntimeError("JWT_ACCESS_TOKEN_EXPIRE_HOURS must be greater than 0")

    return secret_key, algorithm, expire_hours


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
    )


def _utc_now() -> datetime:
    """Return timezone-aware UTC datetime for JWT time claims."""
    return datetime.now(timezone.utc)


def create_access_token(user_id: int, token_version: int = 0) -> str:
    secret_key, algorithm, expire_hours = _get_jwt_settings()

    # Expiration is configured by JWT_ACCESS_TOKEN_EXPIRE_HOURS (default: 24 hours).
    now = _utc_now()
    expire = now + timedelta(hours=expire_hours)
    header = {"alg": algorithm, "typ": "JWT"}
    payload = {
        "sub": str(user_id),
        "ver": int(token_version),
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    header_part = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_part = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_part}.{payload_part}".encode("ascii")
    signature = hmac.new(secret_key.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header_part}.{payload_part}.{_b64url_encode(signature)}"


def verify_access_token(token: str) -> dict[str, int]:
    secret_key, algorithm, _ = _get_jwt_settings()

    try:
        header_part, payload_part, signature_part = token.split(".")
        signing_input = f"{header_part}.{payload_part}".encode("ascii")
        expected_signature = hmac.new(
            secret_key.encode("utf-8"),
            signing_input,
            hashlib.sha256,
        ).digest()
        actual_signature = _b64url_decode(signature_part)
        if not hmac.compare_digest(actual_signature, expected_signature):
            raise _unauthorized()

        header = json.loads(_b64url_decode(header_part).decode("utf-8"))
        if header.get("alg") != algorithm or header.get("typ") != "JWT":
            raise _unauthorized()

        payload = json.loads(_b64url_decode(payload_part).decode("utf-8"))
        if int(payload["exp"]) <= int(_utc_now().timestamp()):
            raise _unauthorized()

        return {
            "user_id": int(payload["sub"]),
            "token_version": int(payload.get("ver", 0)),
        }
    except HTTPException:
        raise
    except (ValueError, KeyError, TypeError, json.JSONDecodeError, binascii.Error):
        raise _unauthorized()
