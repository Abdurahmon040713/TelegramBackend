"""
Authentication utilities: JWT creation/verification and phone helpers.

Keeping JWT logic here means config changes (algorithm, expiry) only
touch one file, and the dependency injection point (verify_token) is easy to mock in tests.
"""
import re
import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import jwt
from fastapi import Header, HTTPException

from config import JWT_SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRATION_HOURS

logger = logging.getLogger(__name__)


# ── Phone helpers ─────────────────────────────────────────────────────────────

def normalize_phone(phone: str) -> str:
    """Strip all non-digit characters so +998901234567 == 998901234567."""
    return re.sub(r"\D", "", phone)


def validate_phone(phone: str) -> bool:
    """Accept phones between 10 and 15 digits (covers all countries)."""
    digits = re.sub(r"\D", "", phone)
    return 10 <= len(digits) <= 15


# ── Token creation ────────────────────────────────────────────────────────────

def create_access_token(
    phone: str,
    expires_delta: timedelta | None = None,
) -> str:
    if expires_delta is None:
        expires_delta = timedelta(hours=JWT_EXPIRATION_HOURS)

    expire = datetime.now(timezone.utc) + expires_delta
    payload = {
        "phone": normalize_phone(phone),
        "exp":   expire,
        "iat":   datetime.now(timezone.utc),
        "jti":   str(uuid4()),  # unique token ID — ready for future revocation
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


# ── Token verification (FastAPI dependency) ───────────────────────────────────

def verify_token(authorization: str = Header(default=None)) -> str:
    """Validate Bearer JWT and return the digits-only phone number.

    Raises HTTPException(401) on any authentication failure so callers
    never need to handle raw jwt exceptions.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization talab qilinadi")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Noto'g'ri token formati (Bearer kerak)")

    token = authorization[7:]
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        phone: str | None = payload.get("phone")
        if not phone:
            raise HTTPException(status_code=401, detail="Token'da phone topilmadi")
        return normalize_phone(phone)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token muddati tugadi")
    except jwt.InvalidTokenError as exc:
        logger.warning("Invalid JWT: %s", exc)
        raise HTTPException(status_code=401, detail="Noto'g'ri token")
