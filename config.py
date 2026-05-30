"""
Centralized application configuration.

All environment variables and constants are defined here.
Other modules import from this file — never read os.getenv() directly in business logic.
"""
import hashlib
import os

from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address

load_dotenv()

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./telegram_app.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# ── Security ──────────────────────────────────────────────────────────────────
JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "")
if not JWT_SECRET_KEY:
    raise ValueError(
        "JWT_SECRET_KEY is required. "
        'Generate one: python -c "import secrets; print(secrets.token_hex(32))"'
    )
if len(JWT_SECRET_KEY.encode()) < 32:
    raise ValueError(
        f"JWT_SECRET_KEY is only {len(JWT_SECRET_KEY.encode())} bytes "
        "— minimum 32 required for HS256."
    )

JWT_ALGORITHM: str = "HS256"
JWT_EXPIRATION_HOURS: int = 24

# ── CORS ──────────────────────────────────────────────────────────────────────
_ALLOWED_ORIGINS_RAW: str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")


def get_allowed_origins() -> list[str]:
    return [o.strip() for o in _ALLOWED_ORIGINS_RAW.split(",") if o.strip()]


# ── AI / ONNX ─────────────────────────────────────────────────────────────────
ONNX_MODEL_PATH: str = os.getenv("ONNX_MODEL_PATH", "./sentiment_onnx")

# Global confidence threshold for the "negative" label.
# XLM-RoBERTa was trained primarily on English/European data; it returns lower
# confidence scores for Uzbek text.  Use AI_UZ_SCORE_THRESHOLD for Uzbek input.
AI_NEGATIVE_SCORE_THRESHOLD: float = float(os.getenv("AI_NEGATIVE_SCORE_THRESHOLD", "0.80"))

# Uzbek-specific threshold — lower because the multilingual model is less
# calibrated for Uzbek and tends to under-score genuine toxic content.
AI_UZ_SCORE_THRESHOLD: float = float(os.getenv("AI_UZ_SCORE_THRESHOLD", "0.65"))

# ── Telethon connection ───────────────────────────────────────────────────────
TELETHON_CONNECTION_TIMEOUT: int = 30
TELETHON_RETRIES: int = 3
TELETHON_RETRY_DELAY: int = 1
BACKOFF_BASE: int = 1
BACKOFF_MAX: int = 30
MAX_RETRY_ATTEMPTS: int = 5

# ── Client / session pool ─────────────────────────────────────────────────────
MAX_ACTIVE_CLIENTS: int = 200
PENDING_2FA_TTL: int = 300  # seconds a partial-auth client is kept alive

# ── Punishment thresholds ─────────────────────────────────────────────────────
# Violations 1 .. WARN_LIMIT-1  → warning notice sent to chat
# Violations WARN_LIMIT .. BAN_AFTER_WARNS-1 → user muted for MUTE_DURATION_SECONDS
# Violations >= BAN_AFTER_WARNS  → permanent ban
WARN_LIMIT: int = int(os.getenv("WARN_LIMIT", "3"))
BAN_AFTER_WARNS: int = int(os.getenv("BAN_AFTER_WARNS", "5"))
MUTE_DURATION_SECONDS: int = int(os.getenv("MUTE_DURATION_SECONDS", "3600"))  # 1 hour

# ── Server ────────────────────────────────────────────────────────────────────
PORT: int = int(os.getenv("PORT", "8001"))


# ── Rate-limiter (user-aware: token hash > IP) ────────────────────────────────
def _rate_limit_key(request) -> str:  # type: ignore[no-untyped-def]
    token: str = request.cookies.get("telegram_token") or ""
    if not token:
        auth: str = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
    if token:
        return "user:" + hashlib.sha256(token.encode()).hexdigest()[:16]
    return get_remote_address(request)


limiter: Limiter = Limiter(key_func=_rate_limit_key)
