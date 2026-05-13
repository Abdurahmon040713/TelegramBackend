from fastapi import FastAPI, Depends, HTTPException, Header, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import hashlib
import os
import re
import time
import random
import uuid as uuid_module
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Set
from enum import Enum

# ── logging must be configured before anything else logs ────────────────────
try:
    from logging_config import setup_logging, mask_phone, RequestIdMiddleware
    setup_logging()
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)

    def mask_phone(phone: str) -> str:  # type: ignore[misc]
        if len(phone) < 6:
            return "***"
        return phone[:3] + "*" * (len(phone) - 6) + phone[-3:]

    RequestIdMiddleware = None  # type: ignore[misc,assignment]

import logging
import jwt
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from pydantic import BaseModel, field_validator, Field
from telethon import TelegramClient, errors
from telethon.sessions import StringSession
from sqlalchemy import (
    Boolean, Column, Float, Integer, String, Table, MetaData, UniqueConstraint,
    create_engine, text as sql_text,
)
from databases import Database
import ai_inference
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

logger = logging.getLogger(__name__)

try:
    from db import (
        DATABASE_URL,
        encrypt_session_string,
        decrypt_session_string,
        encrypt_value,
        decrypt_value,
        decrypt_session_string_legacy,
        decrypt_value_legacy,
    )
except ImportError as exc:
    raise ImportError(f"db.py ni import qilib bo'lmadi: {exc}. Dastur to'xtatildi.")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL topilmadi")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Database
# ─────────────────────────────────────────────────────────────────────────────
database = Database(DATABASE_URL)
metadata = MetaData()

sessions = Table(
    "sessions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("phone", String, unique=True, index=True),
    Column("api_id", Integer),
    Column("api_hash", String),
    Column("session_string", String),
)

messages_table = Table(
    "messages",
    metadata,
    Column("row_id", Integer, primary_key=True),
    Column("phone", String, nullable=False),
    Column("chat_id", Integer, nullable=False),
    Column("message_id", Integer, nullable=False),
    Column("sender_id", Integer, nullable=True),
    Column("text", String, nullable=False),
    Column("is_negative", Boolean, default=False),
    Column("reason", String, nullable=True),
    Column("confidence", Float, default=0.0),
    Column("analyzed_at", String, nullable=True),
    UniqueConstraint("phone", "chat_id", "message_id", name="uq_messages_phone_chat_msg"),
)

analyses_table = Table(
    "analyses",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("phone", String, nullable=False),
    Column("chat_id", Integer, nullable=False),
    Column("fetch_limit", Integer, nullable=False),
    Column("analyzed_count", Integer, default=0),
    Column("negative_count", Integer, default=0),
    Column("completed_at", String, nullable=True),
)

engine = create_engine(DATABASE_URL)
metadata.create_all(engine)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Global state
# ─────────────────────────────────────────────────────────────────────────────

MAX_ACTIVE_CLIENTS = 200
active_clients: OrderedDict[str, TelegramClient] = OrderedDict()
active_clients_lock = asyncio.Lock()

analysis_cache: Dict[str, Dict[str, Any]] = {}
cache_lock = asyncio.Lock()
CACHE_TTL_SECONDS = 3600

# Background job store for async /analyze/start → /analyze/status polling
MAX_JOBS = 1000
job_store: Dict[str, Dict[str, Any]] = {}
job_store_lock = asyncio.Lock()

# 2FA pending clients — kept alive for up to 5 min while user enters password
PENDING_2FA_TTL = 300
pending_2fa_clients: Dict[str, Dict[str, Any]] = {}
pending_2fa_lock = asyncio.Lock()

# JWT
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    raise ValueError(
        "JWT_SECRET_KEY environment variable is required. "
        "Generate one: python -c \"import secrets; print(secrets.token_hex(32))\""
    )
if len(JWT_SECRET_KEY.encode()) < 32:
    raise ValueError(
        f"JWT_SECRET_KEY is only {len(JWT_SECRET_KEY.encode())} bytes — "
        "minimum 32 required for HS256."
    )
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Telethon
TELETHON_CONNECTION_TIMEOUT = 30
TELETHON_RETRIES = 3
TELETHON_RETRY_DELAY = 1

BACKOFF_BASE = 1
BACKOFF_MAX = 30
MAX_RETRY_ATTEMPTS = 5

# ─────────────────────────────────────────────────────────────────────────────
# 2.5. Rate limiting — user-aware (not trivially bypassed by changing IP)
# ─────────────────────────────────────────────────────────────────────────────
def _rate_limit_key(request: Request) -> str:
    token = request.cookies.get("telegram_token") or ""
    if not token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
    if token:
        return "user:" + hashlib.sha256(token.encode()).hexdigest()[:16]
    return get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Negative word list + leet-speak patterns
# ─────────────────────────────────────────────────────────────────────────────
def load_negative_words(file_path: str = "data/uz_negative_words.txt") -> List[str]:
    try:
        if not os.path.exists(file_path):
            logger.warning("Negativ so'zlar fayli topilmadi: %s", file_path)
            return []
        with open(file_path, "r", encoding="utf-8") as f:
            words = [line.strip().lower() for line in f if line.strip()]
        logger.info("Loaded %d negative keywords", len(words))
        return words
    except Exception:
        logger.exception("Error loading negative words")
        return []


NEGATIVE_KEYWORDS = load_negative_words()
KEYWORD_PATTERNS: List[re.Pattern] = []
AI_NEGATIVE_SCORE_THRESHOLD = 0.85

LEET_MAP = {
    "a": r"[a4@]", "b": r"[b8]", "c": r"[c(\[]", "d": r"[d]",
    "e": r"[e3€]", "g": r"[g69q]", "h": r"[h#]", "i": r"[i1!|]",
    "j": r"[j]", "k": r"[k]", "l": r"[l1!|]", "m": r"[m]",
    "n": r"[n]", "o": r"[o0]", "p": r"[p]", "q": r"[q9]",
    "r": r"[r]", "s": r"[s5$]", "t": r"[t7+]", "u": r"[uüv]",
    "v": r"[v]", "x": r"[x×*]", "y": r"[y]", "z": r"[z2]",
}
SEPARATOR = r"[^\w]*"


def build_keyword_pattern(keyword: str) -> re.Pattern:
    normalized = keyword.strip().lower()
    if len(normalized) <= 2:
        return re.compile(rf"(?<!\w){re.escape(normalized)}(?!\w)", re.IGNORECASE)
    parts = [LEET_MAP.get(ch, re.escape(ch)) for ch in normalized]
    return re.compile(rf"(?<!\w){SEPARATOR.join(parts)}(?!\w)", re.IGNORECASE)


def is_toxic_by_keywords(text: str) -> bool:
    text_lower = text.lower()
    return any(p.search(text_lower) for p in KEYWORD_PATTERNS)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Lifespan
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.connect()

    global KEYWORD_PATTERNS

    # ── AI model ──────────────────────────────────────────────────────────────
    logger.info("AI Model yuklanmoqda...")
    if ai_inference.load_model():
        logger.info("AI Model tayyor  [backend=%s]", ai_inference.get_backend())
    else:
        logger.warning("AI model yuklanmadi; faqat keyword tahlili ishlatiladi")

    KEYWORD_PATTERNS = [build_keyword_pattern(k) for k in NEGATIVE_KEYWORDS]
    logger.info("%d keyword pattern tayyor", len(KEYWORD_PATTERNS))

    # ── Schema migration: add columns that older deployments may be missing ──────
    if "postgresql" in DATABASE_URL.lower():
        _alter_cols = [
            # analyses table
            "ALTER TABLE analyses ADD COLUMN IF NOT EXISTS fetch_limit   INTEGER NOT NULL DEFAULT 50",
            "ALTER TABLE analyses ADD COLUMN IF NOT EXISTS analyzed_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE analyses ADD COLUMN IF NOT EXISTS negative_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE analyses ADD COLUMN IF NOT EXISTS completed_at  VARCHAR",
            # messages table
            "ALTER TABLE messages ADD COLUMN IF NOT EXISTS is_negative   BOOLEAN DEFAULT FALSE",
            "ALTER TABLE messages ADD COLUMN IF NOT EXISTS reason        VARCHAR",
            "ALTER TABLE messages ADD COLUMN IF NOT EXISTS confidence    FLOAT   DEFAULT 0.0",
            "ALTER TABLE messages ADD COLUMN IF NOT EXISTS analyzed_at   VARCHAR",
        ]
        try:
            with engine.connect() as conn:
                for stmt in _alter_cols:
                    conn.execute(sql_text(stmt))
                conn.commit()
            logger.info("Schema migration: columns verified/added")
        except Exception as exc:
            logger.warning("Schema migration skipped: %s", exc)

    # ── PostgreSQL full-text search index ─────────────────────────────────────
    if "postgresql" in DATABASE_URL.lower():
        try:
            with engine.connect() as conn:
                conn.execute(sql_text(
                    "CREATE INDEX IF NOT EXISTS idx_messages_fts "
                    "ON messages USING GIN (to_tsvector('simple', text))"
                ))
                conn.execute(sql_text(
                    "CREATE INDEX IF NOT EXISTS idx_messages_phone_chat "
                    "ON messages (phone, chat_id)"
                ))
                conn.execute(sql_text(
                    "CREATE INDEX IF NOT EXISTS idx_analyses_phone_chat "
                    "ON analyses (phone, chat_id)"
                ))
                conn.commit()
            logger.info("PostgreSQL FTS indexes ready")
        except Exception as exc:
            logger.warning("FTS index setup skipped: %s", exc)

    # ── Startup migration 1: re-encrypt api_hash values under new PBKDF2 key ─
    try:
        all_sessions = await database.fetch_all(sessions.select())
        migrated = 0
        for row in all_sessions:
            raw = row["api_hash"]
            # Try new key first
            try:
                decrypt_value(raw)
                continue  # already on new key
            except Exception:
                pass
            # Try legacy key
            try:
                plaintext = decrypt_value_legacy(raw)
                await database.execute(
                    sessions.update()
                    .where(sessions.c.phone == row["phone"])
                    .values(api_hash=encrypt_value(plaintext))
                )
                migrated += 1
                continue
            except Exception:
                pass
            # Plaintext value from before any encryption
            try:
                await database.execute(
                    sessions.update()
                    .where(sessions.c.phone == row["phone"])
                    .values(api_hash=encrypt_value(raw))
                )
                migrated += 1
            except Exception:
                logger.warning("api_hash migration skipped for a row")
        if migrated:
            logger.info("%d api_hash row(s) re-encrypted to PBKDF2 key", migrated)
    except Exception:
        logger.warning("api_hash migration failed — skipped", exc_info=True)

    # ── Startup migration 2: re-encrypt session_strings under new PBKDF2 key ─
    try:
        all_sessions = await database.fetch_all(sessions.select())
        migrated = 0
        for row in all_sessions:
            raw = row["session_string"]
            try:
                decrypt_session_string(raw)
                continue
            except Exception:
                pass
            try:
                plaintext = decrypt_session_string_legacy(raw)
                await database.execute(
                    sessions.update()
                    .where(sessions.c.phone == row["phone"])
                    .values(session_string=encrypt_session_string(plaintext))
                )
                migrated += 1
            except Exception:
                logger.warning("session_string migration skipped for a row")
        if migrated:
            logger.info("%d session_string row(s) re-encrypted to PBKDF2 key", migrated)
    except Exception:
        logger.warning("session_string migration failed — skipped", exc_info=True)

    # ── Startup migration 3: normalize phone numbers to digits-only ───────────
    try:
        all_sessions = await database.fetch_all(sessions.select())
        migrated = 0
        for row in all_sessions:
            normalized = normalize_phone(row["phone"])
            if normalized != row["phone"]:
                try:
                    await database.execute(
                        sessions.update()
                        .where(sessions.c.phone == row["phone"])
                        .values(phone=normalized)
                    )
                    migrated += 1
                except Exception:
                    logger.warning("Phone normalization skipped for row: %s", mask_phone(row["phone"]))
        if migrated:
            logger.info("%d phone number(s) normalized to digits-only", migrated)
    except Exception:
        logger.warning("Phone normalization migration failed — skipped", exc_info=True)

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    async with active_clients_lock:
        for phone, client in active_clients.items():
            try:
                if client.is_connected():
                    await client.disconnect()
            except Exception as exc:
                logger.warning("Error disconnecting %s: %s", mask_phone(phone), exc)

    async with pending_2fa_lock:
        for phone, entry in pending_2fa_clients.items():
            try:
                await entry["client"].disconnect()
            except Exception:
                pass
        pending_2fa_clients.clear()

    await database.disconnect()


# ─────────────────────────────────────────────────────────────────────────────
# 5. App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Telegram Sentiment Backend", lifespan=lifespan)


def get_allowed_origins() -> List[str]:
    raw = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000")
    return [o.strip() for o in raw.split(",") if o.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
app.add_middleware(SlowAPIMiddleware)
if RequestIdMiddleware is not None:
    app.add_middleware(RequestIdMiddleware)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Pydantic models
# ─────────────────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    api_id: int
    api_hash: str
    phone: str

    @field_validator("api_id")
    @classmethod
    def api_id_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("API ID musbat bo'lishi kerak")
        return v

    @field_validator("api_hash")
    @classmethod
    def api_hash_not_empty(cls, v: str) -> str:
        if not v or len(v) < 10:
            raise ValueError("API Hash juda qisqa")
        return v


class VerifyRequest(BaseModel):
    phone: str
    code: str
    phone_code_hash: str
    api_id: int
    api_hash: str

    @field_validator("code")
    @classmethod
    def code_valid(cls, v: str) -> str:
        if not v or len(v) < 3:
            raise ValueError("Kod juda qisqa")
        return v


class AnalyzeRequest(BaseModel):
    phone: str
    chat_id: int
    limit: int = 50

    @field_validator("limit")
    @classmethod
    def limit_valid(cls, v: int) -> int:
        if v <= 0 or v > 1000:
            raise ValueError("Limit 1 dan 1000 gacha bo'lishi kerak")
        return v


class PhoneRequest(BaseModel):
    phone: str

    @field_validator("phone")
    @classmethod
    def phone_valid(cls, v: str) -> str:
        digits = re.sub(r"\D", "", v)
        if not (10 <= len(digits) <= 15):
            raise ValueError("Noto'g'ri telefon raqami")
        return v


class MessageReason(str, Enum):
    KEYWORD_MATCH = "keyword_match"
    AI_SENTIMENT = "ai_sentiment"


class NegativeMessage(BaseModel):
    id: int
    text: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    sender_id: Optional[int] = None
    reason: MessageReason


class AnalyzeResponse(BaseModel):
    analyzed_count: int
    negative_count: int
    negative_messages: List[NegativeMessage]


class TwoFARequest(BaseModel):
    phone: str
    password: str


class SearchRequest(BaseModel):
    phone: str
    query: str
    chat_id: Optional[int] = None
    negative_only: bool = False
    limit: int = 50

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Qidiruv so'zi kamida 2 belgi bo'lishi kerak")
        return v

    @field_validator("limit")
    @classmethod
    def limit_valid(cls, v: int) -> int:
        if v <= 0 or v > 500:
            raise ValueError("Limit 1-500 oralig'ida bo'lishi kerak")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# 7. Helpers
# ─────────────────────────────────────────────────────────────────────────────
def validate_phone(phone: str) -> bool:
    digits = re.sub(r"\D", "", phone)
    return 10 <= len(digits) <= 15


def normalize_phone(phone: str) -> str:
    """Strip everything except digits so +998901234567 == 998901234567."""
    return re.sub(r"\D", "", phone)


async def is_client_healthy(client: TelegramClient) -> bool:
    try:
        if not client.is_connected():
            return False
        me = await asyncio.wait_for(client.get_me(), timeout=5)
        return me is not None
    except asyncio.TimeoutError:
        logger.warning("Client health check timeout")
        return False
    except Exception as exc:
        logger.debug("Health check failed: %s", exc)
        return False


def calculate_backoff_delay(attempt: int) -> float:
    base = min(BACKOFF_BASE * (2 ** attempt), BACKOFF_MAX)
    return base + random.uniform(0, 1)


async def connect_with_retry(
    client: TelegramClient, phone: str, max_attempts: int = MAX_RETRY_ATTEMPTS
) -> bool:
    for attempt in range(max_attempts):
        try:
            await client.connect()
            logger.info("Connected for %s (attempt %d)", mask_phone(phone), attempt + 1)
            return True
        except Exception as exc:
            if attempt < max_attempts - 1:
                delay = calculate_backoff_delay(attempt)
                logger.warning(
                    "Connection failed %d/%d, retry in %.1fs: %s",
                    attempt + 1, max_attempts, delay, type(exc).__name__,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "Connection failed after %d attempts for %s: %s",
                    max_attempts, mask_phone(phone), type(exc).__name__,
                )
                return False
    return False


async def _evict_lru_client_if_needed() -> None:
    """Evict the least-recently-used client when the pool is full."""
    async with active_clients_lock:
        if len(active_clients) >= MAX_ACTIVE_CLIENTS:
            oldest_phone, oldest_client = next(iter(active_clients.items()))
            try:
                await oldest_client.disconnect()
            except Exception:
                pass
            del active_clients[oldest_phone]
            logger.info("LRU evicted client for %s", mask_phone(oldest_phone))


async def get_client_session(phone: str) -> TelegramClient:
    """Return a healthy TelegramClient for *phone* (digits-only expected)."""
    cached_client: Optional[TelegramClient] = None
    async with active_clients_lock:
        if phone in active_clients:
            active_clients.move_to_end(phone)  # mark recently used
            cached_client = active_clients[phone]

    if cached_client is not None:
        if await is_client_healthy(cached_client):
            return cached_client
        logger.warning("Cached client for %s is unhealthy, removing", mask_phone(phone))
        async with active_clients_lock:
            if phone in active_clients:
                try:
                    await active_clients[phone].disconnect()
                except Exception:
                    pass
                active_clients.pop(phone, None)

    query = sessions.select().where(sessions.c.phone == phone)
    user = await database.fetch_one(query)
    if not user:
        raise HTTPException(status_code=404, detail="Avval login qiling")

    try:
        session_string = decrypt_session_string(user["session_string"])
    except ValueError as exc:
        logger.error("Session decryption failed for %s: %s", mask_phone(phone), exc)
        raise HTTPException(
            status_code=401,
            detail="Sessiya o'qilishda xatolik — qayta login qiling",
        )

    try:
        api_hash = decrypt_value(user["api_hash"])
    except Exception:
        raise HTTPException(
            status_code=401,
            detail="API credentials o'qilishda xatolik — qayta login qiling",
        )

    client = TelegramClient(
        StringSession(session_string),
        user["api_id"],
        api_hash,
        timeout=TELETHON_CONNECTION_TIMEOUT,
        connection_retries=TELETHON_RETRIES,
        retry_delay=TELETHON_RETRY_DELAY,
    )

    if not await connect_with_retry(client, phone):
        raise HTTPException(
            status_code=502,
            detail="Telegram serveriga ulanib bo'lmadi. 1-2 daqiqadan so'ng qayta urining",
        )

    try:
        authorized = await asyncio.wait_for(client.is_user_authorized(), timeout=10)
    except Exception as exc:
        logger.warning("is_user_authorized failed for %s: %s", mask_phone(phone), type(exc).__name__)
        authorized = False
    if not authorized:
        await client.disconnect()
        raise HTTPException(
            status_code=401,
            detail="Sessiya tugagan yoki yaroqsiz — qayta login qiling",
        )

    await _evict_lru_client_if_needed()
    async with active_clients_lock:
        active_clients[phone] = client

    logger.info("New client created for %s", mask_phone(phone))
    return client


def get_cache_key(phone: str, chat_id: int, limit: int) -> str:
    return f"{phone}:{chat_id}:{limit}"


async def get_cached_analysis(cache_key: str) -> Optional[Dict[str, Any]]:
    async with cache_lock:
        entry = analysis_cache.get(cache_key)
        if not entry:
            return None
        if entry["expiry"] < time.time():
            analysis_cache.pop(cache_key, None)
            return None
        return entry["data"]


async def set_cached_analysis(cache_key: str, data: Dict[str, Any]) -> None:
    async with cache_lock:
        analysis_cache[cache_key] = {"data": data, "expiry": time.time() + CACHE_TTL_SECONDS}


async def get_db_cached_analysis(
    phone: str, chat_id: int, limit: int
) -> Optional[Dict[str, Any]]:
    """Check analyses_table for a fresh result covering at least *limit* messages."""
    try:
        row = await database.fetch_one(
            "SELECT id, analyzed_count, negative_count, completed_at "
            "FROM analyses "
            "WHERE phone = :phone AND chat_id = :chat_id AND fetch_limit >= :limit "
            "ORDER BY id DESC LIMIT 1",
            values={"phone": phone, "chat_id": chat_id, "limit": limit},
        )
        if not row or not row["completed_at"]:
            return None

        completed_dt = datetime.fromisoformat(row["completed_at"])
        if completed_dt.tzinfo is None:
            completed_dt = completed_dt.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - completed_dt).total_seconds()
        if age > CACHE_TTL_SECONDS:
            return None

        msg_rows = await database.fetch_all(
            "SELECT message_id, text, confidence, sender_id, reason "
            "FROM messages "
            "WHERE phone = :phone AND chat_id = :chat_id AND is_negative = TRUE "
            "ORDER BY message_id DESC LIMIT :limit",
            values={"phone": phone, "chat_id": chat_id, "limit": limit},
        )
        negative_messages = [
            {
                "id": r["message_id"],
                "text": r["text"],
                "confidence": r["confidence"],
                "sender_id": r["sender_id"],
                "reason": r["reason"],
            }
            for r in msg_rows
        ]
        return {
            "analyzed_count": row["analyzed_count"],
            "negative_count": row["negative_count"],
            "negative_messages": negative_messages,
        }
    except Exception as exc:
        logger.error("DB cache lookup failed (%s: %s) — proceeding with fresh analysis",
                     type(exc).__name__, exc)
        return None


async def persist_analysis(
    phone: str, chat_id: int, limit: int, result: "AnalyzeResponse"
) -> None:
    """Persist negative messages and analysis metadata to the database."""
    now = datetime.now(timezone.utc).isoformat()
    try:
        for msg in result.negative_messages:
            # ON CONFLICT DO UPDATE = upsert; avoids separate try/except for duplicate keys
            await database.execute(
                "INSERT INTO messages "
                "(phone, chat_id, message_id, sender_id, text, is_negative, reason, confidence, analyzed_at) "
                "VALUES (:phone, :chat_id, :message_id, :sender_id, :text, TRUE, :reason, :confidence, :analyzed_at) "
                "ON CONFLICT (phone, chat_id, message_id) DO UPDATE SET "
                "is_negative = TRUE, reason = EXCLUDED.reason, "
                "confidence = EXCLUDED.confidence, analyzed_at = EXCLUDED.analyzed_at",
                values={
                    "phone": phone, "chat_id": chat_id, "message_id": msg.id,
                    "sender_id": msg.sender_id, "text": msg.text,
                    "reason": msg.reason.value, "confidence": msg.confidence,
                    "analyzed_at": now,
                },
            )
        await database.execute(
            "INSERT INTO analyses (phone, chat_id, fetch_limit, analyzed_count, negative_count, completed_at) "
            "VALUES (:phone, :chat_id, :fetch_limit, :analyzed_count, :negative_count, :completed_at)",
            values={
                "phone": phone, "chat_id": chat_id, "fetch_limit": limit,
                "analyzed_count": result.analyzed_count,
                "negative_count": result.negative_count,
                "completed_at": now,
            },
        )
    except Exception as exc:
        logger.error("Failed to persist analysis to DB (%s: %s)", type(exc).__name__, exc)


_CONNECTION_ERRORS = (
    ConnectionError,
    errors.RPCError,
    asyncio.TimeoutError,
    OSError,
)


async def execute_with_client_retry(client: TelegramClient, operation, phone: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return await operation()
        except (errors.AuthKeyUnregisteredError, errors.UserDeactivatedError) as exc:
            logger.warning("Auth invalidated for %s: %s", mask_phone(phone), type(exc).__name__)
            async with active_clients_lock:
                active_clients.pop(phone, None)
            raise HTTPException(
                status_code=401,
                detail="Telegram sessiyasi bekor qilindi. Qayta login qiling",
            )
        except _CONNECTION_ERRORS as exc:
            if attempt < max_retries - 1:
                logger.warning(
                    "Client op failed for %s (attempt %d), reconnecting: %s",
                    mask_phone(phone), attempt + 1, type(exc).__name__,
                )
                try:
                    await client.disconnect()
                except Exception:
                    pass
                if await connect_with_retry(client, phone):
                    continue
                raise HTTPException(status_code=502, detail="Telegram serveriga qayta ulanib bo'lmadi")
            logger.error(
                "Client op failed after %d attempts for %s: %s",
                max_retries, mask_phone(phone), type(exc).__name__,
            )
            raise HTTPException(
                status_code=502,
                detail="Telegram bilan bog'lanishda uzluksizlik yuz berdi",
            )


def verify_token(authorization: str = Header(None)) -> str:
    """Validate Bearer JWT; return digits-only phone."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization talab qilinadi")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Noto'g'ri token formati")
    token = authorization[7:]
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        phone = payload.get("phone")
        if not phone:
            raise HTTPException(status_code=401, detail="Token'da phone topilmadi")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token muddati tugadi")
    except jwt.InvalidTokenError as exc:
        logger.warning("Invalid token: %s", exc)
        raise HTTPException(status_code=401, detail="Noto'g'ri token")
    return normalize_phone(phone)


def create_access_token(phone: str, expires_delta: Optional[timedelta] = None) -> str:
    if expires_delta is None:
        expires_delta = timedelta(hours=JWT_EXPIRATION_HOURS)
    expire = datetime.now(timezone.utc) + expires_delta
    payload = {
        "phone": normalize_phone(phone),
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": str(uuid4()),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def analyze_texts_batch(texts: List[str]):
    return ai_inference.analyze_batch(texts)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Core analysis logic (shared by sync endpoint and background job)
# ─────────────────────────────────────────────────────────────────────────────
async def _do_analyze(data: AnalyzeRequest, phone: str) -> AnalyzeResponse:
    """Run the full analysis for *data.chat_id*. *phone* must be digits-only."""
    cache_key = get_cache_key(phone, data.chat_id, data.limit)

    # L1: in-memory cache
    cached = await get_cached_analysis(cache_key)
    if cached is not None:
        return AnalyzeResponse(**cached)

    # L2: persistent DB cache (survives restarts; shared across workers)
    db_cached = await get_db_cached_analysis(phone, data.chat_id, data.limit)
    if db_cached is not None:
        await set_cached_analysis(cache_key, db_cached)  # warm L1
        return AnalyzeResponse(**db_cached)

    client = await get_client_session(phone)

    AI_BATCH_SIZE = 100

    async def _flush_ai_batch(
        batch_msgs: list,
        batch_texts: List[str],
        negative_messages: List[NegativeMessage],
        negative_ids: Set[int],
    ) -> None:
        if not batch_texts or not ai_inference.is_available():
            return
        try:
            results = await asyncio.to_thread(analyze_texts_batch, batch_texts)
            for msg, result in zip(batch_msgs, results):
                if msg.id not in negative_ids:
                    score = float(result.get("score", 0.0))
                    if (
                        result.get("label", "").lower() == "negative"
                        and score > AI_NEGATIVE_SCORE_THRESHOLD
                    ):
                        negative_messages.append(
                            NegativeMessage(
                                id=msg.id,
                                text=msg.text,
                                confidence=score,
                                sender_id=msg.sender_id,
                                reason=MessageReason.AI_SENTIMENT,
                            )
                        )
                        negative_ids.add(msg.id)
        except Exception:
            logger.exception("AI batch tahlilida xatolik")

    async def analyze_operation():
        negative_messages: List[NegativeMessage] = []
        uncertain_msgs: list = []
        uncertain_texts: List[str] = []
        negative_ids: Set[int] = set()
        analyzed_count = 0

        try:
            entity = await client.get_entity(data.chat_id)
        except Exception:
            logger.exception("Chat ID %s ga kira olmadi", data.chat_id)
            raise HTTPException(status_code=404, detail="Chat topilmadi yoki kira olmadi")

        try:
            async for msg in client.iter_messages(entity, limit=data.limit):
                if not msg.text:
                    continue
                analyzed_count += 1

                if is_toxic_by_keywords(msg.text):
                    if msg.id not in negative_ids:
                        negative_messages.append(
                            NegativeMessage(
                                id=msg.id,
                                text=msg.text,
                                confidence=0.0,
                                sender_id=msg.sender_id,
                                reason=MessageReason.KEYWORD_MATCH,
                            )
                        )
                        negative_ids.add(msg.id)
                else:
                    uncertain_msgs.append(msg)
                    uncertain_texts.append(msg.text)
                    if len(uncertain_texts) >= AI_BATCH_SIZE:
                        await _flush_ai_batch(
                            uncertain_msgs, uncertain_texts, negative_messages, negative_ids
                        )
                        uncertain_msgs = []
                        uncertain_texts = []
        except HTTPException:
            raise
        except Exception:
            logger.exception("Xabarlarni olishda xatolik")
            raise HTTPException(status_code=500, detail="Xabarlarni olishda xatolik yuz berdi")

        if uncertain_texts:
            await _flush_ai_batch(uncertain_msgs, uncertain_texts, negative_messages, negative_ids)

        return AnalyzeResponse(
            analyzed_count=analyzed_count,
            negative_count=len(negative_messages),
            negative_messages=negative_messages,
        )

    result = await execute_with_client_retry(client, analyze_operation, phone)

    cache_data = {
        "analyzed_count": result.analyzed_count,
        "negative_count": result.negative_count,
        "negative_messages": [m.model_dump() for m in result.negative_messages],
    }
    await set_cached_analysis(cache_key, cache_data)
    await persist_analysis(phone, data.chat_id, data.limit, result)
    return result


async def _run_analysis_job(job_id: str, data: AnalyzeRequest, phone: str) -> None:
    """Background task: run _do_analyze and store result in job_store."""
    async with job_store_lock:
        job_store[job_id]["status"] = "running"
    try:
        result = await _do_analyze(data, phone)
        async with job_store_lock:
            job_store[job_id] = {
                "status": "done",
                "result": result.model_dump(),
                "error": None,
            }
    except HTTPException as exc:
        async with job_store_lock:
            job_store[job_id] = {"status": "error", "result": None, "error": exc.detail}
    except Exception:
        logger.exception("Background analysis job %s failed", job_id)
        async with job_store_lock:
            job_store[job_id] = {
                "status": "error",
                "result": None,
                "error": "Tahlil davomida kutilmagan xatolik yuz berdi",
            }


# ─────────────────────────────────────────────────────────────────────────────
# 9. Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def read_root(request: Request):
    return {"message": "Backend ishlayapti!"}


@app.get("/health")
async def health(request: Request):
    return {
        "status": "ok",
        "ai_available": ai_inference.is_available(),
        "ai_backend": ai_inference.get_backend(),
    }


@app.post("/login")
@limiter.limit("5/minute")
async def login(request: Request, data: LoginRequest):
    if not validate_phone(data.phone):
        raise HTTPException(status_code=400, detail="Noto'g'ri telefon raqami")

    normalized_phone = normalize_phone(data.phone)

    client = TelegramClient(
        StringSession(),
        data.api_id,
        data.api_hash,
        timeout=TELETHON_CONNECTION_TIMEOUT,
        connection_retries=TELETHON_RETRIES,
        retry_delay=TELETHON_RETRY_DELAY,
    )

    if not await connect_with_retry(client, normalized_phone):
        raise HTTPException(
            status_code=502,
            detail="Telegram serveriga ulanib bo'lmadi. 1-2 daqiqadan so'ng qayta urining",
        )

    try:
        sent = await asyncio.wait_for(client.send_code_request(data.phone), timeout=10)
        session_string = client.session.save()

        encrypted_session = encrypt_session_string(session_string)
        encrypted_hash = encrypt_value(data.api_hash)

        try:
            existing = await database.fetch_one(
                sessions.select().where(sessions.c.phone == normalized_phone)
            )
            if existing:
                await database.execute(
                    sessions.update()
                    .where(sessions.c.phone == normalized_phone)
                    .values(
                        api_id=data.api_id,
                        api_hash=encrypted_hash,
                        session_string=encrypted_session,
                    )
                )
            else:
                await database.execute(
                    sessions.insert().values(
                        phone=normalized_phone,
                        api_id=data.api_id,
                        api_hash=encrypted_hash,
                        session_string=encrypted_session,
                    )
                )
        except Exception:
            logger.exception("Database error during /login")
            raise HTTPException(status_code=500, detail="Ma'lumotlar bazasiga saqlashda xatolik")

        logger.info("SMS code sent to %s", mask_phone(normalized_phone))
        return {"status": "waiting_for_code", "phone_code_hash": sent.phone_code_hash}

    except asyncio.TimeoutError:
        logger.error("SMS code request timeout for %s", mask_phone(normalized_phone))
        raise HTTPException(
            status_code=504,
            detail="Telegram javobi kelmadi. Internet ulanishingizni tekshiring",
        )
    except errors.RPCError as exc:
        logger.warning("Telegram RPC error for %s: %s", mask_phone(normalized_phone), exc)
        raise HTTPException(status_code=502, detail="Telegram xatosi. Qayta urining")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error in /login for %s", mask_phone(normalized_phone))
        raise HTTPException(status_code=500, detail="Noma'lum xatolik yuz berdi")
    finally:
        try:
            if client.is_connected():
                await client.disconnect()
        except Exception as exc:
            logger.warning("Failed to disconnect client in /login finally: %s", exc)


@app.post("/verify")
@limiter.limit("5/minute")
async def verify(request: Request, data: VerifyRequest):
    if not validate_phone(data.phone):
        raise HTTPException(status_code=400, detail="Noto'g'ri telefon raqami")

    normalized_phone = normalize_phone(data.phone)

    user = await database.fetch_one(
        sessions.select().where(sessions.c.phone == normalized_phone)
    )
    if not user:
        raise HTTPException(status_code=404, detail="Login qilinmagan — avval /login qiling")

    try:
        session_string = decrypt_session_string(user["session_string"])
    except ValueError:
        logger.error("Session decryption failed for %s", mask_phone(normalized_phone))
        raise HTTPException(
            status_code=401,
            detail="Sessiya o'qilishda xatolik — qayta login qiling",
        )

    try:
        stored_api_hash = decrypt_value(user["api_hash"])
    except Exception:
        logger.error("api_hash decryption failed for %s", mask_phone(normalized_phone))
        raise HTTPException(
            status_code=401,
            detail="API credentials o'qilishda xatolik — qayta login qiling",
        )

    if user["api_id"] != data.api_id or stored_api_hash != data.api_hash:
        logger.warning("API credentials mismatch for %s", mask_phone(normalized_phone))
        raise HTTPException(status_code=401, detail="API credentials mos kelmadi")

    client = TelegramClient(
        StringSession(session_string),
        user["api_id"],
        stored_api_hash,
        timeout=TELETHON_CONNECTION_TIMEOUT,
        connection_retries=TELETHON_RETRIES,
        retry_delay=TELETHON_RETRY_DELAY,
    )

    if not await connect_with_retry(client, normalized_phone):
        raise HTTPException(status_code=502, detail="Telegram serveriga ulanib bo'lmadi")

    try:
        await asyncio.wait_for(
            client.sign_in(
                phone=data.phone,
                code=data.code,
                phone_code_hash=data.phone_code_hash,
            ),
            timeout=10,
        )
        logger.info("SMS verification succeeded for %s", mask_phone(normalized_phone))

    except asyncio.TimeoutError:
        await client.disconnect()
        raise HTTPException(status_code=504, detail="Telegram javobi kelmadi")
    except errors.SessionPasswordNeededError:
        # Keep the partially-authenticated client alive so /verify-2fa can complete it
        async with pending_2fa_lock:
            pending_2fa_clients[normalized_phone] = {
                "client": client,
                "expires_at": time.time() + PENDING_2FA_TTL,
            }
        logger.info("2FA required for %s — client saved, awaiting password", mask_phone(normalized_phone))
        return {"status": "2fa_required"}
    except errors.PhoneCodeInvalidError:
        await client.disconnect()
        raise HTTPException(status_code=400, detail="SMS kod noto'g'ri")
    except errors.PhoneCodeExpiredError:
        await client.disconnect()
        raise HTTPException(
            status_code=410,
            detail="SMS kod muddati tugadi. Yangi kod uchun /login qiling",
        )
    except errors.PhoneCodeEmpty:
        await client.disconnect()
        raise HTTPException(status_code=400, detail="SMS kod kiritilmagan")
    except errors.FloodWaitError as exc:
        await client.disconnect()
        raise HTTPException(
            status_code=429,
            detail=f"Juda ko'p o'rinish. {exc.seconds} soniyadan so'ng qayta urining",
        )
    except errors.RPCError as exc:
        await client.disconnect()
        logger.error("RPC error for %s: %s", mask_phone(normalized_phone), exc)
        raise HTTPException(status_code=502, detail="Telegram xatosi")
    except Exception:
        await client.disconnect()
        logger.exception("Unexpected error during /verify for %s", mask_phone(normalized_phone))
        raise HTTPException(status_code=500, detail="Noma'lum xatolik")

    new_session = client.session.save()
    try:
        await database.execute(
            sessions.update()
            .where(sessions.c.phone == normalized_phone)
            .values(session_string=encrypt_session_string(new_session))
        )
    except Exception:
        logger.exception("Database error saving session for %s", mask_phone(normalized_phone))
        await client.disconnect()
        raise HTTPException(status_code=500, detail="Ma'lumotlar bazasiga saqlashda xatolik")

    async with active_clients_lock:
        if normalized_phone in active_clients:
            old = active_clients[normalized_phone]
            try:
                if old.is_connected():
                    await old.disconnect()
            except Exception:
                pass
        active_clients[normalized_phone] = client

    access_token = create_access_token(normalized_phone)
    return {
        "status": "success",
        "token": access_token,
        "token_type": "bearer",
        "expires_in": JWT_EXPIRATION_HOURS * 3600,
    }


@app.post("/verify-2fa")
@limiter.limit("5/minute")
async def verify_2fa(request: Request, data: TwoFARequest):
    """Complete login for accounts that require Telegram two-step verification."""
    if not validate_phone(data.phone):
        raise HTTPException(status_code=400, detail="Noto'g'ri telefon raqami")

    normalized_phone = normalize_phone(data.phone)

    async with pending_2fa_lock:
        entry = pending_2fa_clients.get(normalized_phone)

    if not entry:
        raise HTTPException(
            status_code=404,
            detail="2FA sessiyasi topilmadi. Qayta /login va /verify qiling.",
        )
    if entry["expires_at"] < time.time():
        async with pending_2fa_lock:
            pending_2fa_clients.pop(normalized_phone, None)
        raise HTTPException(
            status_code=410,
            detail="2FA muddati tugadi (5 daqiqa). Qayta login qiling.",
        )

    client: TelegramClient = entry["client"]

    try:
        await asyncio.wait_for(client.sign_in(password=data.password), timeout=10)
        logger.info("2FA succeeded for %s", mask_phone(normalized_phone))
    except errors.PasswordHashInvalidError:
        raise HTTPException(status_code=400, detail="2FA paroli noto'g'ri")
    except errors.FloodWaitError as exc:
        raise HTTPException(
            status_code=429,
            detail=f"Juda ko'p urinish. {exc.seconds} soniyadan so'ng qayta urining",
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Telegram javobi kelmadi")
    except Exception:
        logger.exception("2FA failed for %s", mask_phone(normalized_phone))
        raise HTTPException(status_code=500, detail="2FA tekshirishda xatolik")

    # Remove from pending store
    async with pending_2fa_lock:
        pending_2fa_clients.pop(normalized_phone, None)

    # Persist session
    new_session = client.session.save()
    try:
        await database.execute(
            sessions.update()
            .where(sessions.c.phone == normalized_phone)
            .values(session_string=encrypt_session_string(new_session))
        )
    except Exception:
        logger.exception("DB error saving 2FA session for %s", mask_phone(normalized_phone))
        await client.disconnect()
        raise HTTPException(status_code=500, detail="Sessiyani saqlashda xatolik")

    async with active_clients_lock:
        if normalized_phone in active_clients:
            try:
                await active_clients[normalized_phone].disconnect()
            except Exception:
                pass
        active_clients[normalized_phone] = client

    token = create_access_token(normalized_phone)
    return {
        "status": "success",
        "token": token,
        "token_type": "bearer",
        "expires_in": JWT_EXPIRATION_HOURS * 3600,
    }


@app.post("/chats")
@limiter.limit("10/minute")
async def get_chats(
    request: Request,
    data: PhoneRequest,
    authenticated_phone: str = Depends(verify_token),
):
    normalized_phone = normalize_phone(data.phone)
    if authenticated_phone != normalized_phone:
        raise HTTPException(status_code=403, detail="Token va so'rov phone mos kelmaydi")

    user = await database.fetch_one(
        sessions.select().where(sessions.c.phone == normalized_phone)
    )
    if not user:
        raise HTTPException(status_code=401, detail="Noto'g'ri token")

    try:
        client = await get_client_session(normalized_phone)

        async def get_chats_op():
            chats = []
            async for dialog in client.iter_dialogs(limit=50):
                chats.append(
                    {
                        "id": dialog.id,
                        "title": dialog.title,
                        "type": (
                            "Group" if dialog.is_group
                            else "Channel" if dialog.is_channel
                            else "Private"
                        ),
                    }
                )
            return chats

        chats = await execute_with_client_retry(client, get_chats_op, normalized_phone)
        return {"chats": chats}
    except HTTPException:
        raise
    except (errors.AuthKeyUnregisteredError, errors.UserDeactivatedError) as exc:
        logger.warning("Auth invalidated for %s: %s", mask_phone(normalized_phone), type(exc).__name__)
        async with active_clients_lock:
            active_clients.pop(normalized_phone, None)
        raise HTTPException(
            status_code=401,
            detail="Telegram sessiyasi bekor qilindi. Qayta login qiling",
        )
    except _CONNECTION_ERRORS as exc:
        logger.error(
            "Connection error in /chats for %s: %s",
            mask_phone(normalized_phone), type(exc).__name__,
        )
        raise HTTPException(
            status_code=502,
            detail="Telegram serveriga ulanishda xatolik. 1-2 daqiqadan so'ng qayta urining",
        )
    except Exception:
        logger.exception("Unexpected error in /chats for %s", mask_phone(normalized_phone))
        raise HTTPException(status_code=500, detail="Chatlarni olishda kutilmagan xatolik")


@app.post("/analyze", response_model=AnalyzeResponse)
@limiter.limit("5/minute")
async def analyze(
    request: Request,
    data: AnalyzeRequest,
    authenticated_phone: str = Depends(verify_token),
):
    """Synchronous analysis — kept for backward compatibility.
    Prefer /analyze/start + /analyze/status for large message limits."""
    normalized_phone = normalize_phone(data.phone)
    if authenticated_phone != normalized_phone:
        raise HTTPException(status_code=403, detail="Token va so'rov phone mos kelmaydi")

    user = await database.fetch_one(
        sessions.select().where(sessions.c.phone == normalized_phone)
    )
    if not user:
        raise HTTPException(status_code=401, detail="Noto'g'ri token")

    try:
        return await _do_analyze(data, normalized_phone)
    except HTTPException:
        raise
    except (errors.AuthKeyUnregisteredError, errors.UserDeactivatedError) as exc:
        logger.warning("Auth invalidated for %s: %s", mask_phone(normalized_phone), type(exc).__name__)
        async with active_clients_lock:
            active_clients.pop(normalized_phone, None)
        raise HTTPException(
            status_code=401,
            detail="Telegram sessiyasi bekor qilindi. Qayta login qiling",
        )
    except _CONNECTION_ERRORS as exc:
        logger.error(
            "Connection error in /analyze for %s: %s",
            mask_phone(normalized_phone), type(exc).__name__,
        )
        raise HTTPException(
            status_code=502,
            detail="Telegram serveriga ulanishda xatolik. 1-2 daqiqadan so'ng qayta urining",
        )
    except Exception:
        logger.exception("Unexpected error in /analyze for %s", mask_phone(normalized_phone))
        raise HTTPException(status_code=500, detail="Tahlil davomida kutilmagan xatolik")


@app.post("/analyze/start")
@limiter.limit("5/minute")
async def analyze_start(
    request: Request,
    data: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    authenticated_phone: str = Depends(verify_token),
):
    """Non-blocking analysis. Returns job_id; poll /analyze/status/{job_id}."""
    normalized_phone = normalize_phone(data.phone)
    if authenticated_phone != normalized_phone:
        raise HTTPException(status_code=403, detail="Token va so'rov phone mos kelmaydi")

    user = await database.fetch_one(
        sessions.select().where(sessions.c.phone == normalized_phone)
    )
    if not user:
        raise HTTPException(status_code=401, detail="Noto'g'ri token")

    cache_key = get_cache_key(normalized_phone, data.chat_id, data.limit)
    cached = await get_cached_analysis(cache_key)
    if cached is not None:
        job_id = str(uuid_module.uuid4())
        async with job_store_lock:
            if len(job_store) >= MAX_JOBS:
                oldest = next(iter(job_store))
                del job_store[oldest]
            job_store[job_id] = {"status": "done", "result": cached, "error": None}
        # task_id is an alias for job_id — both keys are returned for frontend compatibility
        return {"status": "done", "job_id": job_id, "task_id": job_id}

    job_id = str(uuid_module.uuid4())
    async with job_store_lock:
        if len(job_store) >= MAX_JOBS:
            oldest = next(iter(job_store))
            del job_store[oldest]
        job_store[job_id] = {"status": "queued", "result": None, "error": None}

    background_tasks.add_task(_run_analysis_job, job_id, data, normalized_phone)
    # task_id is an alias for job_id — both keys are returned for frontend compatibility
    return {"status": "queued", "job_id": job_id, "task_id": job_id}


@app.get("/analyze/status/{job_id}")
async def analyze_status(
    request: Request,
    job_id: str,
    authenticated_phone: str = Depends(verify_token),
):
    async with job_store_lock:
        job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job topilmadi yoki muddati o'tgan")
    return job


@app.post("/search")
@limiter.limit("20/minute")
async def search_messages(
    request: Request,
    data: SearchRequest,
    authenticated_phone: str = Depends(verify_token),
):
    """Full-text search across stored negative messages for this user.

    Uses PostgreSQL GIN index (to_tsvector) when available; falls back to
    ILIKE for SQLite in development.
    """
    normalized_phone = normalize_phone(data.phone)
    if authenticated_phone != normalized_phone:
        raise HTTPException(status_code=403, detail="Token va so'rov phone mos kelmaydi")

    try:
        query = (
            messages_table.select()
            .where(messages_table.c.phone == normalized_phone)
        )
        if data.chat_id is not None:
            query = query.where(messages_table.c.chat_id == data.chat_id)
        if data.negative_only:
            query = query.where(messages_table.c.is_negative == True)  # noqa: E712

        if "postgresql" in DATABASE_URL.lower():
            query = query.where(
                sql_text("to_tsvector('simple', messages.text) @@ plainto_tsquery('simple', :q)")
            ).bindparams(q=data.query)
        else:
            query = query.where(
                messages_table.c.text.ilike(f"%{data.query}%")
            )

        query = query.order_by(messages_table.c.row_id.desc()).limit(data.limit)
        rows = await database.fetch_all(query)

        results = [
            {
                "id": r["message_id"],
                "chat_id": r["chat_id"],
                "text": r["text"],
                "is_negative": r["is_negative"],
                "reason": r["reason"],
                "confidence": r["confidence"],
                "sender_id": r["sender_id"],
                "analyzed_at": r["analyzed_at"],
            }
            for r in rows
        ]
        return {"results": results, "count": len(results), "query": data.query}

    except HTTPException:
        raise
    except Exception:
        logger.exception("Search failed for %s", mask_phone(normalized_phone))
        raise HTTPException(status_code=500, detail="Qidiruvda xatolik yuz berdi")


@app.get("/stats")
@limiter.limit("30/minute")
async def get_stats(
    request: Request,
    authenticated_phone: str = Depends(verify_token),
):
    """Return aggregate analysis statistics for the authenticated user."""
    try:
        summary = await database.fetch_one(
            "SELECT "
            "COALESCE(SUM(analyzed_count), 0) AS total_analyzed, "
            "COALESCE(SUM(negative_count), 0) AS total_negative, "
            "COUNT(DISTINCT chat_id)          AS chats_analyzed "
            "FROM analyses WHERE phone = :phone",
            values={"phone": authenticated_phone},
        )

        recent = await database.fetch_all(
            "SELECT chat_id, analyzed_count, negative_count, fetch_limit, completed_at "
            "FROM analyses WHERE phone = :phone ORDER BY id DESC LIMIT 5",
            values={"phone": authenticated_phone},
        )

        # NOTE: "date" is a PostgreSQL reserved word — use analysis_day as alias
        chart_rows = await database.fetch_all(
            "SELECT "
            "SUBSTR(completed_at, 1, 10) AS analysis_day, "
            "SUM(analyzed_count)         AS analyzed, "
            "SUM(negative_count)         AS negative "
            "FROM analyses "
            "WHERE phone = :phone AND completed_at IS NOT NULL "
            "GROUP BY SUBSTR(completed_at, 1, 10) "
            "ORDER BY analysis_day ASC LIMIT 14",
            values={"phone": authenticated_phone},
        )

        return {
            "total_analyzed":  int(summary["total_analyzed"]) if summary else 0,
            "total_negative":  int(summary["total_negative"]) if summary else 0,
            "chats_analyzed":  int(summary["chats_analyzed"]) if summary else 0,
            "recent_analyses": [dict(r) for r in recent],
            "chart_data": [
                {
                    "date":     r["analysis_day"],
                    "analyzed": int(r["analyzed"]),
                    "negative": int(r["negative"]),
                }
                for r in chart_rows
            ],
        }
    except Exception as exc:
        logger.error("Stats fetch failed for %s — %s: %s",
                     mask_phone(authenticated_phone), type(exc).__name__, exc)
        raise HTTPException(status_code=500, detail="Statistika olishda xatolik")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
