from fastapi import FastAPI, Depends, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
import re
import time
import random
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum
import logging
import jwt
from datetime import datetime, timedelta
from uuid import uuid4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pydantic import BaseModel, field_validator, Field
from telethon import TelegramClient, errors
from telethon.sessions import StringSession
from sqlalchemy import Column, Integer, String, Table, MetaData, create_engine
from databases import Database
from transformers import pipeline
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Agar db.py dan ololmasa, xatolik beradi (xavfsizlik uchun)
try:
    from db import DATABASE_URL, encrypt_session_string, decrypt_session_string
except ImportError as e:
    raise ImportError(f"db.py ni import qilib bo'lmadi: {e}. Xavfsizlik uchun dastur to'xtatildi.")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL topilmadi")

# -------------------------
# 1. Database
# -------------------------
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

engine = create_engine(DATABASE_URL)
metadata.create_all(engine)

# -------------------------
# 2. Global o'zgaruvchilar
# -------------------------
sentiment_analyzer = None
model_available = False
# Telegram mijozlarini keshda saqlash (Har safar qayta ulanmaslik uchun)
active_clients: Dict[str, TelegramClient] = {}
active_clients_lock = asyncio.Lock()  # CLIENT POOL PROTECTION
analysis_cache: Dict[str, Dict[str, Any]] = {}
cache_lock = asyncio.Lock()
CACHE_TTL_SECONDS = 3600

# JWT Settings
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY environment variable is required. Please set it in your .env or environment.")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Telethon Connection Settings
TELETHON_CONNECTION_TIMEOUT = 30  # seconds
TELETHON_RETRIES = 3
TELETHON_RETRY_DELAY = 1

# Exponential backoff settings for connection retries
BACKOFF_BASE = 1  # Start with 1 second
BACKOFF_MAX = 30  # Cap at 30 seconds
MAX_RETRY_ATTEMPTS = 5

# -------------------------
# 2.5. Rate Limiting
# -------------------------
limiter = Limiter(key_func=get_remote_address)

# -------------------------
# 3. Negativ so'zlar
# -------------------------
def load_negative_words(file_path: str = "data/uz_negative_words.txt"):
    """Load negative keywords synchronously (called at startup, OK to block)."""
    try:
        if not os.path.exists(file_path):
            logger.warning(f"Negativ so'zlar fayli topilmadi: {file_path}")
            return []
        
        with open(file_path, "r", encoding="utf-8") as f:
            words = [line.strip().lower() for line in f if line.strip()]
        logger.info(f"Loaded {len(words)} negative keywords")
        return words
    except Exception as e:
        logger.exception(f"Error loading negative words: {e}")
        return []

NEGATIVE_KEYWORDS = load_negative_words()
KEYWORD_PATTERNS: List[re.Pattern] = []

AI_NEGATIVE_SCORE_THRESHOLD = 0.85

LEET_MAP = {
    "a": r"[a4@]",
    "b": r"[b8]",
    "c": r"[c(\[]"  ,
    "d": r"[d]",
    "e": r"[e3€]",
    "g": r"[g69q]",
    "h": r"[h#]",
    "i": r"[i1!|]",
    "j": r"[j]",
    "k": r"[k]",
    "l": r"[l1!|]",
    "m": r"[m]",
    "n": r"[n]",
    "o": r"[o0]",
    "p": r"[p]",
    "q": r"[q9]",
    "r": r"[r]",
    "s": r"[s5$]",
    "t": r"[t7+]",
    "u": r"[uüv]",
    "v": r"[v]",
    "x": r"[x×*]",
    "y": r"[y]",
    "z": r"[z2]",
}
SEPARATOR = r"[^\w]*"


def build_keyword_pattern(keyword: str) -> re.Pattern:
    normalized = keyword.strip().lower()
    if len(normalized) <= 2:
        return re.compile(rf"(?<!\w){re.escape(normalized)}(?!\w)", re.IGNORECASE)

    parts = []
    for ch in normalized:
        parts.append(LEET_MAP.get(ch, re.escape(ch)))
    pattern = rf"(?<!\w){SEPARATOR.join(parts)}(?!\w)"
    return re.compile(pattern, re.IGNORECASE)


def is_toxic_by_keywords(text: str) -> bool:
    """FIX #9: One-time lowercase for efficiency"""
    text_lower = text.lower()
    for pattern in KEYWORD_PATTERNS:
        if pattern.search(text_lower):
            return True
    return False

# -------------------------
# 4. Lifespan (App ishga tushishi va to'xtashi)
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Ma'lumotlar bazasiga ulanish
    await database.connect()
    
    # 2. Validate JWT secret key
    if JWT_SECRET_KEY == "your-secret-key-change-in-production":
        logger.warning("⚠️ WARNING: Using default JWT_SECRET_KEY! Set JWT_SECRET_KEY environment variable in production!")
    
    # 3. AI Model lazy loading - try to load but don't block startup
    global sentiment_analyzer, KEYWORD_PATTERNS, model_available
    logger.info("AI Model yuklanmoqda... (Bu biroz vaqt olishi mumkin)")
    try:
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
        )
        model_available = True
        logger.info("AI Model muvaffaqiyatli yuklandi!")
    except Exception:
        model_available = False
        sentiment_analyzer = None
        logger.warning("AI modelni yuklashda xatolik yuz berdi. Faqat keyword-based tahlil ishlatiladi.")

    KEYWORD_PATTERNS = [build_keyword_pattern(k) for k in NEGATIVE_KEYWORDS]
    logger.info(f"{len(KEYWORD_PATTERNS)} keyword patternlari tayyor")

    yield

    # 4. App to'xtaganda barcha Telegram ulanishlarni va DB ni yopish (THREAD-SAFE)
    async with active_clients_lock:
        for phone, client in active_clients.items():
            try:
                if client.is_connected():
                    await client.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting {phone}: {e}")
    
    await database.disconnect()

# -------------------------
# 5. App
# -------------------------
app = FastAPI(
    title="Telegram Sentiment Backend",
    lifespan=lifespan
)

# FIX #7: CORS configuration - proper origin parsing
def get_allowed_origins() -> List[str]:
    """Parse CORS origins from environment variable"""
    origins_str = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000")
    return [o.strip() for o in origins_str.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SlowAPIMiddleware)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# -------------------------
# 6. Pydantic modellari
# -------------------------

# FIX #13: Pydantic v2 migration - use field_validator
class LoginRequest(BaseModel):
    api_id: int
    api_hash: str
    phone: str
    
    @field_validator('api_id')
    @classmethod
    def api_id_positive(cls, v):
        if v <= 0:
            raise ValueError('API ID musbat bo\'lishi kerak')
        return v
    
    @field_validator('api_hash')
    @classmethod
    def api_hash_not_empty(cls, v):
        if not v or len(v) < 10:
            raise ValueError('API Hash juda qisqa')
        return v

class VerifyRequest(BaseModel):
    phone: str
    code: str
    phone_code_hash: str
    api_id: int
    api_hash: str
    
    @field_validator('code')
    @classmethod
    def code_valid(cls, v):
        if not v or len(v) < 3:
            raise ValueError('Kod juda qisqa')
        return v

class AnalyzeRequest(BaseModel):
    phone: str
    chat_id: int
    limit: int = 50
    
    @field_validator('limit')
    @classmethod
    def limit_valid(cls, v):
        if v <= 0 or v > 1000:
            raise ValueError('Limit 1 dan 1000 gacha bo\'lishi kerak')
        return v

class PhoneRequest(BaseModel):
    phone: str

# FIX #8: NegativeMessage validation with enum and Field constraints
class MessageReason(str, Enum):
    """Message analysis reason"""
    KEYWORD_MATCH = "keyword_match"
    AI_SENTIMENT = "ai_sentiment"

class NegativeMessage(BaseModel):
    id: int
    text: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)  # 0-1 range validation
    sender_id: Optional[int] = None
    reason: MessageReason

class AnalyzeResponse(BaseModel):
    analyzed_count: int
    negative_count: int
    negative_messages: List[NegativeMessage]

# -------------------------
# 7. Yordamchi funksiyalar
# -------------------------

# FIX #11: Phone validation with international support (10-15 digits)
def validate_phone(phone: str) -> bool:
    """Phone raqamni validatsiya qiladi"""
    digits_only = re.sub(r'\D', '', phone)
    # International format: 10-15 digits (covers most countries)
    # Uzbekistan: +998XXXXXXXXXX (12 digits)
    return 10 <= len(digits_only) <= 15


async def is_client_healthy(client: TelegramClient) -> bool:
    """Check if Telethon client is actually connected and authorized (CONNECTION HEALTH CHECK)."""
    try:
        # Check if connected
        if not client.is_connected():
            return False
        
        # Check if authorized by getting current user (with timeout)
        try:
            me = await asyncio.wait_for(client.get_me(), timeout=5)
            return me is not None
        except asyncio.TimeoutError:
            logger.warning("Client health check timeout - connection likely dead")
            return False
    except Exception as e:
        logger.debug(f"Health check failed: {e}")
        return False


# FIX #3: Backoff delay with proper jitter using random.random()
def calculate_backoff_delay(attempt: int) -> float:
    """Calculate exponential backoff delay with jitter (EXPONENTIAL BACKOFF)."""
    # Formula: min(base * 2^attempt + random_jitter, max)
    base_delay = min(BACKOFF_BASE * (2 ** attempt), BACKOFF_MAX)
    jitter = random.uniform(0, 1)  # Random value between 0-1
    return base_delay + jitter


async def connect_with_retry(client: TelegramClient, phone: str, max_attempts: int = MAX_RETRY_ATTEMPTS) -> bool:
    """Connect with exponential backoff retry logic."""
    for attempt in range(max_attempts):
        try:
            await client.connect()
            logger.info(f"✓ Connected for {phone} (attempt {attempt + 1})")
            return True
        except Exception as e:
            if attempt < max_attempts - 1:
                delay = calculate_backoff_delay(attempt)
                logger.warning(f"Connection failed attempt {attempt + 1}/{max_attempts}, retrying in {delay:.1f}s: {type(e).__name__}")
                await asyncio.sleep(delay)
            else:
                logger.error(f"Connection failed after {max_attempts} attempts for {phone}: {type(e).__name__}")
                return False
    return False


# FIX #2: Race condition - separate cache check from async health check
async def get_client_session(phone: str) -> TelegramClient:
    """TelegramClient ni keshdan oladi yoki bazadan o'qib yangi ulanish yaratadi (THREAD-SAFE + HEALTH CHECK)."""
    # Check cache WITHOUT doing async operation inside lock
    cached_client = None
    async with active_clients_lock:
        if phone in active_clients:
            cached_client = active_clients[phone]
    
    # Do async health check OUTSIDE the lock
    if cached_client is not None:
        if await is_client_healthy(cached_client):
            logger.debug(f"✓ Using cached healthy client for {phone}")
            return cached_client
        else:
            # Remove unhealthy client from cache
            logger.warning(f"Cached client for {phone} is unhealthy, removing from pool")
            async with active_clients_lock:
                if phone in active_clients:
                    try:
                        await active_clients[phone].disconnect()
                    except Exception:
                        pass
                    active_clients.pop(phone, None)

    # Fetch from database
    query = sessions.select().where(sessions.c.phone == phone)
    user = await database.fetch_one(query)

    if not user:
        raise HTTPException(status_code=404, detail="Avval login qiling")

    # Decrypt session (will raise exception on failure now)
    try:
        session_string = decrypt_session_string(user["session_string"])
    except ValueError as e:
        logger.error(f"Session decryption failed for {phone}: {e}")
        raise HTTPException(status_code=401, detail="Sessiya o'qilishda xatolik - qayta login qiling")
    
    # Create new client
    client = TelegramClient(
        StringSession(session_string),
        user["api_id"],
        user["api_hash"],
        timeout=TELETHON_CONNECTION_TIMEOUT,
        connection_retries=TELETHON_RETRIES,
        retry_delay=TELETHON_RETRY_DELAY
    )

    # Connect with exponential backoff
    if not await connect_with_retry(client, phone):
        raise HTTPException(status_code=502, detail="Telegram serveriga ulanib bo'lmadi. Iltimaas 1-2 daqiqa so'ng qayta urining")

    # Verify authorization
    if not await client.is_user_authorized():
        await client.disconnect()
        raise HTTPException(status_code=401, detail="Sessiya tugagan yoki yaroqsiz - qayta login qiling")

    # Add to cache (THREAD-SAFE)
    async with active_clients_lock:
        active_clients[phone] = client
    
    logger.info(f"✓ New client created and cached for {phone}")
    return client


def get_cache_key(phone: str, chat_id: int, limit: int) -> str:
    return f"{phone}:{chat_id}:{limit}"


# FIX #12: Cache race condition - proper timestamp handling
async def get_cached_analysis(cache_key: str) -> Optional[Dict[str, Any]]:
    async with cache_lock:
        entry = analysis_cache.get(cache_key)
        if not entry:
            return None
        
        # Check expiry with current time
        current_time = time.time()
        if entry["expiry"] < current_time:
            analysis_cache.pop(cache_key, None)
            return None
        return entry["data"]


async def set_cached_analysis(cache_key: str, data: Dict[str, Any]) -> None:
    async with cache_lock:
        analysis_cache[cache_key] = {
            "data": data,
            "expiry": time.time() + CACHE_TTL_SECONDS,
        }


# FIX #4: Infinite loop - proper client reconnection logic
async def execute_with_client_retry(client: TelegramClient, operation, phone: str, max_retries: int = 3):
    """Execute a client operation with automatic reconnection on connection errors."""
    for attempt in range(max_retries):
        try:
            return await operation()
        except (errors.ConnectionError, errors.RPCError, asyncio.TimeoutError) as e:
            if attempt < max_retries - 1:
                logger.warning(f"Client operation failed for {phone} (attempt {attempt + 1}), reconnecting: {type(e).__name__}")
                
                # Disconnect current client
                try:
                    await client.disconnect()
                except Exception:
                    pass
                
                # Try to reconnect
                if await connect_with_retry(client, phone):
                    # Successfully reconnected, retry operation
                    continue
                else:
                    # Failed to reconnect
                    raise HTTPException(status_code=502, detail="Telegram serveriga qayta ulanib bo'lmadi")
            else:
                logger.error(f"Client operation failed after {max_retries} attempts for {phone}: {type(e).__name__}")
                raise HTTPException(status_code=502, detail="Telegram bilan bog'lanishda uzluksizlik yuz berdi")


# FIX #1: verify_token - NOT async dependency, use in endpoint instead
def verify_token(authorization: str = Header(None)) -> str:
    """JWT token'ni tekshirib, phone'ni qaytaradi (SECURE).
    
    Note: This is a SYNC dependency because we can't do async in Depends().
    Database check is moved to endpoint level.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization talab qilinadi")
    
    # Token format: "Bearer <jwt_token>"
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Noto'g'ri token formati")
    
    token = authorization[7:]  # "Bearer " ni olib tashlash
    
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        phone = payload.get("phone")
        if not phone:
            raise HTTPException(status_code=401, detail="Token'da phone topilmadi")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token muddati tugadi")
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(status_code=401, detail="Noto'g'ri token")
    
    return phone


def create_access_token(phone: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT token for phone (SECURE)."""
    if expires_delta is None:
        expires_delta = timedelta(hours=JWT_EXPIRATION_HOURS)
    
    expire = datetime.utcnow() + expires_delta
    payload = {
        "phone": phone,
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": str(uuid4())  # Add unique ID to prevent token reuse
    }
    
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token


# FIX #10: analyze_texts_batch error handling
def analyze_texts_batch(texts: List[str]):
    """Matnlarni bittalab emas, birdaniga (batch) ko'rinishda analiz qiladi."""
    global sentiment_analyzer
    
    if not sentiment_analyzer:
        raise ValueError("AI model is not loaded. Cannot perform analysis.")
    
    try:
        return sentiment_analyzer(texts, truncation=True, max_length=512)
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise


# -------------------------
# 8. Endpointlar
# -------------------------
@app.get("/")
async def read_root(request: Request):
    return {"message": "Backend ishlayapti!"}

@app.get("/health")
async def health(request: Request):
    return {"status": "ok"}

@app.post("/login")
@limiter.limit("5/minute")
async def login(request: Request, data: LoginRequest):
    """Initiate Telegram login by sending SMS code."""
    # Phone validatsiyasi
    if not validate_phone(data.phone):
        raise HTTPException(status_code=400, detail="Noto'g'ri telefon raqami")
    
    client = TelegramClient(
        StringSession(),
        data.api_id,
        data.api_hash,
        timeout=TELETHON_CONNECTION_TIMEOUT,
        connection_retries=TELETHON_RETRIES,
        retry_delay=TELETHON_RETRY_DELAY
    )
    
    # Connect with retry
    if not await connect_with_retry(client, data.phone):
        raise HTTPException(status_code=502, detail="Telegram serveriga ulanib bo'lmadi. Iltimaas 1-2 daqiqa so'ng qayta urining")
    
    try:
        # Send SMS code
        sent = await asyncio.wait_for(
            client.send_code_request(data.phone),
            timeout=10
        )
        session_string = client.session.save()

        # Save to database
        try:
            existing_query = sessions.select().where(sessions.c.phone == data.phone)
            existing = await database.fetch_one(existing_query)
            
            encrypted_session_string = encrypt_session_string(session_string)
            if existing:
                await database.execute(
                    sessions.update()
                    .where(sessions.c.phone == data.phone)
                    .values(api_id=data.api_id, api_hash=data.api_hash, session_string=encrypted_session_string)
                )
            else:
                await database.execute(
                    sessions.insert().values(
                        phone=data.phone,
                        api_id=data.api_id,
                        api_hash=data.api_hash,
                        session_string=encrypted_session_string
                    )
                )
        except Exception as db_err:
            logger.exception("Database error during /login")
            raise HTTPException(status_code=500, detail="Ma'lumotlar bazasiga saqlashda xatolik")

        logger.info(f"✓ SMS code sent to {data.phone}")
        return {
            "status": "waiting_for_code",
            "phone_code_hash": sent.phone_code_hash
        }

    except asyncio.TimeoutError:
        logger.error(f"SMS code request timeout for {data.phone}")
        raise HTTPException(status_code=504, detail="Telegram javaabi kelmadi. Internet ulanishingizni tekshiring")
    except errors.RPCError as e:
        logger.warning(f"Telegram RPC error for {data.phone}: {e}")
        raise HTTPException(status_code=502, detail="Telegram xatosi. Iltimaas qayta urining")
    except Exception as e:
        logger.exception(f"Unexpected error in /login: {e}")
        raise HTTPException(status_code=500, detail="Noma'lum xatolik yuz berdi")
    finally:
        # FIX #6: Cleanup with proper async handling
        try:
            if client.is_connected():
                await client.disconnect()
        except Exception as e:
            logger.warning(f"Failed to disconnect client in finally: {e}")

@app.post("/verify")
@limiter.limit("5/minute")  # CRITICAL: Prevent brute force
async def verify(request: Request, data: VerifyRequest):
    # Phone validatsiyasi
    if not validate_phone(data.phone):
        raise HTTPException(status_code=400, detail="Noto'g'ri telefon raqami")
    
    query = sessions.select().where(sessions.c.phone == data.phone)
    user = await database.fetch_one(query)

    if not user:
        raise HTTPException(status_code=404, detail="Login qilinmagan - avval /login qiling")

    try:
        session_string = decrypt_session_string(user["session_string"])
    except ValueError as e:
        logger.error(f"Session decryption failed for {data.phone}")
        raise HTTPException(status_code=401, detail="Sessiya o'qilishda xatolik - qayta login qiling")
    
    # FIX #14: API credentials verification
    if user["api_id"] != data.api_id or user["api_hash"] != data.api_hash:
        logger.warning(f"API credentials mismatch for {data.phone}")
        raise HTTPException(status_code=401, detail="API credentials mos kelmadi")
    
    client = TelegramClient(
        StringSession(session_string),
        user["api_id"],
        user["api_hash"],
        timeout=TELETHON_CONNECTION_TIMEOUT,
        connection_retries=TELETHON_RETRIES,
        retry_delay=TELETHON_RETRY_DELAY
    )

    # Connect with retry
    if not await connect_with_retry(client, data.phone):
        raise HTTPException(status_code=502, detail="Telegram serveriga ulanib bo'lmadi")

    try:
        # Attempt SMS verification with timeout
        await asyncio.wait_for(
            client.sign_in(
                phone=data.phone,
                code=data.code,
                phone_code_hash=data.phone_code_hash
            ),
            timeout=10
        )
        logger.info(f"✓ SMS verification succeeded for {data.phone}")
        
    except asyncio.TimeoutError:
        await client.disconnect()
        logger.warning(f"SMS verification timeout for {data.phone}")
        raise HTTPException(status_code=504, detail="Telegram javaabi kelmadi")
    
    except errors.SessionPasswordNeededError:
        await client.disconnect()
        logger.warning(f"2FA required for {data.phone}")
        raise HTTPException(status_code=403, detail="2-bosqichli autentifikatsiya talab qilinadi")
    
    except errors.PhoneCodeInvalidError:
        await client.disconnect()
        logger.warning(f"Invalid SMS code for {data.phone}")
        raise HTTPException(status_code=400, detail="SMS kod noto'g'ri")
    
    except errors.PhoneCodeExpiredError:
        await client.disconnect()
        logger.warning(f"SMS code expired for {data.phone}")
        raise HTTPException(status_code=410, detail="SMS kod muddati tugadi. Yangi kod uchun /login qiling")
    
    except errors.PhoneCodeEmpty:
        await client.disconnect()
        logger.warning(f"Empty SMS code for {data.phone}")
        raise HTTPException(status_code=400, detail="SMS kod kiritilmagan")
    
    except errors.FloodWaitError as e:
        await client.disconnect()
        logger.warning(f"Flood wait for {data.phone}: {e.seconds}s")
        raise HTTPException(status_code=429, detail=f"Juda ko'p o'rinish. {e.seconds} soniyadan so'ng qayta uringing")
    
    except errors.RPCError as e:
        await client.disconnect()
        logger.error(f"RPC error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=502, detail="Telegram xatosi")
    
    except Exception as e:
        await client.disconnect()
        logger.exception(f"Unexpected error during verification: {e}")
        raise HTTPException(status_code=500, detail="Noma'lum xatolik")

    # Save session BEFORE adding to cache
    session_string = client.session.save()
    encrypted_session_string = encrypt_session_string(session_string)
    
    try:
        await database.execute(
            sessions.update()
            .where(sessions.c.phone == data.phone)
            .values(session_string=encrypted_session_string)
        )
    except Exception as db_err:
        logger.exception("Database error during session save")
        await client.disconnect()
        raise HTTPException(status_code=500, detail="Ma'lumotlar bazasiga saqlashda xatolik")
    
    # Agar bu phone uchun eski client bo'lsa, avval uzamiz (THREAD-SAFE)
    async with active_clients_lock:
        if data.phone in active_clients:
            old_client = active_clients[data.phone]
            try:
                if old_client.is_connected():
                    await old_client.disconnect()
            except Exception as e:
                logger.warning(f"Failed to disconnect old client: {e}")
        
        # Tasdiqlangach uni aktiv mijozlar qatoriga qo'shamiz
        active_clients[data.phone] = client
    
    # Generate JWT token (SECURE)
    access_token = create_access_token(data.phone)
    return {"status": "success", "token": access_token, "token_type": "bearer", "expires_in": JWT_EXPIRATION_HOURS * 3600}

@app.post("/chats")
async def get_chats(request: Request, data: PhoneRequest, authenticated_phone: str = Depends(verify_token)):
    # FIX #1: Verify token in endpoint, not in dependency
    if authenticated_phone != data.phone:
        raise HTTPException(status_code=403, detail="Token va so'rov phone mos kelmaydi")
    
    # Database validation for token
    query = sessions.select().where(sessions.c.phone == authenticated_phone)
    user = await database.fetch_one(query)
    if not user:
        raise HTTPException(status_code=401, detail="Noto'g'ri token")
    
    try:
        client = await get_client_session(data.phone)
        
        async def get_chats_operation():
            chats = []
            async for dialog in client.iter_dialogs(limit=50):
                chats.append({
                    "id": dialog.id,
                    "title": dialog.title,
                    "type": "Group" if dialog.is_group else "Channel" if dialog.is_channel else "Private"
                })
            return chats
        
        chats = await execute_with_client_retry(client, get_chats_operation, data.phone)
        return {"chats": chats}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat olishda xatolik /chats da")
        raise HTTPException(status_code=500, detail="Chatlarni olishda xatolik yuz berdi")

@app.post("/analyze", response_model=AnalyzeResponse)
@limiter.limit("5/minute")
async def analyze(request: Request, data: AnalyzeRequest, authenticated_phone: str = Depends(verify_token)):
    if authenticated_phone != data.phone:
        raise HTTPException(status_code=403, detail="Token va so'rov phone mos kelmaydi")
    
    # Database validation for token
    query = sessions.select().where(sessions.c.phone == authenticated_phone)
    user = await database.fetch_one(query)
    if not user:
        raise HTTPException(status_code=401, detail="Noto'g'ri token")
    
    cache_key = get_cache_key(data.phone, data.chat_id, data.limit)
    cached = await get_cached_analysis(cache_key)
    if cached is not None:
        # FIX #5: Return proper response model
        return AnalyzeResponse(**cached)

    try:
        client = await get_client_session(data.phone)
        
        # FIX #15: Track message IDs to prevent duplicates
        async def analyze_operation():
            negative_messages: List[NegativeMessage] = []
            uncertain_messages = []
            uncertain_texts = []
            negative_ids: Set[int] = set()  # Track processed message IDs

            try:
                entity = await client.get_entity(data.chat_id)
            except Exception as e:
                logger.exception(f"Chat ID {data.chat_id} ga kira olmadi")
                raise HTTPException(status_code=404, detail="Chat topilmadi yoki kira olmadi")
            
            # Xabarlarni yig'ib olamiz
            try:
                messages = [m async for m in client.iter_messages(entity, limit=data.limit) if m.text]
            except Exception as e:
                logger.exception("Xabarlarni olishda xatolik")
                raise HTTPException(status_code=500, detail="Xabarlarni olishda xatolik yuz berdi")

            if not messages:
                return AnalyzeResponse(analyzed_count=0, negative_count=0, negative_messages=[])

            for msg in messages:
                if is_toxic_by_keywords(msg.text):
                    if msg.id not in negative_ids:
                        negative_messages.append(NegativeMessage(
                            id=msg.id,
                            text=msg.text,
                            confidence=0.0,
                            sender_id=msg.sender_id,
                            reason=MessageReason.KEYWORD_MATCH
                        ))
                        negative_ids.add(msg.id)
                else:
                    uncertain_messages.append(msg)
                    uncertain_texts.append(msg.text)

            if uncertain_texts:
                if model_available:
                    try:
                        results = await asyncio.to_thread(analyze_texts_batch, uncertain_texts)
                        for msg, result in zip(uncertain_messages, results):
                            if msg.id not in negative_ids:  # Check for duplicates
                                score = float(result.get("score", 0.0))
                                if result.get("label") == "NEGATIVE" and score > AI_NEGATIVE_SCORE_THRESHOLD:
                                    negative_messages.append(NegativeMessage(
                                        id=msg.id,
                                        text=msg.text,
                                        confidence=score,
                                        sender_id=msg.sender_id,
                                        reason=MessageReason.AI_SENTIMENT
                                    ))
                                    negative_ids.add(msg.id)
                    except Exception:
                        logger.exception("AI modeli bilan tahlil davomida xato yuz berdi, fallback faqat keyword-based tahlil ishlatiladi")
                else:
                    logger.info("AI modeli mavjud emas, faqat keyword-based tahlil ishlatiladi")

            return AnalyzeResponse(
                analyzed_count=len(messages),
                negative_count=len(negative_messages),
                negative_messages=negative_messages
            )
        
        result = await execute_with_client_retry(client, analyze_operation, data.phone)
        
        # FIX #5: Cache response as dict
        cache_data = {
            "analyzed_count": result.analyzed_count,
            "negative_count": result.negative_count,
            "negative_messages": [msg.dict() for msg in result.negative_messages]
        }
        await set_cached_analysis(cache_key, cache_data)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Analyze da kutilmagan xatolik")
        raise HTTPException(status_code=500, detail="Tahlil davomida xatolik yuz berdi")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
