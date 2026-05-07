import base64
import hashlib
import logging
import os
from dotenv import load_dotenv
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# .env faylni o'qiymiz (faqat lokal kompyuterda ishlaganda kerak bo'ladi)
load_dotenv()

logger = logging.getLogger(__name__)

SESSION_ENCRYPTION_KEY = os.getenv("SESSION_ENCRYPTION_KEY")
if not SESSION_ENCRYPTION_KEY:
    raise ValueError("SESSION_ENCRYPTION_KEY .env faylida aniqlanmagan")

if len(SESSION_ENCRYPTION_KEY) < 16:
    raise ValueError(f"SESSION_ENCRYPTION_KEY juda qisqa (minimal 16 character, siz {len(SESSION_ENCRYPTION_KEY)} bergan)")

SESSION_ENCRYPTION_KEY_BYTES = hashlib.sha256(SESSION_ENCRYPTION_KEY.encode("utf-8")).digest()
logger.info(f"✓ Encryption key validated: {len(SESSION_ENCRYPTION_KEY)} chars -> 256-bit hash")

# Render-dagi Environment Variable-ni oladi
# Agar topilmasa, lokal sqlite bazasidan foydalanadi
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./telegram_app.db")

# SQLAlchemy uchun postgres:// ni postgresql:// ga aylantirish (juda muhim!)
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)


def encrypt_session_string(session_string: str) -> str:
    aesgcm = AESGCM(SESSION_ENCRYPTION_KEY_BYTES)
    nonce = os.urandom(12)
    encrypted_data = aesgcm.encrypt(nonce, session_string.encode("utf-8"), None)
    return base64.urlsafe_b64encode(nonce + encrypted_data).decode("utf-8")


def decrypt_session_string(encrypted_value: str) -> str:
    """Decrypt session string - raises exception on failure (no fallback)."""
    if not encrypted_value:
        raise ValueError("Session string is empty")
    
    try:
        decoded = base64.urlsafe_b64decode(encrypted_value.encode("utf-8"))
        if len(decoded) < 13:
            raise ValueError(f"Invalid session length: {len(decoded)} bytes")
        
        nonce = decoded[:12]
        ciphertext = decoded[12:]
        aesgcm = AESGCM(SESSION_ENCRYPTION_KEY_BYTES)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None).decode("utf-8")
        logger.debug(f"✓ Session decrypted: {len(ciphertext)} -> {len(plaintext)} chars")
        return plaintext
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Decryption error: {type(e).__name__}: {str(e)[:50]}")
        raise ValueError(f"Cannot decrypt session: {type(e).__name__}")