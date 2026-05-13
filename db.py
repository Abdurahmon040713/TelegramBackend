import base64
import hashlib
import logging
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SESSION_ENCRYPTION_KEY = os.getenv("SESSION_ENCRYPTION_KEY")
if not SESSION_ENCRYPTION_KEY:
    raise ValueError("SESSION_ENCRYPTION_KEY .env faylida aniqlanmagan")

if len(SESSION_ENCRYPTION_KEY) < 16:
    raise ValueError(
        f"SESSION_ENCRYPTION_KEY juda qisqa "
        f"(minimal 16 belgi, siz {len(SESSION_ENCRYPTION_KEY)} bergan)"
    )

# PBKDF2-HMAC-SHA256  — 600 000 iterations (NIST SP 800-132, 2023 recommendation).
# Fixed domain salt keeps this key separate from any other PBKDF2 usage.
_PBKDF2_SALT = b"chatsphere_v1_kdf"
SESSION_ENCRYPTION_KEY_BYTES = hashlib.pbkdf2_hmac(
    "sha256",
    SESSION_ENCRYPTION_KEY.encode("utf-8"),
    _PBKDF2_SALT,
    iterations=600_000,
    dklen=32,
)

# Legacy key used before the PBKDF2 upgrade (raw SHA-256, no salt).
# Used ONLY inside the startup migration to re-encrypt old database rows.
# Remove this constant once the migration has run on all environments.
_LEGACY_KEY_BYTES = hashlib.sha256(SESSION_ENCRYPTION_KEY.encode("utf-8")).digest()

logger.info(
    "Encryption key ready: %d chars → 256-bit PBKDF2 key", len(SESSION_ENCRYPTION_KEY)
)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./telegram_app.db")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)


# ── low-level helpers ────────────────────────────────────────────────────────

def _encrypt(value: str, key_bytes: bytes) -> str:
    aesgcm = AESGCM(key_bytes)
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, value.encode("utf-8"), None)
    return base64.urlsafe_b64encode(nonce + ciphertext).decode("utf-8")


def _decrypt(encrypted_value: str, key_bytes: bytes) -> str:
    if not encrypted_value:
        raise ValueError("Encrypted value is empty")
    decoded = base64.urlsafe_b64decode(encrypted_value.encode("utf-8"))
    if len(decoded) < 13:
        raise ValueError(f"Invalid encrypted length: {len(decoded)} bytes")
    nonce, ciphertext = decoded[:12], decoded[12:]
    try:
        return AESGCM(key_bytes).decrypt(nonce, ciphertext, None).decode("utf-8")
    except Exception as exc:
        raise ValueError(f"Decryption failed: {type(exc).__name__}") from exc


# ── public API (current PBKDF2 key) ─────────────────────────────────────────

def encrypt_value(value: str) -> str:
    return _encrypt(value, SESSION_ENCRYPTION_KEY_BYTES)


def decrypt_value(encrypted_value: str) -> str:
    try:
        return _decrypt(encrypted_value, SESSION_ENCRYPTION_KEY_BYTES)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Cannot decrypt value: {type(exc).__name__}") from exc


def encrypt_session_string(session_string: str) -> str:
    return _encrypt(session_string, SESSION_ENCRYPTION_KEY_BYTES)


def decrypt_session_string(encrypted_value: str) -> str:
    """Raises ValueError on failure — caller must handle or bubble up."""
    if not encrypted_value:
        raise ValueError("Session string is empty")
    try:
        return _decrypt(encrypted_value, SESSION_ENCRYPTION_KEY_BYTES)
    except ValueError:
        raise
    except Exception as exc:
        logger.error("Session decryption error: %s", type(exc).__name__)
        raise ValueError(f"Cannot decrypt session: {type(exc).__name__}") from exc


# ── migration helpers (legacy SHA-256 key) ───────────────────────────────────

def decrypt_value_legacy(encrypted_value: str) -> str:
    """Decrypt a value that was encrypted with the old SHA-256 key."""
    return _decrypt(encrypted_value, _LEGACY_KEY_BYTES)


def decrypt_session_string_legacy(encrypted_value: str) -> str:
    """Decrypt a session string that was encrypted with the old SHA-256 key."""
    if not encrypted_value:
        raise ValueError("Session string is empty")
    return _decrypt(encrypted_value, _LEGACY_KEY_BYTES)
