"""
Authentication routes: /login  /verify  /verify-2fa

All three share 5/minute rate limiting.  Credential validation, Telegram
client lifecycle, and 2FA state management live here.  Business logic that
is also needed elsewhere lives in auth.py / telegram_service.py.
"""
import asyncio
import logging
import time

from fastapi import APIRouter, HTTPException, Request
from telethon import TelegramClient, errors
from telethon.sessions import StringSession

from auth import create_access_token, normalize_phone, validate_phone
from config import (
    JWT_EXPIRATION_HOURS, PENDING_2FA_TTL,
    TELETHON_CONNECTION_TIMEOUT, TELETHON_RETRIES, TELETHON_RETRY_DELAY,
    limiter,
)
from database import database, sessions
from db import (
    decrypt_session_string, decrypt_value,
    encrypt_session_string, encrypt_value,
)
from logging_config import mask_phone
from models import LoginRequest, TwoFARequest, VerifyRequest
from services.telegram_service import (
    active_clients, active_clients_lock,
    connect_with_retry,
    pending_2fa_clients, pending_2fa_lock,
)

router = APIRouter(tags=["auth"])
logger = logging.getLogger(__name__)


@router.post("/login")
@limiter.limit("5/minute")
async def login(request: Request, data: LoginRequest):
    """Send an SMS verification code to the given phone number."""
    if not validate_phone(data.phone):
        raise HTTPException(status_code=400, detail="Noto'g'ri telefon raqami")

    normalized = normalize_phone(data.phone)

    client = TelegramClient(
        StringSession(), data.api_id, data.api_hash,
        timeout=TELETHON_CONNECTION_TIMEOUT,
        connection_retries=TELETHON_RETRIES,
        retry_delay=TELETHON_RETRY_DELAY,
    )

    if not await connect_with_retry(client, normalized):
        raise HTTPException(
            status_code=502,
            detail="Telegram serveriga ulanib bo'lmadi. 1-2 daqiqadan so'ng qayta urining",
        )

    try:
        sent = await asyncio.wait_for(client.send_code_request(data.phone), timeout=10)
        encrypted_session = encrypt_session_string(client.session.save())
        encrypted_hash = encrypt_value(data.api_hash)

        try:
            existing = await database.fetch_one(
                sessions.select().where(sessions.c.phone == normalized)
            )
            if existing:
                await database.execute(
                    sessions.update().where(sessions.c.phone == normalized).values(
                        api_id=data.api_id, api_hash=encrypted_hash,
                        session_string=encrypted_session,
                    )
                )
            else:
                await database.execute(
                    sessions.insert().values(
                        phone=normalized, api_id=data.api_id,
                        api_hash=encrypted_hash, session_string=encrypted_session,
                    )
                )
        except Exception:
            logger.exception("DB error during /login for %s", mask_phone(normalized))
            raise HTTPException(status_code=500, detail="Ma'lumotlar bazasiga saqlashda xatolik")

        logger.info("SMS code sent to %s", mask_phone(normalized))
        return {"status": "waiting_for_code", "phone_code_hash": sent.phone_code_hash}

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Telegram javobi kelmadi")
    except errors.RPCError as exc:
        logger.warning("Telegram RPC error for %s: %s", mask_phone(normalized), exc)
        raise HTTPException(status_code=502, detail="Telegram xatosi. Qayta urining")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error in /login for %s", mask_phone(normalized))
        raise HTTPException(status_code=500, detail="Noma'lum xatolik yuz berdi")
    finally:
        try:
            if client.is_connected():
                await client.disconnect()
        except Exception:
            pass


@router.post("/verify")
@limiter.limit("5/minute")
async def verify(request: Request, data: VerifyRequest):
    """Confirm the SMS code.  Returns JWT on success or 2FA flag if needed."""
    if not validate_phone(data.phone):
        raise HTTPException(status_code=400, detail="Noto'g'ri telefon raqami")

    normalized = normalize_phone(data.phone)

    user = await database.fetch_one(
        sessions.select().where(sessions.c.phone == normalized)
    )
    if not user:
        raise HTTPException(status_code=404, detail="Login qilinmagan — avval /login qiling")

    try:
        session_string = decrypt_session_string(user["session_string"])
    except ValueError:
        logger.error("Session decryption failed for %s", mask_phone(normalized))
        raise HTTPException(
            status_code=401, detail="Sessiya o'qilishda xatolik — qayta login qiling"
        )

    try:
        stored_hash = decrypt_value(user["api_hash"])
    except Exception:
        raise HTTPException(
            status_code=401, detail="API credentials o'qilishda xatolik — qayta login qiling"
        )

    if user["api_id"] != data.api_id or stored_hash != data.api_hash:
        logger.warning("API credential mismatch for %s", mask_phone(normalized))
        raise HTTPException(status_code=401, detail="API credentials mos kelmadi")

    client = TelegramClient(
        StringSession(session_string), user["api_id"], stored_hash,
        timeout=TELETHON_CONNECTION_TIMEOUT,
        connection_retries=TELETHON_RETRIES,
        retry_delay=TELETHON_RETRY_DELAY,
    )

    if not await connect_with_retry(client, normalized):
        raise HTTPException(status_code=502, detail="Telegram serveriga ulanib bo'lmadi")

    try:
        await asyncio.wait_for(
            client.sign_in(
                phone=data.phone, code=data.code,
                phone_code_hash=data.phone_code_hash,
            ),
            timeout=10,
        )
        logger.info("SMS verification OK for %s", mask_phone(normalized))

    except errors.SessionPasswordNeededError:
        async with pending_2fa_lock:
            pending_2fa_clients[normalized] = {
                "client": client,
                "expires_at": time.time() + PENDING_2FA_TTL,
            }
        logger.info("2FA required for %s — pending client saved", mask_phone(normalized))
        return {"status": "2fa_required"}

    except asyncio.TimeoutError:
        await client.disconnect()
        raise HTTPException(status_code=504, detail="Telegram javobi kelmadi")
    except errors.PhoneCodeInvalidError:
        await client.disconnect()
        raise HTTPException(status_code=400, detail="SMS kod noto'g'ri")
    except errors.PhoneCodeExpiredError:
        await client.disconnect()
        raise HTTPException(status_code=410, detail="SMS kod muddati tugadi. Yangi /login qiling")
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
        logger.error("RPC error for %s: %s", mask_phone(normalized), exc)
        raise HTTPException(status_code=502, detail="Telegram xatosi")
    except Exception:
        await client.disconnect()
        logger.exception("Unexpected /verify error for %s", mask_phone(normalized))
        raise HTTPException(status_code=500, detail="Noma'lum xatolik")

    # Persist updated session
    try:
        await database.execute(
            sessions.update().where(sessions.c.phone == normalized).values(
                session_string=encrypt_session_string(client.session.save())
            )
        )
    except Exception:
        await client.disconnect()
        logger.exception("DB error saving session for %s", mask_phone(normalized))
        raise HTTPException(status_code=500, detail="Sessiyani saqlashda xatolik")

    async with active_clients_lock:
        old = active_clients.get(normalized)
        if old:
            try:
                if old.is_connected():
                    await old.disconnect()
            except Exception:
                pass
        active_clients[normalized] = client

    token = create_access_token(normalized)
    return {
        "status": "success", "token": token, "token_type": "bearer",
        "expires_in": JWT_EXPIRATION_HOURS * 3600,
    }


@router.post("/verify-2fa")
@limiter.limit("5/minute")
async def verify_2fa(request: Request, data: TwoFARequest):
    """Complete login for accounts that require a Telegram cloud password."""
    if not validate_phone(data.phone):
        raise HTTPException(status_code=400, detail="Noto'g'ri telefon raqami")

    normalized = normalize_phone(data.phone)

    async with pending_2fa_lock:
        entry = pending_2fa_clients.get(normalized)

    if not entry:
        raise HTTPException(
            status_code=404,
            detail="2FA sessiyasi topilmadi. Qayta /login va /verify qiling.",
        )
    if entry["expires_at"] < time.time():
        async with pending_2fa_lock:
            pending_2fa_clients.pop(normalized, None)
        raise HTTPException(
            status_code=410,
            detail="2FA muddati tugadi (5 daqiqa). Qayta login qiling.",
        )

    client: TelegramClient = entry["client"]

    try:
        await asyncio.wait_for(client.sign_in(password=data.password), timeout=10)
        logger.info("2FA succeeded for %s", mask_phone(normalized))
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
        logger.exception("2FA failed for %s", mask_phone(normalized))
        raise HTTPException(status_code=500, detail="2FA tekshirishda xatolik")

    async with pending_2fa_lock:
        pending_2fa_clients.pop(normalized, None)

    try:
        await database.execute(
            sessions.update().where(sessions.c.phone == normalized).values(
                session_string=encrypt_session_string(client.session.save())
            )
        )
    except Exception:
        await client.disconnect()
        logger.exception("DB error saving 2FA session for %s", mask_phone(normalized))
        raise HTTPException(status_code=500, detail="Sessiyani saqlashda xatolik")

    async with active_clients_lock:
        old = active_clients.get(normalized)
        if old:
            try:
                if old.is_connected():
                    await old.disconnect()
            except Exception:
                pass
        active_clients[normalized] = client

    token = create_access_token(normalized)
    return {
        "status": "success", "token": token, "token_type": "bearer",
        "expires_in": JWT_EXPIRATION_HOURS * 3600,
    }
