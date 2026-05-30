"""
Telegram client pool management.

Responsibilities:
  - LRU cache of active TelegramClient instances (max MAX_ACTIVE_CLIENTS).
  - Health-checking cached clients before reuse.
  - Exponential backoff reconnection logic.
  - Short-lived 2FA pending client store.
  - execute_with_client_retry() for resilient Telegram API calls.
  - _pinned_clients: phones with active monitors are excluded from LRU eviction.
"""
import asyncio
import logging
import random
import time
from collections import OrderedDict
from typing import Any

from fastapi import HTTPException
from telethon import TelegramClient, errors
from telethon.sessions import StringSession

from config import (
    BACKOFF_BASE, BACKOFF_MAX, MAX_ACTIVE_CLIENTS,
    MAX_RETRY_ATTEMPTS, PENDING_2FA_TTL,
    TELETHON_CONNECTION_TIMEOUT, TELETHON_RETRIES, TELETHON_RETRY_DELAY,
)
from database import database, sessions
from db import decrypt_session_string, decrypt_value
from logging_config import mask_phone

logger = logging.getLogger(__name__)

# ── Client pool ───────────────────────────────────────────────────────────────
active_clients: OrderedDict[str, TelegramClient] = OrderedDict()
active_clients_lock = asyncio.Lock()

# Phones whose clients must never be LRU-evicted (active monitors attached).
# Written by moderation_service.attach_monitor / detach_monitor.
_pinned_clients: set[str] = set()

# ── 2FA pending store ─────────────────────────────────────────────────────────
pending_2fa_clients: dict[str, dict[str, Any]] = {}
pending_2fa_lock = asyncio.Lock()

# ── Error taxonomy ────────────────────────────────────────────────────────────
_CONNECTION_ERRORS = (
    ConnectionError,
    errors.RPCError,
    asyncio.TimeoutError,
    OSError,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def is_client_healthy(client: TelegramClient) -> bool:
    try:
        if not client.is_connected():
            return False
        me = await asyncio.wait_for(client.get_me(), timeout=5)
        return me is not None
    except asyncio.TimeoutError:
        logger.warning("Client health-check timeout")
        return False
    except Exception as exc:
        logger.debug("Health check failed: %s", exc)
        return False


def _backoff_delay(attempt: int) -> float:
    return min(BACKOFF_BASE * (2 ** attempt), BACKOFF_MAX) + random.uniform(0, 1)


async def connect_with_retry(
    client: TelegramClient,
    phone: str,
    max_attempts: int = MAX_RETRY_ATTEMPTS,
) -> bool:
    for attempt in range(max_attempts):
        try:
            await client.connect()
            logger.info("Connected for %s (attempt %d)", mask_phone(phone), attempt + 1)
            return True
        except Exception as exc:
            if attempt < max_attempts - 1:
                delay = _backoff_delay(attempt)
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


async def _evict_lru_if_needed() -> None:
    async with active_clients_lock:
        if len(active_clients) < MAX_ACTIVE_CLIENTS:
            return
        # Skip pinned clients; they have active moderation monitors.
        for oldest_phone, oldest_client in list(active_clients.items()):
            if oldest_phone in _pinned_clients:
                continue
            try:
                await oldest_client.disconnect()
            except Exception:
                pass
            del active_clients[oldest_phone]
            logger.info("LRU evicted client for %s", mask_phone(oldest_phone))
            return
        logger.warning(
            "All %d clients are pinned (active monitors); cannot evict.",
            len(active_clients),
        )


async def get_client_session(phone: str) -> TelegramClient:
    """Return a healthy TelegramClient for *phone* (digits-only expected).

    Checks the in-memory pool first. Falls back to DB session decryption
    and a fresh connection with exponential-backoff retry.
    """
    # ── L1: check pool ────────────────────────────────────────────────────────
    cached: TelegramClient | None = None
    async with active_clients_lock:
        if phone in active_clients:
            active_clients.move_to_end(phone)
            cached = active_clients[phone]

    if cached is not None:
        if await is_client_healthy(cached):
            return cached
        logger.warning("Unhealthy cached client for %s — removing", mask_phone(phone))
        async with active_clients_lock:
            if phone in active_clients:
                try:
                    await active_clients[phone].disconnect()
                except Exception:
                    pass
                active_clients.pop(phone, None)

    # ── Fetch session from DB ─────────────────────────────────────────────────
    user = await database.fetch_one(
        sessions.select().where(sessions.c.phone == phone)
    )
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

    # ── Build + connect ───────────────────────────────────────────────────────
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
        logger.warning(
            "is_user_authorized failed for %s: %s", mask_phone(phone), type(exc).__name__
        )
        authorized = False
    if not authorized:
        await client.disconnect()
        raise HTTPException(
            status_code=401,
            detail="Sessiya tugagan yoki yaroqsiz — qayta login qiling",
        )

    # ── Add to pool ───────────────────────────────────────────────────────────
    await _evict_lru_if_needed()
    async with active_clients_lock:
        active_clients[phone] = client

    logger.info("New client created for %s", mask_phone(phone))
    return client


async def execute_with_client_retry(
    client: TelegramClient,
    operation,
    phone: str,
    max_retries: int = 3,
):
    """Run *operation()* with automatic reconnection on transient network errors."""
    for attempt in range(max_retries):
        try:
            return await operation()
        except (errors.AuthKeyUnregisteredError, errors.UserDeactivatedError) as exc:
            logger.warning(
                "Auth key invalidated for %s: %s", mask_phone(phone), type(exc).__name__
            )
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
                raise HTTPException(
                    status_code=502,
                    detail="Telegram serveriga qayta ulanib bo'lmadi",
                )
            logger.error(
                "Client op failed after %d attempts for %s: %s",
                max_retries, mask_phone(phone), type(exc).__name__,
            )
            raise HTTPException(
                status_code=502,
                detail="Telegram bilan bog'lanishda uzluksizlik yuz berdi",
            )


async def shutdown_all_clients() -> None:
    """Gracefully disconnect all pool and 2FA clients.  Call on app shutdown."""
    async with active_clients_lock:
        for phone, client in active_clients.items():
            try:
                if client.is_connected():
                    await client.disconnect()
            except Exception as exc:
                logger.warning("Error disconnecting %s: %s", mask_phone(phone), exc)
        active_clients.clear()

    async with pending_2fa_lock:
        for _phone, entry in pending_2fa_clients.items():
            try:
                await entry["client"].disconnect()
            except Exception:
                pass
        pending_2fa_clients.clear()
