"""
Telegram routes:
  POST /chats            — list the user's dialogs
  POST /monitor/start    — attach real-time moderation to a chat
  POST /monitor/stop     — detach moderation from a chat
  GET  /monitor/status   — list chats currently being moderated
  GET  /violations       — fetch violation records for a chat
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from telethon import errors

from auth import normalize_phone, verify_token
from config import limiter
from database import database, sessions, violations_table
from logging_config import mask_phone
from models import MonitorRequest, PhoneRequest, ViolationRecord
from services.moderation_service import (
    attach_monitor,
    detach_monitor,
    get_active_monitors,
)
from services.telegram_service import (
    _CONNECTION_ERRORS,
    active_clients, active_clients_lock,
    execute_with_client_retry,
    get_client_session,
)

router = APIRouter(tags=["telegram"])
logger = logging.getLogger(__name__)


async def _require_session(normalized_phone: str, authenticated_phone: str) -> None:
    """Guard: token owner matches request phone AND session row exists."""
    if authenticated_phone != normalized_phone:
        raise HTTPException(status_code=403, detail="Token va so'rov phone mos kelmaydi")
    user = await database.fetch_one(
        sessions.select().where(sessions.c.phone == normalized_phone)
    )
    if not user:
        raise HTTPException(status_code=401, detail="Noto'g'ri token")


# ── Chat list ─────────────────────────────────────────────────────────────────

@router.post("/chats")
@limiter.limit("10/minute")
async def get_chats(
    request: Request,
    data: PhoneRequest,
    authenticated_phone: str = Depends(verify_token),
):
    normalized = normalize_phone(data.phone)
    await _require_session(normalized, authenticated_phone)

    try:
        client = await get_client_session(normalized)

        async def _op():
            chats = []
            async for dialog in client.iter_dialogs(limit=100):
                chats.append({
                    "id":    dialog.id,
                    "title": dialog.title,
                    "type": (
                        "Group"   if dialog.is_group   else
                        "Channel" if dialog.is_channel else
                        "Private"
                    ),
                })
            return chats

        chats = await execute_with_client_retry(client, _op, normalized)
        return {"chats": chats}

    except HTTPException:
        raise
    except (errors.AuthKeyUnregisteredError, errors.UserDeactivatedError) as exc:
        logger.warning("Auth invalidated for %s: %s", mask_phone(normalized), type(exc).__name__)
        async with active_clients_lock:
            active_clients.pop(normalized, None)
        raise HTTPException(
            status_code=401,
            detail="Telegram sessiyasi bekor qilindi. Qayta login qiling",
        )
    except _CONNECTION_ERRORS:
        raise HTTPException(
            status_code=502,
            detail="Telegram serveriga ulanishda xatolik. 1-2 daqiqadan so'ng qayta urining",
        )
    except Exception:
        logger.exception("Unexpected error /chats for %s", mask_phone(normalized))
        raise HTTPException(status_code=500, detail="Chatlarni olishda kutilmagan xatolik")


# ── Monitor: start ────────────────────────────────────────────────────────────

@router.post("/monitor/start")
@limiter.limit("10/minute")
async def monitor_start(
    request: Request,
    data: MonitorRequest,
    authenticated_phone: str = Depends(verify_token),
):
    """Attach a real-time moderation handler to the given chat."""
    normalized = normalize_phone(data.phone)
    await _require_session(normalized, authenticated_phone)

    try:
        client = await get_client_session(normalized)
    except HTTPException:
        raise

    # Verify we can actually reach this chat before wiring the handler.
    try:
        await client.get_entity(data.chat_id)
    except Exception:
        raise HTTPException(
            status_code=404, detail="Chat topilmadi yoki kira olmadi"
        )

    already = not await attach_monitor(client, normalized, data.chat_id)
    if already:
        return {"status": "already_monitoring", "chat_id": data.chat_id}

    logger.info(
        "Monitoring started: phone=%s  chat=%d", mask_phone(normalized), data.chat_id
    )
    return {"status": "monitoring_started", "chat_id": data.chat_id}


# ── Monitor: stop ─────────────────────────────────────────────────────────────

@router.post("/monitor/stop")
@limiter.limit("10/minute")
async def monitor_stop(
    request: Request,
    data: MonitorRequest,
    authenticated_phone: str = Depends(verify_token),
):
    """Detach the moderation handler from the given chat."""
    normalized = normalize_phone(data.phone)
    await _require_session(normalized, authenticated_phone)

    try:
        client = await get_client_session(normalized)
    except HTTPException:
        raise

    stopped = await detach_monitor(client, normalized, data.chat_id)
    if not stopped:
        return {"status": "not_monitoring", "chat_id": data.chat_id}

    logger.info(
        "Monitoring stopped: phone=%s  chat=%d", mask_phone(normalized), data.chat_id
    )
    return {"status": "monitoring_stopped", "chat_id": data.chat_id}


# ── Monitor: status ───────────────────────────────────────────────────────────

@router.get("/monitor/status")
@limiter.limit("30/minute")
async def monitor_status(
    request: Request,
    authenticated_phone: str = Depends(verify_token),
):
    """Return the list of chat_ids currently being moderated by this account."""
    chats = await get_active_monitors(authenticated_phone)
    return {"phone": authenticated_phone, "monitored_chats": chats}


# ── Violations ────────────────────────────────────────────────────────────────

@router.get("/violations/{chat_id}")
@limiter.limit("30/minute")
async def get_violations(
    request: Request,
    chat_id: int,
    authenticated_phone: str = Depends(verify_token),
):
    """Return all violation records for a chat managed by the authenticated account."""
    try:
        rows = await database.fetch_all(
            violations_table.select()
            .where(violations_table.c.phone == authenticated_phone)
            .where(violations_table.c.chat_id == chat_id)
            .order_by(violations_table.c.warn_count.desc())
        )
        return {
            "chat_id": chat_id,
            "violations": [
                ViolationRecord(
                    user_id=r["user_id"],
                    warn_count=r["warn_count"],
                    is_muted=r["is_muted"],
                    is_banned=r["is_banned"],
                    muted_until=r["muted_until"],
                ).model_dump()
                for r in rows
            ],
        }
    except Exception:
        logger.exception(
            "Violations fetch failed for phone=%s chat=%d",
            mask_phone(authenticated_phone), chat_id,
        )
        raise HTTPException(status_code=500, detail="Qoidabuzarlar ro'yxatini olishda xatolik")
