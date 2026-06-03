"""
Veb boshqaruv paneli API:
  GET  /api/chats/banned?chat_id=              — qora ro'yxat (query)
  GET  /api/chats/{chat_id}/banned             — qora ro'yxat (path)
  POST /api/chats/{chat_id}/toggle-restriction — cheklov rejimini yoqish/o'chirish
  POST /api/chats/{chat_id}/unban/{user_id}    — blokdan chiqarish
  POST /api/chats/{chat_id}/mute/{user_id}     — admin qo'lda jimlantirish (ixtiyoriy muddat)
  POST /api/chats/{chat_id}/unmute/{user_id}   — jimlikni olib tashlash
  POST /api/chats/{chat_id}/reset-warns/{user_id} — ogohlantirishlarni nolga tushirish
  GET  /api/chats/{chat_id}/violations         — barcha qoidabuzarlar ro'yxati
  GET  /api/chats/{chat_id}/muted              — jimlantirilganlar ro'yxati
  GET  /api/chats/mute-presets                 — jimlantirish muddati variantlari
  POST /api/chats/{chat_id}/ban                — qo'lda ban qo'shish
  GET  /api/chats/{chat_id}/restriction-mode   — cheklov rejimi holati
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from auth import verify_token
from config import limiter
from models import BannedListResponse, MutedListResponse, ViolationsListResponse
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError

from services.ban_repository import (
    add_banned_user,
    get_restriction_mode,
    is_user_banned,
    list_banned_users,
    set_restriction_mode,
)
from services.bot_service import (
    BotNotConfiguredError,
    get_bot,
    mute_chat_member_telegram,
    unban_chat_member_db_sync,
    unmute_chat_member_telegram,
)
from services.violations_repository import (
    clear_mute_status,
    get_violation,
    list_muted_users,
    list_violations,
    reset_warns,
    set_mute_status,
)

router = APIRouter(prefix="/api", tags=["chats-api"])
logger = logging.getLogger(__name__)


class ToggleRestrictionBody(BaseModel):
    is_enabled: bool


async def _banned_list_payload(chat_id: int) -> BannedListResponse:
    """Frontend: restriction_mode + banned_users bitta ob'ektda."""
    users = await list_banned_users(chat_id)
    restriction_mode = await get_restriction_mode(chat_id)
    return BannedListResponse(
        chat_id=chat_id,
        restriction_mode=restriction_mode,
        banned_users=users,
        count=len(users),
    )


@router.get(
    "/chats/banned",
    response_model=BannedListResponse,
    summary="Qora ro'yxat (query chat_id) — manfiy Telegram ID uchun",
)
@limiter.limit("60/minute")
async def get_banned_users_query(
    request: Request,
    chat_id: int = Query(..., description="Telegram guruh chat_id"),
    authenticated_phone: str = Depends(verify_token),
) -> BannedListResponse:
    _ = authenticated_phone
    if chat_id == 0:
        raise HTTPException(status_code=400, detail="chat_id noto'g'ri")
    try:
        return await _banned_list_payload(chat_id)
    except Exception:
        logger.exception("GET banned (query) failed chat_id=%s", chat_id)
        raise HTTPException(
            status_code=500,
            detail="Bloklangan foydalanuvchilar ro'yxatini olishda xatolik",
        )


@router.get(
    "/chats/{chat_id}/banned",
    response_model=BannedListResponse,
    summary="Qora ro'yxat + kirish cheklovi (frontend: data.restriction_mode, data.banned_users)",
)
@limiter.limit("60/minute")
async def get_banned_users(
    request: Request,
    chat_id: int,
    authenticated_phone: str = Depends(verify_token),
) -> BannedListResponse:
    _ = authenticated_phone
    if chat_id == 0:
        raise HTTPException(status_code=400, detail="chat_id noto'g'ri")
    try:
        return await _banned_list_payload(chat_id)
    except Exception:
        logger.exception("GET banned failed chat_id=%s", chat_id)
        raise HTTPException(
            status_code=500,
            detail="Bloklangan foydalanuvchilar ro'yxatini olishda xatolik",
        )


@router.post("/chats/{chat_id}/toggle-restriction")
@limiter.limit("30/minute")
async def toggle_restriction(
    request: Request,
    chat_id: int,
    body: ToggleRestrictionBody,
    authenticated_phone: str = Depends(verify_token),
):
    """groups_settings.restriction_mode ni yangilaydi."""
    _ = authenticated_phone
    if chat_id == 0:
        raise HTTPException(status_code=400, detail="chat_id noto'g'ri")

    try:
        result = await set_restriction_mode(chat_id, body.is_enabled)
        return {
            "status": "ok",
            "chat_id": result["chat_id"],
            "restriction_mode": result["restriction_mode"],
            "updated_at": result["updated_at"],
        }
    except Exception:
        logger.exception("toggle-restriction failed chat_id=%s", chat_id)
        raise HTTPException(
            status_code=500,
            detail="Cheklov rejimini yangilashda xatolik",
        )


@router.post("/chats/{chat_id}/unban/{user_id}")
@limiter.limit("30/minute")
async def unban_user_api(
    request: Request,
    chat_id: int,
    user_id: int,
    authenticated_phone: str = Depends(verify_token),
):
    """Telegram qora ro'yxatidan chiqaradi va banned_users dan o'chiradi."""
    _ = authenticated_phone
    if chat_id == 0 or user_id == 0:
        raise HTTPException(status_code=400, detail="chat_id yoki user_id noto'g'ri")

    try:
        await unban_chat_member_db_sync(chat_id, user_id)
    except BotNotConfiguredError:
        raise HTTPException(
            status_code=503,
            detail="BOT_TOKEN sozlanmagan — Telegram bot ishga tushirilmagan",
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("unban API failed chat=%s user=%s", chat_id, user_id)
        raise HTTPException(
            status_code=500,
            detail="Blokdan chiqarishda kutilmagan xatolik",
        )

    return {
        "status": "unbanned",
        "chat_id": chat_id,
        "user_id": user_id,
    }


# ── Admin qo'lda jimlantirish ─────────────────────────────────────────────

class MuteBody(BaseModel):
    duration_minutes: int = 60  # default 60 daqiqa

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return v


MUTE_PRESETS = {
    5: "5 daqiqa",
    15: "15 daqiqa",
    30: "30 daqiqa",
    60: "1 soat",
    180: "3 soat",
    360: "6 soat",
    720: "12 soat",
    1440: "1 kun",
    4320: "3 kun",
    10080: "1 hafta",
}


@router.post(
    "/chats/{chat_id}/mute/{user_id}",
    summary="Admin tomonidan foydalanuvchini qo'lda jimlantirish",
)
@limiter.limit("30/minute")
async def mute_user_api(
    request: Request,
    chat_id: int,
    user_id: int,
    body: MuteBody,
    authenticated_phone: str = Depends(verify_token),
):
    """Admin ixtiyoriy muddat bilan foydalanuvchini jimlantiradi.

    duration_minutes: 5, 15, 30, 60, 180, 360, 720, 1440, 4320, 10080
    yoki boshqa ixtiyoriy qiymat (1–10080 oralig'ida).
    """
    if chat_id == 0 or user_id == 0:
        raise HTTPException(status_code=400, detail="chat_id yoki user_id noto'g'ri")
    if body.duration_minutes < 1 or body.duration_minutes > 10080:
        raise HTTPException(
            status_code=400,
            detail="Muddat 1 daqiqadan 10080 daqiqagacha (1 hafta) bo'lishi kerak",
        )

    try:
        muted_until = await mute_chat_member_telegram(
            chat_id, user_id, body.duration_minutes,
            phone=authenticated_phone,
        )
    except BotNotConfiguredError:
        raise HTTPException(
            status_code=503,
            detail="BOT_TOKEN sozlanmagan — Telegram bot ishga tushirilmagan",
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("mute API failed chat=%s user=%s", chat_id, user_id)
        raise HTTPException(
            status_code=500,
            detail="Jimlantirishda kutilmagan xatolik",
        )

    # DB da is_muted=True, muted_until saqlash
    try:
        await set_mute_status(authenticated_phone, chat_id, user_id, muted_until)
    except Exception:
        logger.exception(
            "DB set_mute_status xatosi (Telegram mute muvaffaqiyatli) "
            "chat=%s user=%s", chat_id, user_id,
        )

    return {
        "status": "muted",
        "chat_id": chat_id,
        "user_id": user_id,
        "duration_minutes": body.duration_minutes,
        "muted_until": muted_until,
    }


@router.get(
    "/chats/mute-presets",
    summary="Jimlantirish muddati variantlari",
)
async def get_mute_presets(
    request: Request,
    authenticated_phone: str = Depends(verify_token),
):
    """Frontend uchun jimlantirish muddati variantlarini qaytaradi."""
    _ = authenticated_phone
    return {
        "presets": [
            {"minutes": k, "label": v}
            for k, v in MUTE_PRESETS.items()
        ]
    }


@router.post("/chats/{chat_id}/unmute/{user_id}")
@limiter.limit("30/minute")
async def unmute_user_api(
    request: Request,
    chat_id: int,
    user_id: int,
    authenticated_phone: str = Depends(verify_token),
):
    """Jimlikni olib tashlaydi (Telegram) va violations.is_muted ni yangilaydi."""
    if chat_id == 0 or user_id == 0:
        raise HTTPException(status_code=400, detail="chat_id yoki user_id noto'g'ri")

    row = await get_violation(authenticated_phone, chat_id, user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Qoidabuzarlik yozuvi topilmadi")

    try:
        await unmute_chat_member_telegram(chat_id, user_id, phone=authenticated_phone)
        await clear_mute_status(authenticated_phone, chat_id, user_id)
    except BotNotConfiguredError:
        raise HTTPException(
            status_code=503,
            detail="Server moderatsiya xizmati sozlanmagan",
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("unmute API failed chat=%s user=%s", chat_id, user_id)
        raise HTTPException(
            status_code=500,
            detail="Cheklovni olishda kutilmagan xatolik",
        )

    return {
        "status": "unmuted",
        "chat_id": chat_id,
        "user_id": user_id,
        "is_muted": False,
        "muted_until": None,
    }


@router.post("/chats/{chat_id}/reset-warns/{user_id}")
@limiter.limit("30/minute")
async def reset_warns_api(
    request: Request,
    chat_id: int,
    user_id: int,
    authenticated_phone: str = Depends(verify_token),
):
    """Ogohlantirishlar sonini 0 ga tushiradi, mute taymerini tozalaydi."""
    if chat_id == 0 or user_id == 0:
        raise HTTPException(status_code=400, detail="chat_id yoki user_id noto'g'ri")

    row = await get_violation(authenticated_phone, chat_id, user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Qoidabuzarlik yozuvi topilmadi")

    was_muted = bool(row["is_muted"])

    try:
        updated = await reset_warns(authenticated_phone, chat_id, user_id)
    except Exception:
        logger.exception("reset-warns failed chat=%s user=%s", chat_id, user_id)
        raise HTTPException(
            status_code=500,
            detail="Ogohlantirishlarni tiklashda xatolik",
        )

    if updated is None:
        raise HTTPException(status_code=404, detail="Qoidabuzarlik yozuvi topilmadi")

    # Jimlik Telegramda qolgan bo'lsa, server orqali olib tashlash
    if was_muted:
        try:
            await unmute_chat_member_telegram(chat_id, user_id, phone=authenticated_phone)
        except BotNotConfiguredError:
            logger.warning("reset-warns: BOT_TOKEN yo'q, Telegram unmute o'tkazib yuborildi")
        except (PermissionError, ValueError) as exc:
            logger.warning("reset-warns: Telegram unmute skipped: %s", exc)

    return {
        "status": "warns_reset",
        "chat_id": chat_id,
        **updated,
    }


# ── Qoidabuzarlar ro'yxati ──────────────────────────────────────────────────

@router.get(
    "/chats/{chat_id}/violations",
    response_model=ViolationsListResponse,
    summary="Guruh bo'yicha barcha qoidabuzarlar ro'yxati (warn, mute, ban)",
)
@limiter.limit("60/minute")
async def get_violations_list(
    request: Request,
    chat_id: int,
    authenticated_phone: str = Depends(verify_token),
) -> ViolationsListResponse:
    if chat_id == 0:
        raise HTTPException(status_code=400, detail="chat_id noto'g'ri")
    try:
        items = await list_violations(authenticated_phone, chat_id)
        return ViolationsListResponse(
            chat_id=chat_id,
            violations=items,
            count=len(items),
        )
    except Exception:
        logger.exception("GET violations failed chat_id=%s", chat_id)
        raise HTTPException(
            status_code=500,
            detail="Qoidabuzarlar ro'yxatini olishda xatolik",
        )


# ── Jimlantirilganlar ro'yxati ───────────────────────────────────────────────

@router.get(
    "/chats/{chat_id}/muted",
    response_model=MutedListResponse,
    summary="Guruhda jimlantirilgan foydalanuvchilar ro'yxati",
)
@limiter.limit("60/minute")
async def get_muted_list(
    request: Request,
    chat_id: int,
    authenticated_phone: str = Depends(verify_token),
) -> MutedListResponse:
    if chat_id == 0:
        raise HTTPException(status_code=400, detail="chat_id noto'g'ri")
    try:
        items = await list_muted_users(authenticated_phone, chat_id)
        return MutedListResponse(
            chat_id=chat_id,
            muted_users=items,
            count=len(items),
        )
    except Exception:
        logger.exception("GET muted failed chat_id=%s", chat_id)
        raise HTTPException(
            status_code=500,
            detail="Jimlantirilganlar ro'yxatini olishda xatolik",
        )


# ── Qo'lda ban qo'shish ─────────────────────────────────────────────────────

class ManualBanBody(BaseModel):
    user_id: int
    first_name: str | None = None
    username: str | None = None


@router.post(
    "/chats/{chat_id}/ban",
    summary="Foydalanuvchini qo'lda qora ro'yxatga qo'shish",
)
@limiter.limit("30/minute")
async def ban_user_api(
    request: Request,
    chat_id: int,
    body: ManualBanBody,
    authenticated_phone: str = Depends(verify_token),
):
    if chat_id == 0 or body.user_id == 0:
        raise HTTPException(status_code=400, detail="chat_id yoki user_id noto'g'ri")

    already = await is_user_banned(chat_id, body.user_id)
    if already:
        raise HTTPException(status_code=409, detail="Foydalanuvchi allaqachon qora ro'yxatda")

    try:
        bot = await get_bot()
        await bot.ban_chat_member(chat_id=chat_id, user_id=body.user_id)
    except BotNotConfiguredError:
        raise HTTPException(
            status_code=503,
            detail="BOT_TOKEN sozlanmagan — Telegram bot ishga tushirilmagan",
        )
    except TelegramForbiddenError:
        raise HTTPException(
            status_code=403,
            detail="Botda foydalanuvchini bloklash huquqi yo'q (bot admin bo'lishi kerak)",
        )
    except TelegramBadRequest as exc:
        raise HTTPException(status_code=400, detail=str(exc.message))
    except Exception as exc:
        logger.warning("Telegram ban failed chat=%s user=%s: %s", chat_id, body.user_id, exc)
        raise HTTPException(
            status_code=500,
            detail="Telegramda bloklashda xatolik",
        )

    try:
        await add_banned_user(
            chat_id, body.user_id,
            first_name=body.first_name,
            username=body.username,
        )
    except Exception:
        logger.exception(
            "DB add_banned_user xatosi (Telegram ban muvaffaqiyatli) "
            "chat=%s user=%s", chat_id, body.user_id,
        )

    return {
        "status": "banned",
        "chat_id": chat_id,
        "user_id": body.user_id,
    }


# ── Cheklov rejimini olish ───────────────────────────────────────────────────

@router.get(
    "/chats/{chat_id}/restriction-mode",
    summary="Guruhning restriction_mode holatini olish",
)
@limiter.limit("60/minute")
async def get_restriction_mode_api(
    request: Request,
    chat_id: int,
    authenticated_phone: str = Depends(verify_token),
):
    _ = authenticated_phone
    if chat_id == 0:
        raise HTTPException(status_code=400, detail="chat_id noto'g'ri")
    try:
        mode = await get_restriction_mode(chat_id)
        return {"chat_id": chat_id, "restriction_mode": mode}
    except Exception:
        logger.exception("GET restriction-mode failed chat_id=%s", chat_id)
        raise HTTPException(
            status_code=500,
            detail="Cheklov rejimini olishda xatolik",
        )