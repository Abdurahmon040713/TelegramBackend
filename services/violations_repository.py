"""
violations jadvali — mute holati va ogohlantirishlar (warn_count).
"""
import logging
from typing import Any, Optional

from databases.interfaces import Record

from database import database, violations_table

logger = logging.getLogger(__name__)


async def get_violation(phone: str, chat_id: int, user_id: int) -> Optional[Record]:
    """Bitta foydalanuvchining qoidabuzarlik yozuvini qaytaradi (yoki None)."""
    return await database.fetch_one(
        violations_table.select()
        .where(violations_table.c.phone == phone)
        .where(violations_table.c.chat_id == chat_id)
        .where(violations_table.c.user_id == user_id)
    )


async def list_violations(phone: str, chat_id: int) -> list[dict[str, Any]]:
    """Guruh bo'yicha barcha qoidabuzarlik yozuvlarini qaytaradi."""
    rows = await database.fetch_all(
        violations_table.select()
        .where(violations_table.c.phone == phone)
        .where(violations_table.c.chat_id == chat_id)
        .order_by(violations_table.c.warn_count.desc())
    )
    return [
        {
            "user_id": r["user_id"],
            "first_name": r["first_name"],
            "username": r["username"],
            "warn_count": r["warn_count"],
            "is_muted": bool(r["is_muted"]),
            "is_banned": bool(r["is_banned"]),
            "muted_until": r["muted_until"],
        }
        for r in rows
    ]


async def list_muted_users(phone: str, chat_id: int) -> list[dict[str, Any]]:
    """Faqat jimlantirilgan (is_muted=True) foydalanuvchilarni qaytaradi."""
    rows = await database.fetch_all(
        violations_table.select()
        .where(violations_table.c.phone == phone)
        .where(violations_table.c.chat_id == chat_id)
        .where(violations_table.c.is_muted == True)
        .order_by(violations_table.c.muted_until.desc())
    )
    return [
        {
            "user_id": r["user_id"],
            "first_name": r["first_name"],
            "username": r["username"],
            "warn_count": r["warn_count"],
            "is_muted": True,
            "is_banned": bool(r["is_banned"]),
            "muted_until": r["muted_until"],
        }
        for r in rows
    ]


async def list_banned_violations(phone: str, chat_id: int) -> list[dict[str, Any]]:
    """Faqat banlangan (is_banned=True) foydalanuvchilarni violations jadvalidan qaytaradi."""
    rows = await database.fetch_all(
        violations_table.select()
        .where(violations_table.c.phone == phone)
        .where(violations_table.c.chat_id == chat_id)
        .where(violations_table.c.is_banned == True)
        .order_by(violations_table.c.warn_count.desc())
    )
    return [
        {
            "user_id": r["user_id"],
            "first_name": r["first_name"],
            "username": r["username"],
            "warn_count": r["warn_count"],
            "is_muted": bool(r["is_muted"]),
            "is_banned": True,
            "muted_until": r["muted_until"],
        }
        for r in rows
    ]


async def set_mute_status(
    phone: str, chat_id: int, user_id: int, muted_until: str
) -> bool:
    """is_muted=True, muted_until=<ISO string>. Yozuv yo'q bo'lsa yaratadi."""
    row = await get_violation(phone, chat_id, user_id)
    if row is None:
        # Violations yozuvi yo'q — yangi yaratish (admin qo'lda mute qilganda)
        await database.execute(
            violations_table.insert().values(
                phone=phone,
                chat_id=chat_id,
                user_id=user_id,
                warn_count=0,
                is_muted=True,
                is_banned=False,
                muted_until=muted_until,
            )
        )
        return True
    await database.execute(
        violations_table.update()
        .where(violations_table.c.phone == phone)
        .where(violations_table.c.chat_id == chat_id)
        .where(violations_table.c.user_id == user_id)
        .values(is_muted=True, muted_until=muted_until)
    )
    return True


async def clear_mute_status(phone: str, chat_id: int, user_id: int) -> bool:
    """is_muted=False, muted_until=None. UPDATE to'g'ridan-to'g'ri — ortiqcha SELECT yo'q."""
    result = await database.execute(
        violations_table.update()
        .where(violations_table.c.phone == phone)
        .where(violations_table.c.chat_id == chat_id)
        .where(violations_table.c.user_id == user_id)
        .values(is_muted=False, muted_until=None)
    )
    # execute() yangilangan qatorlar sonini qaytaradi (int yoki rowcount)
    updated = result if isinstance(result, int) else 0
    if updated == 0:
        logger.debug(
            "clear_mute_status: yozuv topilmadi phone=...%s chat=%s user=%s",
            phone[-4:], chat_id, user_id,
        )
        return False
    return True


async def reset_warns(phone: str, chat_id: int, user_id: int) -> dict[str, Any] | None:
    """warn_count=0, mute maydonlarini ham tozalaydi."""
    row = await get_violation(phone, chat_id, user_id)
    if row is None:
        return None
    await database.execute(
        violations_table.update()
        .where(violations_table.c.phone == phone)
        .where(violations_table.c.chat_id == chat_id)
        .where(violations_table.c.user_id == user_id)
        .values(
            warn_count=0,
            is_muted=False,
            muted_until=None,
        )
    )
    return {
        "user_id": user_id,
        "warn_count": 0,
        "is_muted": False,
        "is_banned": bool(row["is_banned"]),
        "muted_until": None,
    }