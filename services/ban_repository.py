"""
groups_settings va banned_users jadvallari uchun DB operatsiyalari.
"""
from datetime import datetime, timezone
from typing import Any

from database import banned_users, database, groups_settings


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def get_restriction_mode(chat_id: int) -> bool:
    row = await database.fetch_one(
        groups_settings.select().where(groups_settings.c.chat_id == chat_id)
    )
    if row is None:
        return False
    return bool(row["restriction_mode"])


async def set_restriction_mode(chat_id: int, is_enabled: bool) -> dict[str, Any]:
    now = _utc_now()
    existing = await database.fetch_one(
        groups_settings.select().where(groups_settings.c.chat_id == chat_id)
    )
    if existing is None:
        await database.execute(
            groups_settings.insert().values(
                chat_id=chat_id,
                restriction_mode=is_enabled,
                updated_at=now,
            )
        )
    else:
        await database.execute(
            groups_settings.update()
            .where(groups_settings.c.chat_id == chat_id)
            .values(restriction_mode=is_enabled, updated_at=now)
        )
    return {"chat_id": chat_id, "restriction_mode": is_enabled, "updated_at": now}


async def list_banned_users(chat_id: int) -> list[dict[str, Any]]:
    rows = await database.fetch_all(
        banned_users.select()
        .where(banned_users.c.chat_id == chat_id)
        .order_by(banned_users.c.banned_at.desc())
    )
    return [
        {
            "user_id": r["user_id"],
            "first_name": r["first_name"],
            "username": r["username"],
            "banned_at": r["banned_at"],
        }
        for r in rows
    ]


async def is_user_banned(chat_id: int, user_id: int) -> bool:
    row = await database.fetch_one(
        banned_users.select()
        .where(banned_users.c.chat_id == chat_id)
        .where(banned_users.c.user_id == user_id)
    )
    return row is not None


async def add_banned_user(
    chat_id: int,
    user_id: int,
    *,
    first_name: str | None = None,
    username: str | None = None,
) -> None:
    now = _utc_now()
    existing = await database.fetch_one(
        banned_users.select()
        .where(banned_users.c.chat_id == chat_id)
        .where(banned_users.c.user_id == user_id)
    )
    if existing is None:
        await database.execute(
            banned_users.insert().values(
                chat_id=chat_id,
                user_id=user_id,
                first_name=first_name,
                username=username,
                banned_at=now,
            )
        )
    else:
        await database.execute(
            banned_users.update()
            .where(banned_users.c.chat_id == chat_id)
            .where(banned_users.c.user_id == user_id)
            .values(
                first_name=first_name or existing["first_name"],
                username=username or existing["username"],
                banned_at=now,
            )
        )


async def remove_banned_user(chat_id: int, user_id: int) -> bool:
    row = await database.fetch_one(
        banned_users.select()
        .where(banned_users.c.chat_id == chat_id)
        .where(banned_users.c.user_id == user_id)
    )
    if row is None:
        return False
    await database.execute(
        banned_users.delete()
        .where(banned_users.c.chat_id == chat_id)
        .where(banned_users.c.user_id == user_id)
    )
    return True