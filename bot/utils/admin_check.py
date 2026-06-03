"""Guruh administratori yoki egasi ekanligini tekshirish."""
from aiogram import Bot
from aiogram.enums import ChatMemberStatus
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from aiogram.types import ChatMemberAdministrator, ChatMemberOwner


_ADMIN_STATUSES = frozenset({
    ChatMemberStatus.ADMINISTRATOR,
    ChatMemberStatus.CREATOR,
})


def _has_ban_rights(member: ChatMemberAdministrator | ChatMemberOwner) -> bool:
    """Creator har doim huquqli; administratorda `can_restrict_members` bo'lishi kerak."""
    if member.status == ChatMemberStatus.CREATOR:
        return True
    if isinstance(member, ChatMemberAdministrator):
        return member.can_restrict_members
    return False


async def is_group_admin(bot: Bot, chat_id: int, user_id: int) -> bool:
    """
    Foydalanuvchi guruhda admin yoki creator ekanini tekshiradi.

    Xato yoki noaniq holatda False qaytaradi (xavfsiz default).
    """
    try:
        member = await bot.get_chat_member(chat_id, user_id)
    except (TelegramBadRequest, TelegramForbiddenError):
        return False
    except Exception:
        return False

    if member.status not in _ADMIN_STATUSES:
        return False

    return _has_ban_rights(member)