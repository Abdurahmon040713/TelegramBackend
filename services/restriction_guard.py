"""
Kirish cheklovi: restriction_mode yoqilgan guruhlarda qora ro'yxatdagilarni chiqarish.
"""
import logging

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError

from services.ban_repository import get_restriction_mode, is_user_banned

logger = logging.getLogger(__name__)

async def should_block_rejoin(chat_id: int, user_id: int) -> bool:
    if not await get_restriction_mode(chat_id):
        return False
    return await is_user_banned(chat_id, user_id)


async def enforce_restriction(
    bot: Bot,
    chat_id: int,
    user_id: int,
    *,
    context: str = "join",
) -> bool:
    """
    Qora ro'yxatdagini guruhdan chiqaradi.

    Returns True agar bloklash amalga oshirilgan bo'lsa.
    """
    if not await should_block_rejoin(chat_id, user_id):
        return False

    try:
        await bot.ban_chat_member(chat_id=chat_id, user_id=user_id)
        logger.info(
            "Restriction enforced [%s]: chat=%s user=%s",
            context, chat_id, user_id,
        )
        return True
    except TelegramForbiddenError:
        logger.warning(
            "Restriction skip (no rights) [%s]: chat=%s user=%s",
            context, chat_id, user_id,
        )
    except TelegramBadRequest as exc:
        logger.warning(
            "Restriction ban failed [%s]: chat=%s user=%s — %s",
            context, chat_id, user_id, exc.message,
        )
    except Exception:
        logger.exception(
            "Restriction unexpected error [%s]: chat=%s user=%s",
            context, chat_id, user_id,
        )
    return False