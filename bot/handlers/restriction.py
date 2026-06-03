"""
Guruhga kirishda qora ro'yxat tekshiruvi (restriction_mode).

restriction_mode == True va foydalanuvchi banned_users da bo'lsa:
  - oddiy qo'shilish → ban_chat_member
  - join request → decline_chat_join_request
"""
import logging

from aiogram import Bot, F, Router
from aiogram.enums import ChatMemberStatus, ChatType
from aiogram.filters import ChatMemberUpdatedFilter, IS_MEMBER
from aiogram.types import ChatJoinRequest, ChatMemberUpdated

from services.restriction_guard import enforce_restriction, should_block_rejoin

logger = logging.getLogger(__name__)

router = Router(name="restriction")

# Foydalanuvchi guruhga qo'shildi (link, qo'shish, tasdiqlangan so'rov)
@router.chat_member(
    ChatMemberUpdatedFilter(IS_MEMBER),
    F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}),
)
async def on_member_joined(event: ChatMemberUpdated, bot: Bot) -> None:
    chat_id = event.chat.id
    user = event.new_chat_member.user
    if user is None or user.is_bot:
        return

    old_status = event.old_chat_member.status
    # Aiogram 3: 'kicked' — guruhdan chiqarilgan / ban (alohida 'banned' status yo'q)
    if old_status not in (
        ChatMemberStatus.LEFT,
        ChatMemberStatus.KICKED,
        ChatMemberStatus.RESTRICTED,
    ):
        return

    user_id = user.id
    if not await should_block_rejoin(chat_id, user_id):
        return

    blocked = await enforce_restriction(
        bot, chat_id, user_id, context="chat_member_join"
    )
    if blocked:
        try:
            await bot.send_message(
                chat_id,
                f"🚫 <b>{user.first_name or user.id}</b> qora ro'yxatda — "
                "cheklov rejimi yoqilgan, guruhga qo'shilmadi.",
            )
        except Exception:
            pass


@router.chat_join_request(F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}))
async def on_join_request(event: ChatJoinRequest, bot: Bot) -> None:
    """Guruhga kirish so'rovi — qora ro'yxatdagini rad etish."""
    if event.from_user is None or event.from_user.is_bot:
        return

    chat_id = event.chat.id
    user_id = event.from_user.id

    if not await should_block_rejoin(chat_id, user_id):
        return

    try:
        await bot.decline_chat_join_request(chat_id=chat_id, user_id=user_id)
        logger.info(
            "Join request declined (blacklist): chat=%s user=%s",
            chat_id, user_id,
        )
    except Exception:
        logger.exception(
            "decline_chat_join_request failed chat=%s user=%s",
            chat_id, user_id,
        )
        await enforce_restriction(
            bot, chat_id, user_id, context="join_request_fallback"
        )