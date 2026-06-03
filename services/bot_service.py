"""
Aiogram Bot singleton — FastAPI va polling o'rtasida umumiy instans.

Mute/unmute/unban operatsiyalari avval bot orqali uriniladi.
Bot guruhda bo'lmasa — foydalanuvchining Telethon clienti orqali fallback qilinadi.
"""
import logging
from datetime import datetime, timedelta, timezone

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError, TelegramNetworkError
from aiogram.types import ChatPermissions

from config import BOT_TOKEN
from services.ban_repository import remove_banned_user

logger = logging.getLogger(__name__)

_bot: Bot | None = None


class BotNotConfiguredError(RuntimeError):
    """BOT_TOKEN o'rnatilmagan."""


async def get_bot() -> Bot:
    global _bot
    if not BOT_TOKEN:
        raise BotNotConfiguredError(
            "BOT_TOKEN topilmadi. .env faylida BOT_TOKEN ni o'rnating."
        )
    if _bot is None:
        _bot = Bot(
            token=BOT_TOKEN,
            default=DefaultBotProperties(parse_mode=ParseMode.HTML),
        )
    return _bot


async def close_bot() -> None:
    global _bot
    if _bot is not None:
        try:
            await _bot.session.close()
        except Exception:
            logger.exception("Bot session yopishda xatolik")
        _bot = None


async def unban_chat_member_db_sync(chat_id: int, user_id: int) -> None:
    """
    Telegram qora ro'yxatidan chiqaradi va banned_users dan o'chiradi.
    violations jadvalidagi is_banned flagini ham yangilaydi.
    """
    bot = await get_bot()
    try:
        await bot.unban_chat_member(
            chat_id=chat_id,
            user_id=user_id,
            only_if_banned=True,
        )
    except TelegramForbiddenError as exc:
        raise PermissionError(
            "Botda foydalanuvchini blokdan chiqarish huquqi yo'q"
        ) from exc
    except TelegramBadRequest as exc:
        raise ValueError(str(exc.message)) from exc
    except TelegramNetworkError as exc:
        raise ConnectionError(
            "Telegram serveriga ulanib bo'lmadi — tarmoq xatosi"
        ) from exc

    try:
        await remove_banned_user(chat_id, user_id)
    except Exception:
        logger.exception(
            "DB remove_banned_user xatosi (Telegram unban muvaffaqiyatli) "
            "chat=%s user=%s", chat_id, user_id,
        )

    # violations jadvalidagi is_banned flagini ham tozalash
    try:
        from database import database, violations_table
        await database.execute(
            violations_table.update()
            .where(violations_table.c.chat_id == chat_id)
            .where(violations_table.c.user_id == user_id)
            .values(is_banned=False)
        )
    except Exception:
        logger.exception(
            "violations.is_banned yangilanmadi chat=%s user=%s",
            chat_id, user_id,
        )

    logger.info("Unbanned chat=%s user=%s (Telegram + DB + violations)", chat_id, user_id)


# Guruhda xabar yozish va boshqa standart huquqlarni qaytarish (unmute)
_UNMUTE_PERMISSIONS = ChatPermissions(
    can_send_messages=True,
    can_send_audios=True,
    can_send_documents=True,
    can_send_photos=True,
    can_send_videos=True,
    can_send_video_notes=True,
    can_send_voice_notes=True,
    can_send_polls=True,
    can_send_other_messages=True,
    can_add_web_page_previews=True,
    can_change_info=False,
    can_invite_users=True,
    can_pin_messages=False,
)


# Jimlantirilgan foydalanuvchining ruxsatlari (faqat o'qish)
_MUTED_PERMISSIONS = ChatPermissions(
    can_send_messages=False,
    can_send_audios=False,
    can_send_documents=False,
    can_send_photos=False,
    can_send_videos=False,
    can_send_video_notes=False,
    can_send_voice_notes=False,
    can_send_polls=False,
    can_send_other_messages=False,
    can_add_web_page_previews=False,
    can_change_info=False,
    can_invite_users=False,
    can_pin_messages=False,
)


async def mute_chat_member_telegram(
    chat_id: int,
    user_id: int,
    duration_minutes: int,
    *,
    phone: str | None = None,
) -> str:
    """Telegram orqali foydalanuvchini jimlantirish.

    Avval Aiogram bot, keyin Telethon fallback.
    Returns: muted_until ISO-8601 string.
    """
    until_dt = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
    bot_failed_chat_not_found = False

    # ── 1-urinish: Aiogram bot ───────────────────────────────────────────────
    try:
        bot = await get_bot()
        await bot.restrict_chat_member(
            chat_id=chat_id,
            user_id=user_id,
            permissions=_MUTED_PERMISSIONS,
            until_date=until_dt,
        )
        logger.info(
            "Muted via bot: chat=%s user=%s duration=%d min",
            chat_id, user_id, duration_minutes,
        )
        return until_dt.isoformat()
    except BotNotConfiguredError:
        logger.info("BOT_TOKEN yo'q, Telethon fallback ishlatiladi (mute)")
        bot_failed_chat_not_found = True
    except TelegramBadRequest as exc:
        if "chat not found" in str(exc.message).lower():
            logger.warning("Bot guruhni topa olmadi (chat=%s) — Telethon fallback", chat_id)
            bot_failed_chat_not_found = True
        else:
            raise ValueError(str(exc.message)) from exc
    except TelegramForbiddenError as exc:
        raise PermissionError(
            "Botda foydalanuvchini jimlantirish huquqi yo'q (bot admin bo'lishi kerak)"
        ) from exc
    except TelegramNetworkError as exc:
        raise ConnectionError("Telegram serveriga ulanib bo'lmadi — tarmoq xatosi") from exc
    except Exception as exc:
        logger.exception("Kutilmagan mute xatosi chat=%s user=%s", chat_id, user_id)
        raise RuntimeError("Jimlantirishda kutilmagan xatolik") from exc

    # ── 2-urinish: Telethon client (fallback) ────────────────────────────────
    if bot_failed_chat_not_found and phone:
        try:
            from services.telegram_service import get_client_session
            from telethon.tl.functions.channels import EditBannedRequest
            from telethon.tl.types import ChatBannedRights

            client = await get_client_session(phone)
            await client(EditBannedRequest(
                channel=chat_id,
                participant=user_id,
                banned_rights=ChatBannedRights(
                    until_date=until_dt,
                    send_messages=True,
                ),
            ))
            logger.info(
                "Muted via Telethon fallback: chat=%s user=%s duration=%d min",
                chat_id, user_id, duration_minutes,
            )
            return until_dt.isoformat()
        except Exception as exc:
            logger.exception("Telethon fallback mute muvaffaqiyatsiz: chat=%s user=%s", chat_id, user_id)
            raise RuntimeError(
                "Jimlantirishda xatolik: bot guruhda yo'q va Telethon orqali ham amalga oshmadi"
            ) from exc

    if bot_failed_chat_not_found:
        raise ValueError("Bot bu guruhni topa olmadi. Botni guruhga qo'shing yoki admin qiling.")

    return until_dt.isoformat()


async def unmute_chat_member_telegram(
    chat_id: int,
    user_id: int,
    *,
    phone: str | None = None,
) -> None:
    """Telegram orqali jimlikni olib tashlash.

    Avval Aiogram bot orqali urinadi.  Agar bot guruhda bo'lmasa
    ("chat not found") va *phone* berilgan bo'lsa — foydalanuvchining
    Telethon clienti orqali fallback qiladi.
    """
    bot_failed_chat_not_found = False

    # ── 1-urinish: Aiogram bot ───────────────────────────────────────────────
    try:
        bot = await get_bot()
        await bot.restrict_chat_member(
            chat_id=chat_id,
            user_id=user_id,
            permissions=_UNMUTE_PERMISSIONS,
        )
        logger.info("Unmuted via bot: chat=%s user=%s", chat_id, user_id)
        return
    except BotNotConfiguredError:
        # BOT_TOKEN yo'q — to'g'ridan-to'g'ri fallback ga o'tish
        logger.info("BOT_TOKEN yo'q, Telethon fallback ishlatiladi")
        bot_failed_chat_not_found = True
    except TelegramBadRequest as exc:
        if "chat not found" in str(exc.message).lower():
            logger.warning(
                "Bot guruhni topa olmadi (chat=%s) — Telethon fallback",
                chat_id,
            )
            bot_failed_chat_not_found = True
        else:
            raise ValueError(str(exc.message)) from exc
    except TelegramForbiddenError as exc:
        raise PermissionError(
            "Serverda cheklovni olib tashlash huquqi yo'q (bot admin bo'lishi kerak)"
        ) from exc
    except TelegramNetworkError as exc:
        raise ConnectionError(
            "Telegram serveriga ulanib bo'lmadi — tarmoq xatosi"
        ) from exc
    except Exception as exc:
        logger.exception("Kutilmagan unmute xatosi chat=%s user=%s", chat_id, user_id)
        raise RuntimeError(
            "Jimlikni olib tashlashda kutilmagan xatolik"
        ) from exc

    # ── 2-urinish: Telethon client (fallback) ────────────────────────────────
    if bot_failed_chat_not_found and phone:
        try:
            from services.telegram_service import get_client_session
            from telethon.tl.functions.channels import EditBannedRequest
            from telethon.tl.types import ChatBannedRights

            client = await get_client_session(phone)
            # Barcha cheklovlarni olib tashlash (until_date=0 → cheksiz)
            await client(EditBannedRequest(
                channel=chat_id,
                participant=user_id,
                banned_rights=ChatBannedRights(
                    until_date=datetime(1970, 1, 1, tzinfo=timezone.utc),
                ),
            ))
            logger.info(
                "Unmuted via Telethon fallback: chat=%s user=%s phone=...%s",
                chat_id, user_id, phone[-4:],
            )
            return
        except Exception as exc:
            logger.exception(
                "Telethon fallback unmute ham muvaffaqiyatsiz: chat=%s user=%s",
                chat_id, user_id,
            )
            raise RuntimeError(
                "Jimlikni olib tashlashda xatolik: bot guruhda yo'q va "
                "Telethon orqali ham amalga oshmadi"
            ) from exc

    # phone berilmagan, bot ham ishlamadi
    if bot_failed_chat_not_found:
        raise ValueError(
            "Bot bu guruhni topa olmadi. Botni guruhga qo'shing yoki "
            "admin qilib tayinlang."
        )