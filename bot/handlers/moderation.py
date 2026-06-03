"""
Guruh moderatsiyasi: /ban va inline «blokdan ochish».

Talablar:
  - /ban — reply qilingan foydalanuvchini guruhdan bloklaydi.
  - Xabar ostida unban inline tugmasi (callback_data: unban_<user_id>).
  - Tugma bosilganda faqat admin/creator unban qila oladi.
"""
import logging

from aiogram import Bot, F, Router
from aiogram.enums import ChatType
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message

from bot.keyboards.moderation import build_unban_keyboard, parse_unban_user_id
from bot.utils.admin_check import is_group_admin
from services.ban_repository import add_banned_user, remove_banned_user
from services.bot_service import unban_chat_member_db_sync

logger = logging.getLogger(__name__)

router = Router(name="moderation")


def _display_name(user) -> str:
    """Telegram ismini yoki username ni qaytaradi."""
    if user is None:
        return "Foydalanuvchi"
    parts = [user.first_name or "", user.last_name or ""]
    full = " ".join(p for p in parts if p).strip()
    if full:
        return full
    if user.username:
        return f"@{user.username}"
    return f"ID {user.id}"


# ── /ban ──────────────────────────────────────────────────────────────────────

@router.message(Command("ban"))
async def cmd_ban(message: Message, bot: Bot) -> None:
    """
    Reply qilingan xabarni yozgan foydalanuvchini guruhdan bloklaydi.

    Faqat guruh/superguruhda ishlaydi; buyruqni bergan shaxs admin bo'lishi kerak.
    """
    if message.chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        await message.reply("⚠️ Bu buyruq faqat guruhlarda ishlaydi.")
        return

    if not message.from_user:
        return

    chat_id = message.chat.id
    admin_id = message.from_user.id

    # Admin huquqini tekshirish
    if not await is_group_admin(bot, chat_id, admin_id):
        await message.reply("❌ Siz administrator emassiz — bloklash mumkin emas.")
        return

    # Reply majburiy
    if not message.reply_to_message or not message.reply_to_message.from_user:
        await message.reply(
            "ℹ️ Foydalanuvchini bloklash uchun uning xabariga "
            "<b>reply</b> qilib /ban yuboring.",
            parse_mode="HTML",
        )
        return

    target = message.reply_to_message.from_user
    target_id = target.id
    target_name = _display_name(target)

    # O'zini yoki botni bloklashga urinish
    if target_id == admin_id:
        await message.reply("⚠️ O'zingizni bloklay olmaysiz.")
        return
    if target.is_bot:
        await message.reply("⚠️ Botlarni bu buyruq orqali bloklash shart emas.")
        return

    # Maqsadli foydalanuvchi ham admin bo'lmasligi kerak
    if await is_group_admin(bot, chat_id, target_id):
        await message.reply("⚠️ Guruh administratorini bloklab bo'lmaydi.")
        return

    try:
        await bot.ban_chat_member(chat_id=chat_id, user_id=target_id)
    except TelegramForbiddenError:
        await message.reply(
            "❌ Botda «Foydalanuvchilarni bloklash» huquqi yo'q. "
            "Botni guruh admini qiling va kerakli huquqlarni bering."
        )
        return
    except TelegramBadRequest as exc:
        logger.warning("ban_chat_member failed chat=%s user=%s: %s", chat_id, target_id, exc)
        await message.reply(f"❌ Bloklash amalga oshmadi: {exc.message}")
        return
    except Exception:
        logger.exception("Unexpected ban error chat=%s user=%s", chat_id, target_id)
        await message.reply("❌ Kutilmagan xatolik yuz berdi. Keyinroq qayta urinib ko'ring.")
        return

    try:
        await add_banned_user(
            chat_id,
            target_id,
            first_name=target.first_name,
            username=target.username,
        )
    except Exception:
        logger.exception(
            "banned_users yozilmadi (Telegram ban muvaffaqiyatli) chat=%s user=%s",
            chat_id, target_id,
        )

    admin_name = _display_name(message.from_user)
    text = (
        f"🚫 <b>{target_name}</b> guruhdan bloklandi.\n"
        f"Admin: {admin_name}\n\n"
        "Qora ro'yxatdan chiqarish uchun quyidagi tugmadan foydalaning."
    )

    try:
        await message.reply(
            text,
            parse_mode="HTML",
            reply_markup=build_unban_keyboard(target_id, target_name),
        )
    except Exception:
        logger.exception("Failed to send ban confirmation chat=%s", chat_id)


# ── Unban callback ───────────────────────────────────────────────────────────

@router.callback_query(F.data.startswith("unban_"))
async def on_unban_callback(callback: CallbackQuery, bot: Bot) -> None:
    """Inline «blokdan ochish» tugmasi — faqat admin/creator uchun."""
    if not callback.message or not callback.from_user:
        await callback.answer()
        return

    if callback.message.chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        await callback.answer("Bu tugma faqat guruhlarda ishlaydi.", show_alert=True)
        return

    chat_id = callback.message.chat.id
    actor_id = callback.from_user.id

    # Admin tekshiruvi
    if not await is_group_admin(bot, chat_id, actor_id):
        await callback.answer("Siz administrator emassiz!", show_alert=True)
        return

    target_id = parse_unban_user_id(callback.data or "")
    if target_id is None:
        await callback.answer("Noto'g'ri tugma ma'lumoti.", show_alert=True)
        return

    try:
        await unban_chat_member_db_sync(chat_id, target_id)
    except PermissionError:
        await callback.answer(
            "Botda blokdan chiqarish huquqi yo'q.",
            show_alert=True,
        )
        return
    except ValueError as exc:
        await callback.answer(f"Blokdan chiqarib bo'lmadi: {exc}", show_alert=True)
        return
    except Exception:
        logger.exception("Unexpected unban error chat=%s user=%s", chat_id, target_id)
        await callback.answer("Kutilmagan xatolik.", show_alert=True)
        return

    admin_name = _display_name(callback.from_user)
    target_name = f"ID {target_id}"
    try:
        target_member = await bot.get_chat_member(chat_id, target_id)
        if target_member.user:
            target_name = _display_name(target_member.user)
    except Exception:
        pass

    success_text = (
        f"✅ <b>{admin_name}</b> tomonidan <b>{target_name}</b> blokdan ochildi.\n"
        "Endi u guruhga qayta qo'shilishi mumkin."
    )

    try:
        await callback.message.edit_text(success_text, parse_mode="HTML", reply_markup=None)
    except TelegramBadRequest:
        # Matn o'zgarmagan bo'lsa ham tugmani olib tashlash
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except Exception:
            pass
    except Exception:
        logger.exception("Failed to edit unban message chat=%s", chat_id)

    await callback.answer("Foydalanuvchi qora ro'yxatdan olib tashlandi ✅")