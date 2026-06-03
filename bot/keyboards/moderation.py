"""Moderatsiya inline klaviaturalari."""
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

UNBAN_CALLBACK_PREFIX = "unban_"


def build_unban_keyboard(user_id: int, display_name: str) -> InlineKeyboardMarkup:
    """Bloklangan foydalanuvchini qora ro'yxatdan chiqarish tugmasi."""
    label = display_name.strip() or f"ID {user_id}"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"🔓 {label}ni blokdan ochish",
                    callback_data=f"{UNBAN_CALLBACK_PREFIX}{user_id}",
                )
            ]
        ]
    )


def parse_unban_user_id(callback_data: str) -> int | None:
    """`unban_1234567` dan user_id ni ajratib oladi."""
    if not callback_data.startswith(UNBAN_CALLBACK_PREFIX):
        return None
    raw = callback_data[len(UNBAN_CALLBACK_PREFIX):]
    if not raw.isdigit():
        return None
    return int(raw)