"""
ChatSphere moderatsiya botini ishga tushirish.

  set BOT_TOKEN=<@BotFather token>
  python run_bot.py

Bot guruhda admin bo'lishi va «Foydalanuvchilarni cheklash» huquqiga ega bo'lishi kerak.
"""
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

from bot.dispatcher_setup import build_dispatcher, stop_bot_polling
from database import database
from services.bot_service import close_bot, get_bot

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    token = os.getenv("BOT_TOKEN", "").strip()
    if not token:
        logger.error(
            "BOT_TOKEN topilmadi. .env fayliga qo'shing:\n"
            "  BOT_TOKEN=123456789:ABCdefGHI..."
        )
        sys.exit(1)

    await database.connect()
    bot = await get_bot()
    dp = build_dispatcher()

    logger.info("ChatSphere bot polling boshlandi…")
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await stop_bot_polling()
        await database.disconnect()


if __name__ == "__main__":
    asyncio.run(main())