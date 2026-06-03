"""
Aiogram Dispatcher va polling — FastAPI lifespan yoki run_bot.py da ishlatiladi.
"""
import asyncio
import logging
from typing import Optional

from aiogram import Dispatcher

from bot.handlers import moderation_router
from bot.handlers.restriction import router as restriction_router
from services.bot_service import close_bot, get_bot

logger = logging.getLogger(__name__)

_dispatcher: Optional[Dispatcher] = None
_polling_task: Optional[asyncio.Task] = None


def build_dispatcher() -> Dispatcher:
    global _dispatcher
    if _dispatcher is None:
        dp = Dispatcher()
        dp.include_router(moderation_router)
        dp.include_router(restriction_router)
        _dispatcher = dp
    return _dispatcher


async def start_bot_polling() -> None:
    """Background task: Telegram yangilanishlarini tinglash."""
    global _polling_task
    if _polling_task is not None and not _polling_task.done():
        logger.debug("Bot polling allaqachon ishlayapti")
        return

    bot = await get_bot()
    dp = build_dispatcher()

    async def _run() -> None:
        try:
            logger.info("Aiogram polling boshlandi")
            await dp.start_polling(
                bot,
                allowed_updates=dp.resolve_used_update_types(),
            )
        except asyncio.CancelledError:
            logger.info("Aiogram polling to'xtatildi")
            raise
        except Exception:
            logger.exception("Aiogram polling xatosi")
            raise

    _polling_task = asyncio.create_task(_run(), name="aiogram-polling")


async def stop_bot_polling() -> None:
    global _polling_task, _dispatcher
    if _dispatcher is not None:
        try:
            await _dispatcher.stop_polling()
        except Exception:
            pass
    if _polling_task is not None:
        _polling_task.cancel()
        try:
            await _polling_task
        except asyncio.CancelledError:
            pass
        _polling_task = None
    await close_bot()
    _dispatcher = None