"""
Real-time moderation: message handler + punishment escalation.

Punishment escalation per (phone, chat_id, user_id):
  warn_count < WARN_LIMIT          → delete message + send warning notice
  WARN_LIMIT <= warn_count < BAN   → delete message + mute for MUTE_DURATION_SECONDS
  warn_count >= BAN_AFTER_WARNS    → delete message + permanent ban

attach_monitor() wires a Telethon NewMessage handler to a live client.
detach_monitor() removes it.  Both are safe to call from async endpoints.
"""
import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from telethon import TelegramClient, events
from telethon.errors import (
    ChatAdminRequiredError,
    ChatWriteForbiddenError,
    UserAdminInvalidError,
    UserIdInvalidError,
)
from telethon.tl.functions.channels import EditBannedRequest
from telethon.tl.types import ChatBannedRights

import ai_inference
from config import (
    AI_NEGATIVE_SCORE_THRESHOLD, AI_UZ_SCORE_THRESHOLD,
    CONTEXT_WEIGHT_TOXIC, CONTEXT_WEIGHT_SUSPICIOUS,
    BAN_AFTER_WARNS, MUTE_DURATION_SECONDS, WARN_LIMIT,
)
from context_scorer import score_text as _context_score
from database import database, violations_table
from services.ban_repository import add_banned_user
from services.violations_repository import get_violation as _repo_get_violation
from keywords import is_toxic_by_keywords

logger = logging.getLogger(__name__)

# (phone, chat_id) → registered callback function
_monitor_registry: dict[tuple[str, int], Any] = {}
_registry_lock = asyncio.Lock()


# ── DB helpers ────────────────────────────────────────────────────────────────

async def _get_violation(phone: str, chat_id: int, user_id: int):
    """violations_repository.get_violation ga delegatsiya."""
    return await _repo_get_violation(phone, chat_id, user_id)


async def _increment_violation(phone: str, chat_id: int, user_id: int) -> int:
    """Atomically bump warn_count.  Returns the new value."""
    row = await _get_violation(phone, chat_id, user_id)
    if row is None:
        await database.execute(
            violations_table.insert().values(
                phone=phone, chat_id=chat_id, user_id=user_id,
                warn_count=1, is_muted=False, is_banned=False,
            )
        )
        return 1
    new_count = row["warn_count"] + 1
    await database.execute(
        violations_table.update()
        .where(violations_table.c.phone == phone)
        .where(violations_table.c.chat_id == chat_id)
        .where(violations_table.c.user_id == user_id)
        .values(warn_count=new_count)
    )
    return new_count


async def _set_violation_flags(
    phone: str, chat_id: int, user_id: int, **flags
) -> None:
    """Update status flags (is_muted, is_banned, muted_until) without touching warn_count."""
    await database.execute(
        violations_table.update()
        .where(violations_table.c.phone == phone)
        .where(violations_table.c.chat_id == chat_id)
        .where(violations_table.c.user_id == user_id)
        .values(**flags)
    )


# ── Language detection ────────────────────────────────────────────────────────

# Markers characteristic of Uzbek (Latin or Cyrillic script).
# Used to pick the lower AI confidence threshold for Uzbek input.
_UZ_MARKERS = re.compile(
    r"o['''ʻʼ`ʹ′]"     # o' — Uzbek vowel marker (Latin)
    r"|g['''ʻʼ`ʹ′]"    # g' — Uzbek consonant marker (Latin)
    r"|sh|ch|ng"         # common Uzbek digraphs
    r"|[ўқғҳ]",         # Uzbek-specific Cyrillic letters
    re.IGNORECASE,
)


def _is_likely_uzbek(text: str) -> bool:
    """Return True when *text* shows strong Uzbek-language markers."""
    return bool(_UZ_MARKERS.search(text))


# ── Punishment actions ────────────────────────────────────────────────────────

async def _delete_message(client: TelegramClient, chat_id: int, msg_id: int) -> None:
    try:
        await client.delete_messages(chat_id, [msg_id])
        logger.info("Deleted message %d from chat %d", msg_id, chat_id)
    except Exception as exc:
        logger.warning("Delete failed — msg=%d chat=%d: %s", msg_id, chat_id, exc)


async def _mute_user(client: TelegramClient, chat_id: int, user_id: int) -> bool:
    until = datetime.now(timezone.utc) + timedelta(seconds=MUTE_DURATION_SECONDS)
    try:
        await client(EditBannedRequest(
            channel=chat_id,
            participant=user_id,
            banned_rights=ChatBannedRights(until_date=until, send_messages=True),
        ))
        return True
    except (ChatAdminRequiredError, UserAdminInvalidError):
        logger.warning("No admin rights to mute user=%d in chat=%d", user_id, chat_id)
    except UserIdInvalidError:
        logger.warning("Invalid user_id=%d for mute in chat=%d", user_id, chat_id)
    except Exception as exc:
        logger.warning("Mute failed user=%d chat=%d: %s", user_id, chat_id, exc)
    return False


async def _ban_user(client: TelegramClient, chat_id: int, user_id: int) -> bool:
    try:
        await client(EditBannedRequest(
            channel=chat_id,
            participant=user_id,
            banned_rights=ChatBannedRights(until_date=None, view_messages=True),
        ))
        return True
    except (ChatAdminRequiredError, UserAdminInvalidError):
        logger.warning("No admin rights to ban user=%d in chat=%d", user_id, chat_id)
    except UserIdInvalidError:
        logger.warning("Invalid user_id=%d for ban in chat=%d", user_id, chat_id)
    except Exception as exc:
        logger.warning("Ban failed user=%d chat=%d: %s", user_id, chat_id, exc)
    return False


async def _send_notice(
    client: TelegramClient,
    chat_id: int,
    user_id: int,
    action: str,
    warn_count: int,
) -> None:
    mention = f"[{user_id}](tg://user?id={user_id})"
    mute_mins = MUTE_DURATION_SECONDS // 60

    texts = {
        "warn": (
            f"⚠️ {mention} ogohlantirish oldi "
            f"({warn_count}/{WARN_LIMIT - 1}). "
            "Iltimos, guruh qoidalariga rioya qiling."
        ),
        "mute": (
            f"🔇 {mention} qoidani buzgani uchun "
            f"{mute_mins} daqiqa sukutga olindi."
        ),
        "ban": (
            f"🚫 {mention} qoidalarni qayta-qayta buzgani uchun "
            "guruhdan doimiy ravishda chiqarildi."
        ),
    }
    try:
        await client.send_message(chat_id, texts[action], parse_mode="markdown")
    except (ChatWriteForbiddenError, ChatAdminRequiredError):
        logger.warning("Cannot send notice to chat=%d (no write permission)", chat_id)
    except Exception as exc:
        logger.warning("Notice send failed for chat=%d: %s", chat_id, exc)


# ── Core message handler ──────────────────────────────────────────────────────

def _build_handler(client: TelegramClient, phone: str, chat_id: int):
    """
    Returns a Telethon event callback for (phone, chat_id).

    The handler is entirely non-blocking:
      - Regex keyword check is synchronous but O(n_keywords) — fast.
      - AI inference is dispatched via asyncio.to_thread so it never
        blocks the event loop.
      - All DB and Telegram API calls are awaited coroutines.
    """
    async def on_new_message(event: events.NewMessage.Event) -> None:
        try:
            msg = event.message
            logger.debug(
                "Handler triggered: chat=%d msg_id=%s sender=%s has_text=%s",
                chat_id,
                getattr(msg, "id", None),
                getattr(msg, "sender_id", None),
                bool(getattr(msg, "text", None)),
            )

            if not msg or not msg.text or not msg.sender_id:
                return

            # Never moderate our own account
            me = await client.get_me()
            if msg.sender_id == me.id:
                return

            # ════════════════════════════════════════════════════════════════
            # HYBRID 3-LAYER PIPELINE
            # ════════════════════════════════════════════════════════════════

            # ── Layer 1: Regex keyword scan (fastest — O(n_keywords)) ─────────
            is_toxic       = is_toxic_by_keywords(msg.text)
            detect_reason  = "keyword_match" if is_toxic else None

            logger.debug(
                "L1-Keyword: chat=%d msg_id=%d toxic=%s preview='%.60s'",
                chat_id, msg.id, is_toxic, msg.text,
            )

            if not is_toxic:
                # ── Layer 2: Contextual weight scoring ────────────────────────
                # Pure Python regex — no I/O, runs in <1 ms.
                ctx = _context_score(msg.text)

                logger.debug(
                    "L2-Context: chat=%d msg_id=%d score=%.2f signals=%s",
                    chat_id, msg.id, ctx.score, ctx.signals,
                )

                if ctx.score >= CONTEXT_WEIGHT_TOXIC:
                    # Layer 2 is confident — skip the expensive AI call.
                    is_toxic      = True
                    detect_reason = "context_weight"
                    logger.info(
                        "L2-Context TOXIC: chat=%d msg_id=%d score=%.2f "
                        "signals=%s preview='%.60s'",
                        chat_id, msg.id, ctx.score, ctx.signals, msg.text,
                    )

                elif ai_inference.is_available():
                    # ── Layer 3: AI verification ──────────────────────────────
                    # If Layer 2 raised a suspicious signal, tighten the AI
                    # threshold by 15 % so the model catches borderline cases.
                    is_uz   = _is_likely_uzbek(msg.text)
                    base_t  = AI_UZ_SCORE_THRESHOLD if is_uz else AI_NEGATIVE_SCORE_THRESHOLD
                    threshold = base_t * 0.85 if ctx.score >= CONTEXT_WEIGHT_SUSPICIOUS else base_t

                    try:
                        ai_results = await asyncio.to_thread(
                            ai_inference.analyze_batch, [msg.text]
                        )
                        r      = ai_results[0]
                        score  = float(r.get("score", 0.0))
                        label  = r.get("label", "").lower()

                        if label == "negative" and score > threshold:
                            is_toxic      = True
                            detect_reason = "ai_sentiment"
                        elif is_uz and ctx.score >= 0.50:
                            # AI O'zbek tilida ishonchsiz — Layer 2 signali
                            # yetarlicha kuchli (>= 0.50) bo'lsa, AI ga emas
                            # Layer 2 ga ishonish.
                            is_toxic      = True
                            detect_reason = "context_weight"
                            logger.info(
                                "L2-override (AI unreliable for Uzbek): "
                                "chat=%d msg_id=%d ctx_score=%.2f ai_label=%s",
                                chat_id, msg.id, ctx.score, label,
                            )

                        logger.info(
                            "L3-AI: chat=%d msg_id=%d | label=%-8s score=%.3f "
                            "threshold=%.2f toxic=%-5s lang=%s ctx_sus=%s",
                            chat_id, msg.id,
                            label, score, threshold, is_toxic,
                            "uz" if is_uz else "other",
                            ctx.score >= CONTEXT_WEIGHT_SUSPICIOUS,
                        )
                    except Exception:
                        logger.exception(
                            "L3-AI inference error — falling back to Layer 2 result "
                            "(chat=%d msg_id=%d)", chat_id, msg.id,
                        )

            if not is_toxic:
                return

            logger.info(
                "Toxic message detected [%s]: chat=%d msg_id=%d sender=%d preview='%.80s'",
                detect_reason, chat_id, msg.id, msg.sender_id, msg.text,
            )

            # ── Delete the offending message ───────────────────────────────────
            await _delete_message(client, chat_id, msg.id)

            # ── Step 4: increment counter and apply escalating punishment ──────
            warn_count = await _increment_violation(phone, chat_id, msg.sender_id)
            logger.info(
                "Violation recorded: chat=%d user=%d warn_count=%d",
                chat_id, msg.sender_id, warn_count,
            )

            if warn_count >= BAN_AFTER_WARNS:
                ok = await _ban_user(client, chat_id, msg.sender_id)
                if ok:
                    await _set_violation_flags(
                        phone, chat_id, msg.sender_id, is_banned=True
                    )
                    try:
                        await add_banned_user(chat_id, msg.sender_id)
                    except Exception:
                        logger.exception(
                            "banned_users yozuvi qo'shilmadi chat=%d user=%d",
                            chat_id, msg.sender_id,
                        )
                    await _send_notice(client, chat_id, msg.sender_id, "ban", warn_count)

            elif warn_count >= WARN_LIMIT:
                muted_until = (
                    datetime.now(timezone.utc) + timedelta(seconds=MUTE_DURATION_SECONDS)
                ).isoformat()
                ok = await _mute_user(client, chat_id, msg.sender_id)
                if ok:
                    await _set_violation_flags(
                        phone, chat_id, msg.sender_id,
                        is_muted=True, muted_until=muted_until,
                    )
                    await _send_notice(client, chat_id, msg.sender_id, "mute", warn_count)

            else:
                await _send_notice(client, chat_id, msg.sender_id, "warn", warn_count)

        except Exception:
            logger.exception(
                "Unhandled error in moderation handler (chat=%d)", chat_id
            )

    return on_new_message


# ── Public API ────────────────────────────────────────────────────────────────

async def attach_monitor(
    client: TelegramClient,
    phone: str,
    chat_id: int,
    entity: Optional[Any] = None,
) -> bool:
    """Register a NewMessage handler for chat_id.  Returns False if already active.

    Pass the pre-resolved ``entity`` (from ``client.get_input_entity()``) when
    available — this ensures Telethon's event filter matches incoming updates
    reliably for supergroups/channels whose peer IDs need resolution.
    """
    key = (phone, chat_id)
    async with _registry_lock:
        if key in _monitor_registry:
            logger.debug("Monitor already active: phone=...%s chat=%d", phone[-4:], chat_id)
            return False

        callback = _build_handler(client, phone, chat_id)

        # Use the pre-resolved InputPeer when provided; fall back to raw integer.
        # The resolved entity avoids a lazy-resolution step when the first message
        # arrives, which can silently fail for supergroups not yet in the session cache.
        filter_target = entity if entity is not None else chat_id
        client.add_event_handler(callback, events.NewMessage(chats=[filter_target]))
        _monitor_registry[key] = callback

    # Pin this client so the LRU pool never evicts it while monitoring is active.
    from services.telegram_service import _pinned_clients
    _pinned_clients.add(phone)

    handlers_count = len(client.list_event_handlers())
    logger.info(
        "Monitor attached: phone=...%s  chat=%d  entity_type=%s  total_handlers=%d",
        phone[-4:], chat_id, type(filter_target).__name__, handlers_count,
    )
    return True


async def detach_monitor(
    client: TelegramClient, phone: str, chat_id: int
) -> bool:
    """Unregister the handler for chat_id.  Returns False if not active."""
    key = (phone, chat_id)
    async with _registry_lock:
        callback = _monitor_registry.pop(key, None)
        still_monitoring = any(p == phone for p, _ in _monitor_registry)

    if callback is None:
        logger.debug("Monitor not found: phone=...%s chat=%d", phone[-4:], chat_id)
        return False

    client.remove_event_handler(callback)

    # Un-pin only when this account has no remaining monitors.
    if not still_monitoring:
        from services.telegram_service import _pinned_clients
        _pinned_clients.discard(phone)

    handlers_count = len(client.list_event_handlers())
    logger.info(
        "Monitor detached: phone=...%s  chat=%d  total_handlers=%d",
        phone[-4:], chat_id, handlers_count,
    )
    return True


async def get_active_monitors(phone: str) -> list[int]:
    """Return the list of chat_ids currently being monitored for phone."""
    async with _registry_lock:
        return [cid for (p, cid) in _monitor_registry if p == phone]


async def detach_all_monitors(client: TelegramClient, phone: str) -> None:
    """Remove every handler registered for phone.  Called on shutdown / logout."""
    async with _registry_lock:
        keys = [(p, cid) for (p, cid) in list(_monitor_registry) if p == phone]
        for key in keys:
            callback = _monitor_registry.pop(key)
            try:
                client.remove_event_handler(callback)
            except Exception:
                pass

    from services.telegram_service import _pinned_clients
    _pinned_clients.discard(phone)
    logger.info("All monitors detached for phone=...%s", phone[-4:])
