"""
Pydantic request / response models for the moderation bot.
"""
import re
from datetime import datetime
from typing import Annotated, Optional

from pydantic import BaseModel, BeforeValidator, field_serializer, field_validator


# ── Shared phone type ─────────────────────────────────────────────────────────

def _normalize_phone(v: str) -> str:
    digits = re.sub(r"\D", "", v)
    if not (10 <= len(digits) <= 15):
        raise ValueError("Noto'g'ri telefon raqami")
    return "+" + digits

Phone = Annotated[str, BeforeValidator(_normalize_phone)]


# ── Auth request models ───────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    api_id: int
    api_hash: str
    phone: Phone

    @field_validator("api_id")
    @classmethod
    def api_id_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("API ID musbat bo'lishi kerak")
        return v

    @field_validator("api_hash")
    @classmethod
    def api_hash_not_empty(cls, v: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{32}", v):
            raise ValueError("API Hash noto'g'ri formatda (32 ta kichik harfli hex belgi bo'lishi kerak)")
        return v


class VerifyRequest(BaseModel):
    phone: Phone
    code: str
    phone_code_hash: str
    api_id: int
    api_hash: str

    @field_validator("code")
    @classmethod
    def code_valid(cls, v: str) -> str:
        if not v or len(v) < 3:
            raise ValueError("Kod juda qisqa")
        return v


class TwoFARequest(BaseModel):
    phone: Phone
    password: str

    @field_validator("password")
    @classmethod
    def password_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("Parol bo'sh bo'lmasligi kerak")
        return v


# ── Telegram utility models ───────────────────────────────────────────────────

class PhoneRequest(BaseModel):
    phone: Phone


# ── Moderation / monitoring models ───────────────────────────────────────────

class MonitorRequest(BaseModel):
    phone: Phone
    chat_id: int

    @field_validator("chat_id", mode="before")
    @classmethod
    def chat_id_coerce(cls, v) -> int:
        try:
            v = int(v)
        except (TypeError, ValueError):
            raise ValueError("chat_id butun son bo'lishi kerak")
        if v == 0:
            raise ValueError("chat_id 0 bo'lishi mumkin emas")
        return v


# ── Response models ───────────────────────────────────────────────────────────

class ToggleRestrictionRequest(BaseModel):
    is_enabled: bool


class BannedUserRecord(BaseModel):
    user_id: int
    first_name: Optional[str] = None
    username: Optional[str] = None
    banned_at: Optional[str] = None


class BannedListResponse(BaseModel):
    """Frontend blacklist-panel kutadigan struktura."""
    chat_id: int
    restriction_mode: bool
    banned_users: list[BannedUserRecord]
    count: int


class ViolationRecord(BaseModel):
    user_id: int
    first_name: Optional[str] = None
    username: Optional[str] = None
    warn_count: int
    is_muted: bool
    is_banned: bool
    muted_until: Optional[str] = None


class ViolationsListResponse(BaseModel):
    chat_id: int
    violations: list[ViolationRecord]
    count: int


class MutedListResponse(BaseModel):
    chat_id: int
    muted_users: list[ViolationRecord]
    count: int
