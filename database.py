"""
Database connection, table definitions, and SQLAlchemy engine.

All other modules that need DB access import `database`, `engine`, and table
objects from here — never create new Database() instances elsewhere.
"""
from sqlalchemy import (
    BigInteger, Boolean, Column, Integer, String, Table,
    MetaData, UniqueConstraint, create_engine, text as sql_text,
)
from databases import Database

from config import DATABASE_URL

# ── Async database (queries) ──────────────────────────────────────────────────
database: Database = Database(DATABASE_URL)

# ── Sync engine (DDL only) ────────────────────────────────────────────────────
engine = create_engine(DATABASE_URL)

metadata = MetaData()

# ── Telegram sessions (login credentials) ────────────────────────────────────
sessions = Table(
    "sessions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("phone", String, unique=True, index=True),
    Column("api_id", Integer),
    Column("api_hash", String),
    Column("session_string", String),
)

# ── Per-user violation records ────────────────────────────────────────────────
# chat_id and user_id use BigInteger because Telegram supergroup/channel IDs
# exceed the 32-bit INTEGER range (e.g. -1001234567890 is ~13 digits).
violations_table = Table(
    "violations",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("phone", String, nullable=False, index=True),
    Column("chat_id", BigInteger, nullable=False, index=True),
    Column("user_id", BigInteger, nullable=False, index=True),
    Column("warn_count", Integer, default=0, nullable=False),
    Column("is_muted", Boolean, default=False, nullable=False),
    Column("is_banned", Boolean, default=False, nullable=False),
    Column("muted_until", String, nullable=True),           # ISO-8601 UTC
    UniqueConstraint(
        "phone", "chat_id", "user_id",
        name="uq_violations_phone_chat_user",
    ),
)

# ── Guruh sozlamalari (cheklov rejimi) ───────────────────────────────────────
groups_settings = Table(
    "groups_settings",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("chat_id", BigInteger, nullable=False, unique=True, index=True),
    Column("restriction_mode", Boolean, default=False, nullable=False),
    Column("updated_at", String, nullable=True),
)

# ── Qora ro'yxat (Telegram ban + veb panel) ───────────────────────────────────
banned_users = Table(
    "banned_users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("chat_id", BigInteger, nullable=False, index=True),
    Column("user_id", BigInteger, nullable=False, index=True),
    Column("first_name", String, nullable=True),
    Column("username", String, nullable=True),
    Column("banned_at", String, nullable=True),
    UniqueConstraint("chat_id", "user_id", name="uq_banned_chat_user"),
)

# Create all tables synchronously at import time (idempotent).
metadata.create_all(engine)

# SQLite: eski bazalarda jadvallar bo'lmasa yaratish
if DATABASE_URL.startswith("sqlite"):
    with engine.connect() as conn:
        conn.execute(sql_text(
            "CREATE TABLE IF NOT EXISTS groups_settings ("
            "id INTEGER PRIMARY KEY, chat_id BIGINT NOT NULL UNIQUE, "
            "restriction_mode BOOLEAN NOT NULL DEFAULT 0, updated_at VARCHAR)"
        ))
        conn.execute(sql_text(
            "CREATE TABLE IF NOT EXISTS banned_users ("
            "id INTEGER PRIMARY KEY, chat_id BIGINT NOT NULL, user_id BIGINT NOT NULL, "
            "first_name VARCHAR, username VARCHAR, banned_at VARCHAR, "
            "UNIQUE(chat_id, user_id))"
        ))
        conn.commit()
