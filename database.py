"""
Database connection, table definitions, and SQLAlchemy engine.

All other modules that need DB access import `database`, `engine`, and table
objects from here — never create new Database() instances elsewhere.
"""
from sqlalchemy import (
    BigInteger, Boolean, Column, Integer, String, Table,
    MetaData, UniqueConstraint, create_engine,
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

# Create all tables synchronously at import time (idempotent).
metadata.create_all(engine)
