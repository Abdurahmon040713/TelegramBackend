from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
from contextlib import asynccontextmanager
from typing import List, Optional
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from telethon import TelegramClient, errors
from telethon.sessions import StringSession
from sqlalchemy import Column, Integer, String, Table, MetaData, create_engine
from databases import Database
from transformers import pipeline
from dotenv import load_dotenv
from sqlalchemy.dialects.postgresql import insert

# -------------------------
# 1. Konfiguratsiya
# -------------------------
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL .env faylida topilmadi")

# -------------------------
# 2. Database
# -------------------------
database = Database(DATABASE_URL)
metadata = MetaData()

sessions = Table(
    "sessions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("phone", String, unique=True, index=True),
    Column("api_id", Integer),
    Column("api_hash", String),
    Column("session_string", String),
)

engine = create_engine(DATABASE_URL)
metadata.create_all(engine)

sentiment_analyzer = None

# -------------------------
# 3. Negativ so‘zlar
# -------------------------
def load_negative_words(file_path: str = "data/uz_negative_words.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]

NEGATIVE_KEYWORDS = load_negative_words()

# -------------------------
# 4. Lifespan
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.connect()
    yield
    await database.disconnect()

# -------------------------
# 5. App
# -------------------------
app = FastAPI(
    title="Telegram Sentiment Backend",
    lifespan=lifespan
)

# ✅ CORS (MUHIM)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# 6. Pydantic modellari
# -------------------------
class LoginRequest(BaseModel):
    api_id: int
    api_hash: str
    phone: str

class VerifyRequest(BaseModel):
    phone: str
    code: str
    phone_code_hash: str
    api_id: int
    api_hash: str

class AnalyzeRequest(BaseModel):
    phone: str
    chat_id: int
    limit: int = 50

class NegativeMessage(BaseModel):
    id: int
    text: str
    confidence: float
    sender_id: Optional[int] = None

class AnalyzeResponse(BaseModel):
    analyzed_count: int
    negative_count: int
    negative_messages: List[NegativeMessage]

# -------------------------
# 7. Yordamchi funksiyalar
# -------------------------
async def get_client_session(phone: str):
    query = sessions.select().where(sessions.c.phone == phone)
    user = await database.fetch_one(query)

    if not user:
        raise HTTPException(status_code=404, detail="Avval login qiling")

    client = TelegramClient(
        StringSession(user["session_string"]),
        user["api_id"],
        user["api_hash"]
    )

    await client.connect()

    if not await client.is_user_authorized():
        await client.disconnect()
        raise HTTPException(status_code=401, detail="Sessiya tugagan")

    return client

def analyze_text_sync(text: str):
    global sentiment_analyzer
    # Model faqat funksiya chaqirilganda yuklanadi
    if sentiment_analyzer is None:
        sentiment_analyzer = pipeline(
            "text-classification", 
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    return sentiment_analyzer(text[:512])[0]

# -------------------------
# 8. Endpointlar
# -------------------------
@app.get("/")
async def read_root():
    return {"message": "Backend ishlayapti!"}

@app.post("/login")
async def login(data: LoginRequest):
    client = TelegramClient(StringSession(), data.api_id, data.api_hash)
    await client.connect()

    try:
        sent = await client.send_code_request(data.phone)
        session_string = client.session.save()

        query = insert(sessions).values(
            phone=data.phone,
            api_id=data.api_id,
            api_hash=data.api_hash,
            session_string=session_string
        ).on_conflict_do_update(
            index_elements=[sessions.c.phone],
            set_={
                "api_id": data.api_id,
                "api_hash": data.api_hash,
                "session_string": session_string
            }
        )

        await database.execute(query)

        return {
            "status": "waiting_for_code",
            "phone_code_hash": sent.phone_code_hash
        }

    finally:
        await client.disconnect()

@app.post("/verify")
async def verify(data: VerifyRequest):
    query = sessions.select().where(sessions.c.phone == data.phone)
    user = await database.fetch_one(query)

    if not user:
        raise HTTPException(status_code=404, detail="Login qilinmagan")

    client = TelegramClient(
        StringSession(user["session_string"]),
        data.api_id,
        data.api_hash
    )

    await client.connect()

    try:
        await client.sign_in(
            phone=data.phone,
            code=data.code,
            phone_code_hash=data.phone_code_hash
        )

        session_string = client.session.save()
        await database.execute(
            sessions.update()
            .where(sessions.c.phone == data.phone)
            .values(session_string=session_string)
        )

        return {"status": "success"}

    finally:
        await client.disconnect()

@app.post("/chats")
async def get_chats(phone: str):
    client = await get_client_session(phone)

    try:
        chats = []
        async for dialog in client.iter_dialogs(limit=50):
            chats.append({
                "id": dialog.id,
                "title": dialog.title,
                "type": "Group" if dialog.is_group else "Channel" if dialog.is_channel else "Private"
            })
        return {"chats": chats}

    finally:
        await client.disconnect()

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(data: AnalyzeRequest):
    client = await get_client_session(data.phone)
    negative_messages = []

    try:
        entity = await client.get_entity(data.chat_id)
        messages = [m async for m in client.iter_messages(entity, limit=data.limit) if m.text]

        for msg in messages:
            result = await asyncio.to_thread(analyze_text_sync, msg.text)
            keyword_match = any(k in msg.text.lower() for k in NEGATIVE_KEYWORDS)

            if result["label"] == "NEGATIVE" or keyword_match:
                negative_messages.append({
                    "id": msg.id,
                    "text": msg.text,
                    "confidence": result["score"],
                    "sender_id": msg.sender_id
                })

        return {
            "analyzed_count": len(messages),
            "negative_count": len(negative_messages),
            "negative_messages": negative_messages
        }

    finally:
        await client.disconnect()

if __name__ == "__main__":
    import uvicorn
    # Render avtomatik beradigan PORT ni oladi, bo'lmasa 10000 ni ishlatadi
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)