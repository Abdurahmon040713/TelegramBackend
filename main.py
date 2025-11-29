from fastapi import FastAPI, Depends, HTTPException
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
# 1. Konfiguratsiya va Muhit
# -------------------------
load_dotenv()  # .env faylidan o'qish

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL .env faylida topilmadi!")

# -------------------------
# 2. Database va Model (Global State)
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

# Sinxron engine faqat jadvallarni yaratish uchun ishlatiladi
engine = create_engine(DATABASE_URL)
metadata.create_all(engine)

# Model uchun global o'zgaruvchi
sentiment_analyzer = None

# 2.1 Negativ so'zlar bazasi
# -------------------------
def load_negative_words(file_path: str = "data/uz_negative_words.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        words = [line.strip().lower() for line in f if line.strip()]
    return words

NEGATIVE_KEYWORDS = load_negative_words()

# -------------------------
# 3. Lifespan (Startup/Shutdown o'rniga)
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    print("🚀 Database ulanmoqda...")
    await database.connect()
    
    print("🧠 AI Model yuklanmoqda (bu biroz vaqt oladi)...")
    # Modelni global o'zgaruvchiga yuklaymiz
    global sentiment_analyzer
    sentiment_analyzer = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    print("✅ Tizim ishga tushdi!")
    
    yield
    
    # --- Shutdown ---
    print("🛑 Database uzilmoqda...")
    await database.disconnect()

app = FastAPI(title="Telegram Sentiment Backend", lifespan=lifespan)

# -------------------------
# 4. Pydantic Modellar
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
# 5. Yordamchi Funksiyalar
# -------------------------

async def get_client_session(phone: str):
    """
    DB dan foydalanuvchi sessiyasini olib, TelegramClient qaytaradi.
    """
    query = sessions.select().where(sessions.c.phone == phone)
    user = await database.fetch_one(query)

    if not user:
        raise HTTPException(status_code=404, detail="Foydalanuvchi topilmadi. Avval login qiling.")

    try:
        client = TelegramClient(StringSession(user["session_string"]), user["api_id"], user["api_hash"])
        await client.connect()

        if not await client.is_user_authorized():
            await client.disconnect()
            raise HTTPException(status_code=401, detail="Sessiya muddati tugagan. Qayta login qiling.")

        return client
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Telegram ulanish xatosi: {str(e)}")

def analyze_text_sync(text: str):
    """
    Modelni sinxron chaqirish uchun funksiya.
    Buni threadpool da ishlatamiz.
    """
    if not sentiment_analyzer:
        raise RuntimeError("Model hali yuklanmadi")
    # Matn juda uzun bo'lsa qirqib olamiz (model max 512 token)
    return sentiment_analyzer(text[:512])[0]

# -------------------------
# 6. Endpointlar
# -------------------------

@app.post("/login")
async def login(data: LoginRequest):
    client = TelegramClient(StringSession(), data.api_id, data.api_hash)
    await client.connect()

    try:
        if await client.is_user_authorized():
            return {"status": "authorized", "message": "Allaqachon tizimga kirilgan"}

        sent = await client.send_code_request(data.phone)
        session_string = client.session.save()

        # Upsert query (agar phone mavjud bo‘lsa, yangilaydi)
        query = insert(sessions).values(
            phone=data.phone,
            api_id=data.api_id,
            api_hash=data.api_hash,
            session_string=session_string
        )
        query = query.on_conflict_do_update(
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
            "phone_code_hash": sent.phone_code_hash,
            "message": "Tasdiqlash kodi yuborildi"
        }
    except errors.PhoneNumberInvalidError:
        raise HTTPException(status_code=400, detail="Telefon raqam noto'g'ri formatda")
    except errors.FloodWaitError as e:
        raise HTTPException(status_code=429, detail=f"Telegram blokladi. {e.seconds} soniya kuting.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Xatolik: {str(e)}")
    finally:
        await client.disconnect()


@app.post("/verify")
async def verify(data: VerifyRequest):
    # Avval DB’dan session_string ni olish
    query = sessions.select().where(sessions.c.phone == data.phone)
    user = await database.fetch_one(query)
    if not user:
        raise HTTPException(status_code=404, detail="Avval login qiling")

    client = TelegramClient(StringSession(user["session_string"]), data.api_id, data.api_hash)
    await client.connect()
    logging.info(f"Verify bosqichi: hash={data.phone_code_hash}, code={data.code}")

    try:
        await client.sign_in(
            phone=data.phone,
            code=data.code,
            phone_code_hash=data.phone_code_hash
        )
        # Yangi sessiyani saqlash
        session_string = client.session.save()
        update_query = sessions.update().where(sessions.c.phone == data.phone).values(
            session_string=session_string
        )
        await database.execute(update_query)

        return {"status": "success", "message": "Muvaffaqiyatli kirildi"}
    except Exception as e:
        logging.error(f"Verify xatolik: {type(e).__name__} - {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        await client.disconnect()



@app.post("/chats")
async def get_chats(phone: str):
    client = await get_client_session(phone)
    
    try:
        dialogs = []
        async for dialog in client.iter_dialogs(limit=50):
            chat_type = "Private"
            if dialog.is_group: chat_type = "Group"
            elif dialog.is_channel: chat_type = "Channel"
            
            dialogs.append({
                "id": dialog.id,
                "title": dialog.title or "No Title",
                "type": chat_type
            })
        return {"chats": dialogs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await client.disconnect()

@app.post("/analyze", response_model=AnalyzeResponse)
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(data: AnalyzeRequest):
    client = await get_client_session(data.phone)
    negative_messages = []
    analyzed_count = 0
    
    try:
        entity = await client.get_entity(data.chat_id)
        messages_to_process = []
        
        async for message in client.iter_messages(entity, limit=data.limit):
            if message.text and len(message.text.strip()) > 0:
                messages_to_process.append(message)
        
        analyzed_count = len(messages_to_process)
        
        for msg in messages_to_process:
            text_lower = msg.text.lower()
            
            # 1️⃣ Model orqali tahlil
            result = await asyncio.to_thread(analyze_text_sync, msg.text)
            
            # 2️⃣ So'zlar bazasi orqali tekshirish
            keyword_match = any(kw in text_lower for kw in NEGATIVE_KEYWORDS)
            
            if result["label"] == "NEGATIVE" or keyword_match:
                negative_messages.append({
                    "id": msg.id,
                    "text": msg.text,
                    "confidence": round(result["score"], 4) if not keyword_match else 1.0,
                    "sender_id": msg.sender_id
                })
                    
        return {
            "analyzed_count": analyzed_count,
            "negative_count": len(negative_messages),
            "negative_messages": negative_messages
        }
        
    except ValueError:
         raise HTTPException(status_code=400, detail="Chat topilmadi (ID noto'g'ri)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await client.disconnect()
