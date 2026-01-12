import os
from dotenv import load_dotenv

# .env faylni o'qiymiz (agar mavjud bo'lsa)
load_dotenv()

# Render-dagi Environment Variable-ni oladi
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./telegram_app.db")

# SQLAlchemy uchun postgres:// ni postgresql:// ga aylantirish (muhim!)
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)