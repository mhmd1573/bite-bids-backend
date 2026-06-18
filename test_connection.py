# test_connection.py
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

async def test_connection():
    DATABASE_URL = "postgresql+asyncpg://postgres:root@localhost:5432/bitebids"
    engine = create_async_engine(DATABASE_URL)
    
    async with engine.begin() as conn:
        result = await conn.execute(text("SELECT version()"))
        print("✅ Connected to PostgreSQL!")
        print(result.scalar())
    
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(test_connection())

