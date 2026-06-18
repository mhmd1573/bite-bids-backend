# app/database.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from typing import AsyncGenerator

# Try to import settings, fallback to environment variables if config not available
try:
    from app.config import settings
except ImportError:
    import os
    class FallbackSettings:
        DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:root@localhost:5432/bitebids")
        DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    settings = FallbackSettings()

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG if hasattr(settings, 'DEBUG') else False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

# Base class for models (re-export from models.base)
from app.models.base import Base

# Dependency for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session dependency"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Helper function to get database session (for non-FastAPI contexts)
async def get_db_session() -> AsyncSession:
    """Get a database session directly (for background tasks, etc.)"""
    return AsyncSessionLocal()