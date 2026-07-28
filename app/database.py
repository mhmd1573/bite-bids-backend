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
        DATABASE_URL = os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://postgres:root@localhost:5432/bitebids"
        )
        DEBUG = os.getenv("DEBUG", "False").lower() == "true"

    settings = FallbackSettings()


# Normalize PostgreSQL URL for asyncpg
database_url = settings.DATABASE_URL

if database_url.startswith("postgresql://"):
    database_url = database_url.replace(
        "postgresql://",
        "postgresql+asyncpg://",
        1
    )

elif database_url.startswith("postgres://"):
    database_url = database_url.replace(
        "postgres://",
        "postgresql+asyncpg://",
        1
    )


# Create async engine
engine = create_async_engine(
    database_url,
    echo=getattr(settings, "DEBUG", False),
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    connect_args={
        "ssl": True
    }
)


# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)


# Base class for models
from app.models.base import Base


# Dependency for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session dependency"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# Helper function to get database session
async def get_db_session() -> AsyncSession:
    """Get a database session directly for background tasks, etc."""
    return AsyncSessionLocal()