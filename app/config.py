# app/config.py
import os
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment variables"""
    
    # Application
    APP_NAME: str = "BiteBids API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:root@localhost:5432/bitebids"
    )
    
    # Security
    JWT_SECRET: str = os.getenv("JWT_SECRET")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_DAYS: int = int(os.getenv("JWT_EXPIRATION_DAYS", "7"))
    ENCRYPTION_KEY: str = os.getenv("ENCRYPTION_KEY")
    
    # URLs
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
    BASE_URL: str = os.getenv("BASE_URL", "http://localhost:8001")
    
    # CORS
    CORS_ORIGINS: List[str] = [
        origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]
    
    # Email (SMTP)
    SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", 587))
    SMTP_USERNAME: str = os.getenv("SMTP_USERNAME")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD")
    EMAIL_FROM: str = os.getenv("EMAIL_FROM")
    ADMIN_EMAIL: str = os.getenv("ADMIN_EMAIL", "bitebids@gmail.com")
    
    # Stripe
    STRIPE_SECRET_KEY: str = os.getenv("STRIPE_SECRET_KEY")
    STRIPE_PUBLISHABLE_KEY: str = os.getenv("STRIPE_PUBLISHABLE_KEY")
    STRIPE_WEBHOOK_SECRET: str = os.getenv("STRIPE_WEBHOOK_SECRET")
    STRIPE_CONNECT_WEBHOOK_SECRET: str = os.getenv("STRIPE_CONNECT_WEBHOOK_SECRET")
    
    # Cloudflare R2
    R2_ACCOUNT_ID: str = os.getenv("R2_ACCOUNT_ID")
    R2_ACCESS_KEY_ID: str = os.getenv("R2_ACCESS_KEY_ID")
    R2_SECRET_ACCESS_KEY: str = os.getenv("R2_SECRET_ACCESS_KEY")
    R2_BUCKET_NAME: str = os.getenv("R2_BUCKET_NAME", "bitebids-projects")
    
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    
    # GitHub
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN")
    GITHUB_CLIENT_ID: str = os.getenv("GITHUB_CLIENT_ID")
    GITHUB_CLIENT_SECRET: str = os.getenv("GITHUB_CLIENT_SECRET")
    
    # Google OAuth
    GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET: str = os.getenv("GOOGLE_CLIENT_SECRET")
    
    # 2Checkout
    CHECKOUT_MERCHANT_CODE: str = os.getenv("CHECKOUT_MERCHANT_CODE")
    CHECKOUT_SECRET_KEY: str = os.getenv("CHECKOUT_SECRET_KEY")
    CHECKOUT_ENVIRONMENT: str = os.getenv("CHECKOUT_ENVIRONMENT", "sandbox")
    
    # Platform Fees
    PLATFORM_FEE_PERCENTAGE: int = int(os.getenv("PLATFORM_FEE_PERCENTAGE", "6"))
    PLATFORM_FIXED_FEE: int = int(os.getenv("PLATFORM_FIXED_FEE", "30"))
    PROJECT_POSTING_FEE: float = float(os.getenv("PROJECT_POSTING_FEE", "0.99"))
    
    # File Upload
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", "5242880"))  # 5MB default
    ALLOWED_IMAGE_EXTENSIONS: List[str] = ['.png', '.jpg', '.jpeg']
    
    # ImgBB
    IMGBB_API_KEY: str = os.getenv("IMGBB_API_KEY")
    IMGBB_UPLOAD_URL: str = "https://api.imgbb.com/1/upload"
    
    # Upload directories
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads/chat_files")
    PROJECT_IMAGES_DIR: str = os.getenv("PROJECT_IMAGES_DIR", "uploads/project_images")
    
    def __init__(self):
        # Validate required settings
        if not self.JWT_SECRET:
            raise ValueError("JWT_SECRET is required but not set in environment variables")
        if not self.DATABASE_URL:
            raise ValueError("DATABASE_URL is required but not set in environment variables")


# Create a single instance of settings
settings = Settings()