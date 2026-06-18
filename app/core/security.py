# app/core/security.py
import jwt
import bcrypt
import uuid
import secrets
import random
from datetime import datetime, timedelta, timezone
from cryptography.fernet import Fernet
from typing import Optional, Dict, Any
from app.config import settings

# Get encryption key for token encryption
ENCRYPTION_KEY = settings.ENCRYPTION_KEY
if ENCRYPTION_KEY:
    cipher_suite = Fernet(ENCRYPTION_KEY.encode())
else:
    cipher_suite = None


def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against a hash"""
    if not password or not hashed:
        return False
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


def get_password_hash(password: str) -> str:
    """Alias for hash_password"""
    return hash_password(password)


def create_jwt_token(user_dict: Dict[str, Any]) -> str:
    """Create JWT token for user"""
    user_id = user_dict.get("_id") or user_dict.get("id")
    
    # Ensure UUIDs are converted to strings
    if isinstance(user_id, uuid.UUID):
        user_id = str(user_id)
    
    payload = {
        "user_id": user_id,
        "email": user_dict.get("email"),
        "role": user_dict.get("role"),
        "exp": datetime.now(timezone.utc) + timedelta(days=settings.JWT_EXPIRATION_DAYS)
    }
    
    # Add admin access flag if user is admin
    if user_dict.get("role") == "admin":
        payload["admin"] = True
    
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_jwt_token(token: str) -> Dict[str, Any]:
    """Decode and verify JWT token"""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError("Token expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")


def encrypt_token(token: str) -> str:
    """Encrypt a token using Fernet encryption"""
    if not cipher_suite:
        raise ValueError("ENCRYPTION_KEY not configured")
    return cipher_suite.encrypt(token.encode()).decode()


def decrypt_token(encrypted_token: str) -> str:
    """Decrypt a token using Fernet encryption"""
    if not cipher_suite:
        raise ValueError("ENCRYPTION_KEY not configured")
    return cipher_suite.decrypt(encrypted_token.encode()).decode()


def generate_verification_token() -> str:
    """Generate a secure verification token"""
    return secrets.token_urlsafe(48)


def generate_otp() -> str:
    """Generate a 6-digit OTP"""
    return str(random.randint(100000, 999999))