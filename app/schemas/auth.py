# app/schemas/auth.py
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any

class UserCreate(BaseModel):
    """Used when registering a new user"""
    email: EmailStr
    password: str
    role: str  # developer, investor, admin
    status: Optional[str] = "active"
    name: str
    company: Optional[str] = None
    address: Optional[str] = None

class UserLogin(BaseModel):
    """Used when logging in"""
    email: EmailStr
    password: str

class OAuthValidation(BaseModel):
    """Used to validate OAuth session"""
    session_id: str

class OAuthConfig(BaseModel):
    """OAuth provider configuration"""
    client_id: str
    client_secret: str

class OAuthSetupResponse(BaseModel):
    """OAuth setup response"""
    provider: str
    redirect_url: str
    configured: bool
    message: str