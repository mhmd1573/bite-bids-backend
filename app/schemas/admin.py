# app/schemas/admin.py
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class AdminLogin(BaseModel):
    """Used when admin logs in"""
    email: str
    password: str

class AdminChatFilter(BaseModel):
    """Filter options for admin chat viewing"""
    search: Optional[str] = None
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    has_unread: Optional[bool] = None

class BroadcastMessage(BaseModel):
    """Used when admin sends broadcast"""
    title: str
    message: str
    audience: str = "all"  # all, developers, investors

class BanRequest(BaseModel):
    """Used when admin bans a user"""
    reason: Optional[str] = None