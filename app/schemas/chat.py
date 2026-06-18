# app/schemas/chat.py
from pydantic import BaseModel
from typing import Optional

class ChatMessageCreate(BaseModel):
    """Used when sending a chat message"""
    message: str
    message_type: Optional[str] = "text"
    file_url: Optional[str] = None
    file_name: Optional[str] = None
    file_type: Optional[str] = None
    file_size: Optional[int] = None