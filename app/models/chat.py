# app/models/chat.py
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Boolean, TIMESTAMP, Text, ForeignKey, CheckConstraint, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.models.base import Base


class ChatRoom(Base):
    __tablename__ = 'chat_rooms'
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign Keys
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id', ondelete='CASCADE'), nullable=False, index=True)
    developer_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    investor_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Status
    status = Column(String(50), default='active', index=True)
    
    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    closed_at = Column(TIMESTAMP)
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("status IN ('active', 'closed', 'archived')"),
        UniqueConstraint('project_id', 'investor_id', name='chat_rooms_project_investor_unique'),
    )
    
    def __repr__(self):
        return f"<ChatRoom(id={self.id}, project_id={self.project_id}, status={self.status})>"


class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign Keys
    room_id = Column(UUID(as_uuid=True), ForeignKey('chat_rooms.id', ondelete='CASCADE'), nullable=False, index=True)
    sender_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Message content
    message = Column(Text, nullable=False)
    message_type = Column(String(50), default='text')  # text, file, system, payment_update
    
    # File attachment (if any)
    file_url = Column(String(500))
    file_name = Column(String(255))
    file_type = Column(String(100))
    file_size = Column(Integer)
    
    # Status
    read = Column(Boolean, default=False)
    read_at = Column(TIMESTAMP)
    
    # Moderation
    flagged = Column(Boolean, default=False)
    moderation_status = Column(String(50), default='pending')  # pending, approved, rejected
    moderation_reason = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now(), index=True)
    edited_at = Column(TIMESTAMP)
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("message_type IN ('text', 'file', 'system', 'payment_update')"),
        CheckConstraint("moderation_status IN ('pending', 'approved', 'rejected')"),
    )
    
    def __repr__(self):
        return f"<ChatMessage(id={self.id}, room_id={self.room_id}, sender_id={self.sender_id})>"