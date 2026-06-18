# app/models/notification.py
import uuid
from sqlalchemy import Column, String, Boolean, TIMESTAMP, Text, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func

from app.models.base import Base


class Notification(Base):
    __tablename__ = 'notifications'
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign Key
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Notification details
    type = Column(String(50), nullable=False)  # bid_received, payment_required, project_approved, etc.
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    link = Column(String(500), nullable=True)
    details = Column(JSONB, nullable=True)  # Additional metadata
    
    # Status
    read = Column(Boolean, default=False)
    read_at = Column(TIMESTAMP, nullable=True)
    
    # Timestamp
    created_at = Column(TIMESTAMP, server_default=func.now(), index=True)
    
    # Table constraints
    __table_args__ = (
        Index('idx_notifications_read', 'read', postgresql_where=(read == False)),
    )
    
    def __repr__(self):
        return f"<Notification(id={self.id}, user_id={self.user_id}, type={self.type}, read={self.read})>"