# app/models/bid.py
import uuid
from datetime import datetime
from sqlalchemy import Column, String, DECIMAL, TIMESTAMP, Text, Integer, ForeignKey, CheckConstraint, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.models.base import Base


class Bid(Base):
    __tablename__ = 'bids'
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign Keys
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id', ondelete='CASCADE'), nullable=False, index=True)
    investor_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Bid details
    amount = Column(DECIMAL(12,2), nullable=False)
    proposal = Column(Text, nullable=True)
    timeline = Column(String(100), nullable=True)
    estimated_hours = Column(Integer, nullable=True)
    
    # Status
    status = Column(String(50), default='pending', index=True)
    
    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now(), index=True)
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    accepted_at = Column(TIMESTAMP)
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'accepted', 'rejected', 'withdrawn')"),
        UniqueConstraint('project_id', 'investor_id'),
    )
    
    def __repr__(self):
        return f"<Bid(id={self.id}, project_id={self.project_id}, amount={self.amount}, status={self.status})>"