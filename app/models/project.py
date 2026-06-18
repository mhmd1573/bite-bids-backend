# app/models/project.py
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Boolean, DECIMAL, TIMESTAMP, Text, ARRAY, ForeignKey, CheckConstraint, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.models.base import Base


class Project(Base):
    __tablename__ = 'projects'
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Basic info
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    tech_stack = Column(ARRAY(Text), nullable=False)
    requirements = Column(Text, nullable=False)
    budget = Column(DECIMAL(12,2), nullable=False, index=True)
    deadline = Column(TIMESTAMP, index=True)
    
    # Owner (developer who created the project)
    developer_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    developer = relationship("User", back_populates="projects", foreign_keys=[developer_id])
    
    # Status
    status = Column(String(50), default='open', index=True)
    featured = Column(Boolean, default=False)
    priority = Column(String(20), default='medium')
    
    # Assignment
    assigned_to = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'))
    
    # Bids
    bids_count = Column(Integer, default=0)
    highest_bid = Column(DECIMAL(12,2))
    lowest_bid = Column(DECIMAL(12,2))
    
    # Location
    location = Column(String(255))
    remote = Column(Boolean, default=True)
    
    # Categories
    category = Column(String(50), nullable=False, index=True)
    tags = Column(ARRAY(Text))
    
    # Ratings
    rating = Column(DECIMAL(3,2), default=0)
    reviews_count = Column(Integer, default=0)
    
    # Progress
    progress = Column(Integer, default=0)
    
    # Images
    images = Column(ARRAY(String), default=[])
    
    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now(), index=True)
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    completed_at = Column(TIMESTAMP)
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("status IN ('open', 'in_progress', 'completed', 'closed', 'cancelled', 'fixed_price', 'disputed')"),
        CheckConstraint("priority IN ('low', 'medium', 'high')"),
        CheckConstraint("progress >= 0 AND progress <= 100"),
    )
    
    def __repr__(self):
        return f"<Project(id={self.id}, title={self.title}, status={self.status})>"