# app/models/dispute.py
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, TIMESTAMP, Text, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.models.base import Base


class ProjectDisputeSimple(Base):
    __tablename__ = 'project_disputes_simple'
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign Keys
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id', ondelete='CASCADE'), nullable=False, index=True)
    disputed_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    investor_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False, index=True)
    resolved_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    
    # Dispute details
    reason = Column(String(100), nullable=False)
    notes = Column(Text, nullable=True)
    previous_status = Column(String(50), nullable=True)
    
    # Resolution
    resolved = Column(Boolean, default=False)
    resolution = Column(String(50), nullable=True)  # refund_investor, refund_developer, continue_project
    admin_notes = Column(Text, nullable=True)
    
    # Timestamps
    disputed_at = Column(TIMESTAMP, server_default=func.now())
    resolved_at = Column(TIMESTAMP, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<ProjectDisputeSimple(id={self.id}, project_id={self.project_id}, resolved={self.resolved})>"