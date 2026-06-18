# app/models/github.py
import uuid
from sqlalchemy import Column, String, Boolean, TIMESTAMP, Text, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.models.base import Base


class ProjectGithubRepo(Base):
    __tablename__ = 'project_github_repos'
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign Keys
    room_id = Column(UUID(as_uuid=True), ForeignKey('chat_rooms.id', ondelete='CASCADE'), unique=True, nullable=False)
    submitted_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    
    # Repository details
    repo_url = Column(String(500), nullable=False)
    is_private = Column(Boolean, default=False)
    encrypted_access_token = Column(Text, nullable=True)
    
    # Timestamps
    submitted_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    
    # Table constraints
    __table_args__ = (
        Index('idx_github_repo_room', 'room_id'),
    )
    
    def __repr__(self):
        return f"<ProjectGithubRepo(id={self.id}, room_id={self.room_id}, repo_url={self.repo_url})>"