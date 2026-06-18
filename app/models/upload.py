# app/models/upload.py
import uuid
from sqlalchemy import Column, String, Integer, TIMESTAMP, ForeignKey, CheckConstraint, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func

from app.models.base import Base


class ProjectUpload(Base):
    __tablename__ = 'project_uploads'
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign Keys
    room_id = Column(UUID(as_uuid=True), ForeignKey('chat_rooms.id', ondelete='CASCADE'), unique=True, nullable=False)
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    
    # Upload info
    file_key = Column(String(500), nullable=False)  # R2 object key
    file_name = Column(String(255), nullable=False)  # Original filename
    file_size = Column(Integer, nullable=False)  # Size in bytes
    file_tree = Column(JSONB, nullable=False)  # Folder structure for preview
    
    # Status
    status = Column(String(50), default='pending')  # pending, confirmed, downloaded
    
    # Timestamps
    uploaded_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    
    # Table constraints
    __table_args__ = (
        Index('idx_project_upload_room', 'room_id'),
        CheckConstraint("status IN ('pending', 'confirmed', 'downloaded')"),
    )
    
    def __repr__(self):
        return f"<ProjectUpload(id={self.id}, room_id={self.room_id}, file_name={self.file_name}, status={self.status})>"