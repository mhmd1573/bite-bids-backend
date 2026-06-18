# app/models/contact.py
import uuid
from sqlalchemy import Column, String, Boolean, TIMESTAMP, Text, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.models.base import Base


class ContactFormRecord(Base):
    __tablename__ = 'contact_form_submissions'
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Form fields
    name = Column(String(100), nullable=False)
    email = Column(String(255), nullable=False)
    subject = Column(String(200), nullable=False)
    category = Column(String(50), nullable=False)  # general, technical, billing, partnership, feedback
    message = Column(Text, nullable=False)
    
    # Status
    responded = Column(Boolean, default=False)
    responded_at = Column(TIMESTAMP, nullable=True)
    
    # Timestamp
    submitted_at = Column(TIMESTAMP, server_default=func.now())
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("category IN ('general', 'technical', 'billing', 'partnership', 'feedback')"),
    )
    
    def __repr__(self):
        return f"<ContactFormRecord(id={self.id}, email={self.email}, category={self.category})>"