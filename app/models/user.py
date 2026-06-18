# app/models/user.py
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Boolean, DECIMAL, TIMESTAMP, Text, ARRAY, CheckConstraint, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.models.base import Base


class User(Base):
    __tablename__ = 'users'
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Authentication
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=True)  # Nullable for OAuth users
    name = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, index=True)
    status = Column(String(50), default="active", index=True)
    company = Column(String(255))
    address = Column(Text)
    
    # Email verification
    email_verified = Column(Boolean, default=False)
    verification_token = Column(String(255), nullable=True)
    verification_sent_at = Column(TIMESTAMP, nullable=True)
    
    # Email change flow
    pending_email = Column(String(255), nullable=True)
    email_change_token = Column(String(255), nullable=True)
    email_change_sent_at = Column(TIMESTAMP, nullable=True)
    
    # Profile
    avatar = Column(String(500))
    bio = Column(Text)
    skills = Column(ARRAY(Text))
    verified = Column(Boolean, default=False)
    verification_date = Column(TIMESTAMP)
    profile = Column(JSONB)  # Additional profile data as JSON
    
    # Statistics
    projects_completed = Column(Integer, default=0)
    total_earnings = Column(DECIMAL(12,2), default=0)
    total_spent = Column(DECIMAL(12,2), default=0)
    avg_rating = Column(DECIMAL(3,2), default=0)
    total_reviews = Column(Integer, default=0)
    response_rate = Column(Integer, default=100)
    on_time_delivery = Column(Integer, default=100)
    reputation_score = Column(Integer, default=0)
    
    # Platform features
    posting_credits = Column(Integer, default=0)
    
    # OAuth
    oauth_provider = Column(String(50))
    oauth_id = Column(String(255))
    
    # Stripe Connect
    stripe_account_id = Column(String(255), nullable=True, unique=True)
    stripe_account_status = Column(String(50), nullable=True)
    stripe_payouts_enabled = Column(Boolean, default=False)
    stripe_onboarding_completed = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    last_login = Column(TIMESTAMP)
    
    # Relationships
    projects = relationship("Project", back_populates="developer", foreign_keys="Project.developer_id")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("role IN ('developer', 'investor', 'admin')"),
        CheckConstraint("status IN ('active', 'banned', 'suspended', 'pending', 'deleted')"),
        Index('idx_users_oauth', 'oauth_provider', 'oauth_id', unique=True, postgresql_where=(oauth_provider != None)),
        Index('idx_users_stripe_account_id', 'stripe_account_id', unique=True, postgresql_where=(stripe_account_id != None)),
    )
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, role={self.role})>"