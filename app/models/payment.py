# app/models/payment.py
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Boolean, DECIMAL, TIMESTAMP, Text, ForeignKey, CheckConstraint, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func

from app.models.base import Base


class CheckoutSession(Base):
    __tablename__ = 'checkout_sessions'
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Internal references
    session_id = Column(String(255), unique=True, index=True, nullable=True)
    order_reference = Column(String(100), unique=True, index=True, nullable=True)
    external_reference = Column(String(255), nullable=True)
    
    # Payment details
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id', ondelete='SET NULL'), nullable=True, index=True)
    amount = Column(DECIMAL(12,2), nullable=False)
    total_with_fees = Column(DECIMAL(12,2), nullable=False)
    fees = Column(DECIMAL(12,2), nullable=False)
    payment_method = Column(String(100), nullable=False)
    
    # Customer
    customer_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=True, index=True)
    
    # Additional data (JSON)
    payment_request = Column(JSONB, nullable=True)
    fee_calculation = Column(JSONB, nullable=True)
    extra_data = Column(JSONB, nullable=True)
    
    # Status
    status = Column(String(50), default='pending', index=True)
    
    # Payment gateway info
    payment_url = Column(String(1000), nullable=True)
    payment_method_used = Column(String(100), nullable=True)
    
    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now())
    expires_at = Column(TIMESTAMP, nullable=True)
    completed_at = Column(TIMESTAMP, nullable=True)
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'completed', 'cancelled', 'expired', 'refunded')"),
    )
    
    def __repr__(self):
        return f"<CheckoutSession(id={self.id}, order_reference={self.order_reference}, status={self.status})>"


class DeveloperPayout(Base):
    __tablename__ = 'developer_payouts'
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # References
    developer_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id', ondelete='SET NULL'), nullable=True, index=True)
    checkout_session_id = Column(UUID(as_uuid=True), ForeignKey('checkout_sessions.id', ondelete='SET NULL'), nullable=True)
    investor_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    
    # Amount details
    gross_amount = Column(DECIMAL(12,2), nullable=False)
    platform_fee = Column(DECIMAL(12,2), nullable=False)
    net_amount = Column(DECIMAL(12,2), nullable=False)
    currency = Column(String(10), default='USD')
    
    # Stripe Connect Transfer
    stripe_transfer_id = Column(String(255), nullable=True)
    stripe_transfer_status = Column(String(50), nullable=True)
    
    # Status tracking
    status = Column(String(50), default='pending', index=True)  # pending, processing, completed, failed, cancelled
    
    # Processing details
    processed_by = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    processed_at = Column(TIMESTAMP, nullable=True)
    completed_at = Column(TIMESTAMP, nullable=True)
    
    # Transaction reference
    transaction_id = Column(String(255), nullable=True)
    transaction_notes = Column(Text, nullable=True)
    
    # Failure tracking
    failure_reason = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    
    # Description
    description = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    
    # Table constraints
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')"),
        Index('idx_payout_developer_status', 'developer_id', 'status'),
    )
    
    def __repr__(self):
        return f"<DeveloperPayout(id={self.id}, developer_id={self.developer_id}, amount={self.net_amount}, status={self.status})>"