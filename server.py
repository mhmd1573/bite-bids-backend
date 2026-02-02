from fastapi import WebSocket, WebSocketDisconnect
from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, Form, Request, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import Response, HTMLResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
import os
import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import jwt
import bcrypt
import requests
import hashlib
import hmac
import shutil
from pathlib import Path
from decimal import Decimal
import logging
from enum import Enum
from authlib.integrations.starlette_client import OAuth
import httpx
import re
import urllib.parse

# SQLAlchemy imports
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, Integer, Boolean, DECIMAL, TIMESTAMP, Text, ARRAY, ForeignKey, CheckConstraint, UniqueConstraint, Index
from sqlalchemy import select, func, and_, or_, update, delete
from sqlalchemy.dialects.postgresql import UUID, JSONB, INET
from sqlalchemy.sql import func as sql_func
from sqlalchemy.orm import relationship
from sqlalchemy.orm import joinedload
from sqlalchemy import Numeric
from sqlalchemy import Column, String, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
import uuid


import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from secrets import token_urlsafe
import stripe
from collections import defaultdict
from datetime import datetime, timedelta
import asyncio

import os
import aiofiles
from fastapi import UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
import mimetypes

import requests
import json
import os
import shutil
from datetime import datetime
from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO
from cryptography.fernet import Fernet
import boto3
from botocore.config import Config

# Load environment variables
load_dotenv()

ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")

if not ENCRYPTION_KEY:
    raise RuntimeError("ENCRYPTION_KEY is required in production")

cipher_suite = Fernet(ENCRYPTION_KEY.encode())


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BiteBids API", version="1.0.0")



CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# ============================================
# DATABASE SETUP - POSTGRESQL WITH SQLALCHEMY
# ============================================

# Database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:root@localhost:5432/bitebids")
                       
FRONTEND_URL = os.getenv("FRONTEND_URL")

BASE_URL = os.getenv("BASE_URL")




SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")   # your email
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")   # app password or smtp key
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USERNAME)


# Add at the top of server.py with other environment variables
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY")
stripe.api_key = STRIPE_SECRET_KEY

# ============================================
# CLOUDFLARE R2 CONFIGURATION
# ============================================
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "bitebids-projects")

# Initialize R2 client (S3-compatible)
r2_client = None
if R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY:
    r2_client = boto3.client(
        's3',
        endpoint_url=f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version='s3v4'),
        region_name='auto'
    )
    logger.info("✅ Cloudflare R2 client initialized")
else:
    logger.warning("⚠️ Cloudflare R2 credentials not configured")

# Create async engine
engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)

# Create async session maker
async_session_maker = async_sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# Base class for models
Base = declarative_base()

# ============================================
# SQLALCHEMY MODELS
# ============================================

class User(Base):
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=True)  # Nullable for OAuth users
    name = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, index=True)
    company = Column(String(255))

        # NEW STATUS FIELD
    status = Column(String(50), default="active", index=True)

    email_verified = Column(Boolean, default=False)
    verification_token = Column(String(255), nullable=True)
    verification_sent_at = Column(TIMESTAMP, nullable=True)

    # ⭐ ADD THESE THREE LINES - Email change verification ⭐
    pending_email = Column(String(255), nullable=True)
    email_change_token = Column(String(255), nullable=True)
    email_change_sent_at = Column(TIMESTAMP, nullable=True)


    address = Column(Text)
    
    # Profile
    avatar = Column(String(500))
    bio = Column(Text)
    skills = Column(ARRAY(Text))
    verified = Column(Boolean, default=False)
    verification_date = Column(TIMESTAMP)
    
    # Statistics
    projects_completed = Column(Integer, default=0)
    total_earnings = Column(DECIMAL(12,2), default=0)
    total_spent = Column(DECIMAL(12,2), default=0)
    avg_rating = Column(DECIMAL(3,2), default=0)
    total_reviews = Column(Integer, default=0)
    response_rate = Column(Integer, default=100)
    on_time_delivery = Column(Integer, default=100)
    reputation_score = Column(Integer, default=0)
    
    # OAuth
    oauth_provider = Column(String(50))
    oauth_id = Column(String(255))
    
    # Profile data (stored as JSONB)
    profile = Column(JSONB)
    
    # Timestamps
    created_at = Column(TIMESTAMP, server_default=sql_func.now())
    updated_at = Column(TIMESTAMP, server_default=sql_func.now(), onupdate=sql_func.now())
    last_login = Column(TIMESTAMP)

    # ✅ ADD THIS LINE
    posting_credits = Column(Integer, default=0)

    # ✅ Developer Payout Preferences
    payout_method = Column(String(50), nullable=True)  # paypal, wise, bank_transfer, crypto, other
    payout_email = Column(String(255), nullable=True)  # For PayPal, Wise
    payout_details = Column(JSONB, nullable=True)  # Bank details, crypto wallet, etc.
    payout_currency = Column(String(10), default='USD')  # Preferred currency
    payout_verified = Column(Boolean, default=False)  # Admin verified payout info

    # Relationship
    projects = relationship("Project", back_populates="developer", foreign_keys="Project.developer_id")

    
    __table_args__ = (
        CheckConstraint("role IN ('developer', 'investor', 'admin')"),
        CheckConstraint("status IN ('active', 'banned', 'suspended', 'pending', 'deleted')"),
    )


class Project(Base):
    __tablename__ = 'projects'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    tech_stack = Column(ARRAY(Text), nullable=False)
    requirements = Column(Text, nullable=False)
    budget_range = Column(String(100))
    budget = Column(DECIMAL(12,2), nullable=False)
    deadline = Column(TIMESTAMP)

    # OWNER = developer
    developer_id = Column(UUID(as_uuid=True),
                          ForeignKey('users.id', ondelete='CASCADE'),
                          nullable=False, index=True)

    developer = relationship("User", back_populates="projects", foreign_keys=[developer_id])

    # Status
    status = Column(String(50), default="open", index=True)
    featured = Column(Boolean, default=False)
    priority = Column(String(20), default="medium")

    # Assignee (when bid accepted)
    assigned_to = Column(UUID(as_uuid=True),
                         ForeignKey('users.id', ondelete='SET NULL'))

    # Bids
    bids_count = Column(Integer, default=0)
    highest_bid = Column(DECIMAL(12,2))
    lowest_bid = Column(DECIMAL(12,2))

    location = Column(String(255))
    remote = Column(Boolean, default=True)

    category = Column(String(50), nullable=False, index=True)
    tags = Column(ARRAY(Text))

    rating = Column(DECIMAL(3,2), default=0)
    reviews_count = Column(Integer, default=0)

    progress = Column(Integer, default=0)

    created_at = Column(TIMESTAMP, server_default=sql_func.now())
    updated_at = Column(TIMESTAMP, server_default=sql_func.now(), onupdate=sql_func.now())
    completed_at = Column(TIMESTAMP)

    images = Column(ARRAY(String), default=[])

        # ✅ ADD THESE THREE LINES FOR PROJECT REVIEW FEATURE
    project_files_path = Column(String(500))  # Path to extracted .rar files
    project_files_uploaded_at = Column(TIMESTAMP)  # When developer uploaded files
    review_button_enabled_until = Column(TIMESTAMP)  # 24-hour window expires at this time

    delivery_type = Column(String(20), default="upload")  
    # values: "git" | "upload"

    delivery_repo_url = Column(String(500))
    delivery_repo_branch = Column(String(100), default="main")
    delivery_repo_commit = Column(String(100))
    delivery_submitted_at = Column(TIMESTAMP)


class GitDeliveryRequest(BaseModel):
    repo_url: str
    branch: str = "main"
    commit: str | None = None


class Bid(Base):
    __tablename__ = 'bids'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True),
                        ForeignKey('projects.id', ondelete='CASCADE'),
                        nullable=False, index=True)

    investor_id = Column(UUID(as_uuid=True),
                         ForeignKey('users.id', ondelete='CASCADE'),
                         nullable=False, index=True)

    amount = Column(DECIMAL(12,2), nullable=False)

    status = Column(String(50), default="pending", index=True)

    created_at = Column(TIMESTAMP, server_default=sql_func.now())
    updated_at = Column(TIMESTAMP, server_default=sql_func.now(), onupdate=sql_func.now())
    accepted_at = Column(TIMESTAMP)

    __table_args__ = (
        CheckConstraint("status IN ('pending','accepted','rejected','withdrawn')"),
        UniqueConstraint('project_id', 'investor_id'),
    )


class Auction(Base):
    __tablename__ = 'auctions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    seller_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Auction details
    starting_bid = Column(DECIMAL(12,2), nullable=False)
    current_bid = Column(DECIMAL(12,2))
    minimum_increment = Column(DECIMAL(12,2), default=100)
    reserve_price = Column(DECIMAL(12,2))
    
    # Timing
    start_time = Column(TIMESTAMP, nullable=False, server_default=sql_func.now())
    end_time = Column(TIMESTAMP, nullable=False, index=True)
    duration_days = Column(Integer, nullable=False)
    
    # Status
    status = Column(String(50), default='active', index=True)
    is_hot = Column(Boolean, default=False)
    
    # Engagement
    bids_count = Column(Integer, default=0)
    highest_bidder_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'))
    watchers_count = Column(Integer, default=0)
    views = Column(Integer, default=0)
    
    # Category
    category = Column(String(100), index=True)
    tags = Column(ARRAY(Text))
    
    # Timestamps
    created_at = Column(TIMESTAMP, server_default=sql_func.now())
    updated_at = Column(TIMESTAMP, server_default=sql_func.now(), onupdate=sql_func.now())
    sold_at = Column(TIMESTAMP)
    
    __table_args__ = (
        CheckConstraint("status IN ('active', 'ended', 'cancelled', 'sold')"),
    )


class AuctionBid(Base):
    __tablename__ = 'auction_bids'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    auction_id = Column(UUID(as_uuid=True), ForeignKey('auctions.id', ondelete='CASCADE'), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    amount = Column(DECIMAL(12,2), nullable=False)
    timestamp = Column(TIMESTAMP, server_default=sql_func.now(), index=True)
    
    __table_args__ = (
        CheckConstraint("amount > 0"),
    )


class CheckoutSession(Base):
    __tablename__ = 'checkout_sessions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Internal references
    session_id = Column(String(255), unique=True, index=True, nullable=True)
    order_reference = Column(String(100), unique=True, index=True, nullable=True)
    external_reference = Column(String(255), nullable=True)

    # NEW FIELDS based on your endpoint
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id', ondelete='SET NULL'), nullable=True, index=True)
    amount = Column(Numeric(12, 2), nullable=False)
    total_with_fees = Column(Numeric(12, 2), nullable=False)
    fees = Column(Numeric(12, 2), nullable=False)
    payment_method = Column(String(100), nullable=False)

    # Customer
    customer_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=True, index=True)

    # Not used in your endpoint but keeping for compatibility
    payment_request = Column(JSONB, nullable=True)
    fee_calculation = Column(JSONB, nullable=True)
    extra_data = Column(JSONB, nullable=True)

    # Status
    status = Column(String(50), default='pending', index=True)

    # Optional payment gateway result info
    payment_url = Column(String(1000), nullable=True)
    payment_method_used = Column(String(100), nullable=True)  # optional

    # Timestamps
    created_at = Column(TIMESTAMP, server_default=sql_func.now())
    expires_at = Column(TIMESTAMP, nullable=True)
    completed_at = Column(TIMESTAMP, nullable=True)

    __table_args__ = (
        CheckConstraint("status IN ('pending', 'completed', 'cancelled', 'expired', 'refunded')"),
    )


class DeveloperPayout(Base):
    """Track developer payouts for completed projects"""
    __tablename__ = 'developer_payouts'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # References
    developer_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id', ondelete='SET NULL'), nullable=True, index=True)
    checkout_session_id = Column(UUID(as_uuid=True), ForeignKey('checkout_sessions.id', ondelete='SET NULL'), nullable=True)
    investor_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'), nullable=True)

    # Amount details
    gross_amount = Column(DECIMAL(12, 2), nullable=False)  # Original project amount
    platform_fee = Column(DECIMAL(12, 2), nullable=False)  # Platform commission
    net_amount = Column(DECIMAL(12, 2), nullable=False)  # Amount to pay developer
    currency = Column(String(10), default='USD')

    # Payout method (snapshot at time of payout request)
    payout_method = Column(String(50), nullable=True)  # paypal, wise, bank_transfer, crypto
    payout_email = Column(String(255), nullable=True)
    payout_details = Column(JSONB, nullable=True)  # Bank details, wallet, etc.

    # Status tracking
    status = Column(String(50), default='pending', index=True)  # pending, processing, completed, failed, cancelled

    # Processing details
    processed_by = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'), nullable=True)  # Admin who processed
    processed_at = Column(TIMESTAMP, nullable=True)
    completed_at = Column(TIMESTAMP, nullable=True)

    # Transaction reference (external payment reference)
    transaction_id = Column(String(255), nullable=True)  # PayPal transaction ID, bank reference, etc.
    transaction_notes = Column(Text, nullable=True)  # Admin notes about the transaction

    # Failure tracking
    failure_reason = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(TIMESTAMP, server_default=sql_func.now())
    updated_at = Column(TIMESTAMP, server_default=sql_func.now(), onupdate=sql_func.now())

    # Description/context
    description = Column(Text, nullable=True)  # e.g., "Payment for project: XYZ"

    __table_args__ = (
        CheckConstraint("status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')"),
        Index('idx_payout_developer_status', 'developer_id', 'status'),
    )


class Notification(Base):
    __tablename__ = 'notifications'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Notification details
    type = Column(String(50), nullable=False)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    link = Column(String(500))
    

    # ✅ ADD THIS: Payment/action metadata (renamed from 'metadata' to avoid SQLAlchemy conflict)
    details = Column(JSONB)


    # Status
    read = Column(Boolean, default=False)
    read_at = Column(TIMESTAMP)
    
    # Timestamp
    created_at = Column(TIMESTAMP, server_default=sql_func.now(), index=True)


class ActivityLog(Base):
    __tablename__ = 'activity_log'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'), index=True)
    user_name = Column(String(255))
    
    # Activity
    type = Column(String(50), nullable=False, index=True)
    action = Column(String(255), nullable=False)
    details = Column(JSONB)
    
    # Metadata
    ip_address = Column(INET)
    user_agent = Column(Text)
    
    # Timestamp
    created_at = Column(TIMESTAMP, server_default=sql_func.now(), index=True)


class ChatRoom(Base):
    __tablename__ = 'chat_rooms'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id', ondelete='CASCADE'), nullable=False, index=True)  # ✅ REMOVED unique=True
    developer_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    investor_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    
    status = Column(String(50), default='active', index=True)
    
    created_at = Column(TIMESTAMP, server_default=sql_func.now())
    closed_at = Column(TIMESTAMP)
    
    __table_args__ = (
        CheckConstraint("status IN ('active', 'closed', 'archived')"),
        UniqueConstraint('project_id', 'investor_id', name='chat_rooms_project_investor_unique'),  # ✅ NEW
    )


class ChatMessage(Base):
    """Messages in chat rooms"""
    __tablename__ = 'chat_messages'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    room_id = Column(UUID(as_uuid=True), ForeignKey('chat_rooms.id', ondelete='CASCADE'), nullable=False, index=True)
    sender_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Message content
    message = Column(Text, nullable=False)
    message_type = Column(String(50), default='text')  # text, file, system
    
    # File attachment (if any)
    file_url = Column(String(500))
    file_name = Column(String(255))
    file_type = Column(String(100))
    file_size = Column(Integer)
    
    # Status
    read = Column(Boolean, default=False)
    read_at = Column(TIMESTAMP)

    # Moderation (for async moderation)
    flagged = Column(Boolean, default=False)
    moderation_status = Column(String(50), default='pending')  # pending, approved, rejected
    moderation_reason = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(TIMESTAMP, default=datetime.utcnow, index=True)
    edited_at = Column(TIMESTAMP)

    __table_args__ = (
        CheckConstraint("message_type IN ('text', 'file', 'system', 'payment_update')"),
        CheckConstraint("moderation_status IN ('pending', 'approved', 'rejected')"),
    )


class ProjectDelivery(Base):
    """Track project deliveries and verification status"""
    __tablename__ = 'project_deliveries'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id', ondelete='CASCADE'), nullable=False, unique=True, index=True)
    chat_room_id = Column(UUID(as_uuid=True), ForeignKey('chat_rooms.id', ondelete='CASCADE'), nullable=False)
    
    # Parties involved
    developer_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    investor_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    
    # Delivery details
    delivery_url = Column(String(500))
    delivery_notes = Column(Text)
    delivery_files = Column(JSONB)
    
    # Status tracking
    status = Column(String(50), default='pending', index=True)
    
    submitted_at = Column(TIMESTAMP)
    reviewed_at = Column(TIMESTAMP)
    approved_at = Column(TIMESTAMP)
    disputed_at = Column(TIMESTAMP)
    resolved_at = Column(TIMESTAMP)
    
    # Dispute information
    dispute_reason = Column(Text)
    dispute_notes = Column(Text)
    dispute_evidence = Column(JSONB)
    
    # Admin resolution
    admin_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    admin_notes = Column(Text)
    resolution = Column(String(50))
    
    # Payment information
    project_amount = Column(DECIMAL(12,2), nullable=False)
    platform_commission_rate = Column(DECIMAL(5,2), default=6.00)
    platform_commission = Column(DECIMAL(12,2))
    developer_payout = Column(DECIMAL(12,2))
    
    payment_released = Column(Boolean, default=False)
    payment_released_at = Column(TIMESTAMP)
    
    created_at = Column(TIMESTAMP, server_default=sql_func.now())
    updated_at = Column(TIMESTAMP, server_default=sql_func.now(), onupdate=sql_func.now())
    
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'submitted', 'under_review', 'approved', 'disputed', 'resolved', 'cancelled')"),
        CheckConstraint("resolution IN ('approve_developer', 'approve_investor', 'partial_refund', 'full_refund', 'custom')"),
    )


class DisputeMessage(Base):
    """Messages during dispute resolution"""
    __tablename__ = 'dispute_messages'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    delivery_id = Column(UUID(as_uuid=True), ForeignKey('project_deliveries.id', ondelete='CASCADE'), nullable=False, index=True)
    sender_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    message = Column(Text, nullable=False)
    attachments = Column(JSONB)
    
    created_at = Column(TIMESTAMP, server_default=sql_func.now())


class ContentFilterLog(Base):
    """Log filtered messages for review"""
    __tablename__ = 'content_filter_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_room_id = Column(UUID(as_uuid=True), ForeignKey('chat_rooms.id', ondelete='CASCADE'), index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), index=True)
    original_message = Column(Text, nullable=False)
    filtered_content = Column(ARRAY(Text))
    action_taken = Column(String(50))
    
    created_at = Column(TIMESTAMP, server_default=sql_func.now())


class ProjectDisputeSimple(Base):
    __tablename__ = "project_disputes_simple"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id', ondelete='CASCADE'), nullable=False, index=True)  # ✅ Changed unique=True to index=True
    reason = Column(String(100), nullable=False)
    notes = Column(Text)
    disputed_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    disputed_at = Column(DateTime, default=datetime.utcnow)
    previous_status = Column(String(50))

    # ✅ NEW: Track which investor this dispute is for (important for fixed_price projects)
    investor_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False, index=True)
    
    # Resolution fields
    resolved = Column(Boolean, default=False)
    resolution = Column(String(50))
    admin_notes = Column(Text)
    resolved_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    resolved_at = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Add this new model for tracking file access
class ProjectFileAccess(Base):
    __tablename__ = 'project_file_access'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id', ondelete='CASCADE'), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    file_path = Column(String(1000), nullable=False)
    accessed_at = Column(TIMESTAMP, server_default=sql_func.now())
    
    __table_args__ = (
        Index('idx_project_file_access', 'project_id', 'user_id'),
    )

    
class ProjectGithubRepo(Base):
    __tablename__ = "project_github_repos"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    room_id = Column(UUID(as_uuid=True), ForeignKey("chat_rooms.id", ondelete='CASCADE'), unique=True, nullable=False)
    repo_url = Column(String(500), nullable=False)
    submitted_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    submitted_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    
    # ✅ NEW: For private repositories
    is_private = Column(Boolean, default=False)
    encrypted_access_token = Column(Text, nullable=True)
    
    __table_args__ = (
        Index('idx_github_repo_room', 'room_id'),
    )


class ProjectUpload(Base):
    """Store direct project uploads to R2 cloud storage"""
    __tablename__ = "project_uploads"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    room_id = Column(UUID(as_uuid=True), ForeignKey("chat_rooms.id", ondelete='CASCADE'), unique=True, nullable=False)

    # Upload info
    file_key = Column(String(500), nullable=False)  # R2 object key
    file_name = Column(String(255), nullable=False)  # Original filename
    file_size = Column(Integer, nullable=False)  # Size in bytes

    # File tree structure (JSON)
    file_tree = Column(JSONB, nullable=False)  # Folder structure for preview

    # Uploader info
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    uploaded_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Status
    status = Column(String(50), default='pending')  # pending, confirmed, downloaded

    __table_args__ = (
        Index('idx_project_upload_room', 'room_id'),
        CheckConstraint("status IN ('pending', 'confirmed', 'downloaded')"),
    )


# ============================================
# CONTENT FILTERING SYSTEM
# ============================================

class ContentFilter:
    """Filter messages to prevent sharing contact information"""
    
    # Patterns to detect contact information
    PHONE_PATTERNS = [
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # US: 123-456-7890
        r'\b\d{10,}\b',  # Any 10+ digit sequence
        r'\+\d{1,3}\s?\d{6,}',  # International: +1 234567890
        r'\(\d{3}\)\s?\d{3}[-.\s]?\d{4}',  # (123) 456-7890
    ]
    
    EMAIL_PATTERNS = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        r'\b[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\s*\.\s*[A-Z|a-z]{2,}\b',  # With spaces
        r'\b[A-Za-z0-9._%+-]+\s*\[at\]\s*[A-Za-z0-9.-]+\s*\[dot\]\s*[A-Z|a-z]{2,}\b',  # Obfuscated
    ]
    
    SOCIAL_PATTERNS = [
        r'@[A-Za-z0-9_]{1,15}\b',  # Twitter/Instagram handles
        r'facebook\.com/[A-Za-z0-9._]+',
        r'linkedin\.com/in/[A-Za-z0-9-]+',
        r'telegram\.me/[A-Za-z0-9_]+',
        r'wa\.me/\d+',  # WhatsApp
        r'snapchat\.com/add/[A-Za-z0-9._-]+',
    ]
    
    BYPASS_PATTERNS = [
        r'\b(contact|call|text|email|phone|whatsapp|telegram|skype)\s+(me|us)\b',
        r'\b(my|our)\s+(email|phone|number|whatsapp)\b',
        r'\b(reach|contact|call)\s+(me|us)\s+(at|on)\b',
        r'\b(dm|message)\s+(me|us)\b',
        r'\b(let\'?s\s+)?(talk|chat|discuss)\s+(outside|off|directly)\b',
    ]
    
    @classmethod
    def check_message(cls, message: str) -> dict:
        """
        Check message for prohibited content
        Returns: {
            'is_safe': bool,
            'violations': list,
            'filtered_message': str
        }
        """
        violations = []
        filtered_message = message
        
        # Check for phone numbers
        for pattern in cls.PHONE_PATTERNS:
            matches = re.findall(pattern, message, re.IGNORECASE)
            if matches:
                violations.append({'type': 'phone', 'matches': matches})
                for match in matches:
                    filtered_message = filtered_message.replace(match, '[PHONE REMOVED]')
        
        # Check for emails
        for pattern in cls.EMAIL_PATTERNS:
            matches = re.findall(pattern, message, re.IGNORECASE)
            if matches:
                violations.append({'type': 'email', 'matches': matches})
                for match in matches:
                    filtered_message = filtered_message.replace(match, '[EMAIL REMOVED]')
        
        # Check for social media
        for pattern in cls.SOCIAL_PATTERNS:
            matches = re.findall(pattern, message, re.IGNORECASE)
            if matches:
                violations.append({'type': 'social', 'matches': matches})
                for match in matches:
                    filtered_message = filtered_message.replace(match, '[SOCIAL MEDIA REMOVED]')
        
        # Check for bypass attempts
        for pattern in cls.BYPASS_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                violations.append({'type': 'bypass_attempt', 'pattern': pattern})
        
        return {
            'is_safe': len(violations) == 0,
            'violations': violations,
            'filtered_message': filtered_message if violations else message,
            'original_message': message
        }


class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self.requests = defaultdict(list)
        self.lock = asyncio.Lock()
    
    async def check_rate_limit(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """
        Check if request is within rate limit
        Returns True if allowed, False if rate limited
        """
        async with self.lock:
            now = datetime.utcnow()
            window_start = now - timedelta(seconds=window_seconds)
            
            # Clean old requests
            self.requests[key] = [
                req_time for req_time in self.requests[key]
                if req_time > window_start
            ]
            
            # Check if within limit
            if len(self.requests[key]) >= max_requests:
                return False
            
            # Add current request
            self.requests[key].append(now)
            return True

# Create global rate limiter
rate_limiter = RateLimiter()


# ============================================
# DEPENDENCY: GET DATABASE SESSION
# ============================================

async def get_db() -> AsyncSession:
    """Dependency to get database session"""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


# ============================================
# HELPER FUNCTIONS
# ============================================

def model_to_dict(model_instance):
    """Convert SQLAlchemy model instance to dictionary"""
    if model_instance is None:
        return None
    
    result = {}
    for column in model_instance.__table__.columns:
        value = getattr(model_instance, column.name)
        
        # Handle UUID
        if isinstance(value, uuid.UUID):
            result[column.name] = str(value)
        # Handle datetime
        elif isinstance(value, datetime):
            # Append 'Z' to indicate UTC timezone for naive datetimes
            result[column.name] = value.isoformat() + 'Z'
        # Handle Decimal
        elif isinstance(value, Decimal):
            result[column.name] = float(value)
        else:
            result[column.name] = value
    
    # Map 'id' to '_id' for backward compatibility with frontend
    if 'id' in result:
        result['_id'] = result['id']
    
    return result


def models_to_list(model_instances):
    """Convert list of SQLAlchemy model instances to list of dictionaries"""
    return [model_to_dict(instance) for instance in model_instances]


# JWT setup
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET is not set")

JWT_ALGORITHM = "HS256"
security = HTTPBearer()

# OAuth setup
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")


# OAuth client configuration
oauth = OAuth()

# GitHub OAuth
if GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET:
    oauth.register(
        name='github',
        client_id=GITHUB_CLIENT_ID,
        client_secret=GITHUB_CLIENT_SECRET,
        server_metadata_url='https://api.github.com/.well-known/oauth_server_metadata',
        client_kwargs={
            'scope': 'user:email'
        }
    )

# Google OAuth
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name='google',
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url='https://accounts.google.com/.well-known/openid_configuration',
        client_kwargs={
            'scope': 'openid email profile'
        }
    )



# 2Checkout setup
CHECKOUT_MERCHANT_CODE = os.getenv("CHECKOUT_MERCHANT_CODE")
CHECKOUT_SECRET_KEY = os.getenv("CHECKOUT_SECRET_KEY")
CHECKOUT_PUBLISHABLE_KEY = os.getenv("CHECKOUT_PUBLISHABLE_KEY")
CHECKOUT_PRIVATE_KEY = os.getenv("CHECKOUT_PRIVATE_KEY")
CHECKOUT_INS_SECRET_WORD = os.getenv("CHECKOUT_INS_SECRET_WORD")
CHECKOUT_BUY_LINK_SECRET_WORD = os.getenv("CHECKOUT_BUY_LINK_SECRET_WORD")
CHECKOUT_ENVIRONMENT = os.getenv("CHECKOUT_ENVIRONMENT", "sandbox")



# Payment calculation constants
PLATFORM_FEE_PERCENTAGE = 6  # 6% platform fee
PLATFORM_FIXED_FEE = 30      # $30 fixed fee
PROJECT_POSTING_FEE = 0.99  # $0.99 fee for posting a project


# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
MODERATION_FAIL_MODE = os.getenv("MODERATION_FAIL_MODE", "fail_closed")


# ============================================
# IMGBB CONFIGURATION
# ============================================
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")
IMGBB_UPLOAD_URL = "https://api.imgbb.com/1/upload"

# ============================================
# FILES CONFIGURATION
# ============================================
UPLOAD_DIR = "uploads/chat_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Image upload configuration (temp directory for processing - images stored on ImgBB)
PROJECT_IMAGES_DIR = "uploads/project_images"
os.makedirs(PROJECT_IMAGES_DIR, exist_ok=True)


ALLOWED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']


# ✅ Load GitHub token from environment
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')


def encrypt_token(token: str) -> str:
    """Encrypt GitHub token before storing"""
    if not cipher_suite:
        raise ValueError("ENCRYPTION_KEY not configured")
    return cipher_suite.encrypt(token.encode()).decode()

def decrypt_token(encrypted_token: str) -> str:
    """Decrypt GitHub token when needed"""
    if not cipher_suite:
        raise ValueError("ENCRYPTION_KEY not configured")
    return cipher_suite.decrypt(encrypted_token.encode()).decode()


# -----------------------------
# BiteBids Ultra-Strict Image Moderation
# -----------------------------
async def openai_image_moderation(image_path: str) -> dict:
    """
    Returns:
        - contains_harmful_content: bool
        - contains_contact_info: bool
    """

    import base64, json, os, re

    if not openai_client:
        logger.error("OpenAI client not initialized")
        return {
            "contains_harmful_content": True,
            "contains_contact_info": True,
        }

    try:
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        ext = os.path.splitext(image_path)[1].lower()
        mime_type = "image/png" if ext == ".png" else "image/jpeg"
        data_url = f"data:{mime_type};base64,{image_base64}"

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an image moderation system for BiteBids, a private project-bidding platform.\n\n"
                        "ALLOWED images:\n"
                        "- Screenshots of code, diagrams, UI mockups, architecture, flowcharts\n"
                        "- Software dashboards, terminals, databases, APIs\n"
                        "- Project-related technical graphics ONLY\n\n"
                        "DISALLOWED images (flag as harmful OR contact info):\n"
                        "- Any human faces, bodies, hands, selfies, avatars, profile photos\n"
                        "- Any real-world photos (rooms, devices, papers, whiteboards, cards)\n"
                        "- Any text containing emails, phone numbers, URLs, QR codes\n"
                        "- Any usernames, social handles, names, logos, watermarks, signatures\n"
                        "- Any attempt to reveal identity or external communication\n"
                        "- Any non-software or non-technical content\n"
                        "- Any harmful, illegal, sexual, violent, or unsafe content\n\n"
                        "If the image is NOT clearly a software or technical project image, it MUST be flagged.\n"
                        "When uncertain, always return true for both fields.\n\n"
                        "Reply ONLY in valid JSON exactly:\n"
                        '{"contains_harmful_content": true/false, "contains_contact_info": true/false}'
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this image strictly."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            max_tokens=50,
            temperature=0.0,
        )

        content = response.choices[0].message.content.strip()
        match = re.search(r"\{[\s\S]*\}", content)

        # Fail closed if response is invalid
        if not match:
            return {
                "contains_harmful_content": True,
                "contains_contact_info": True,
            }

        result = json.loads(match.group())

        return {
            "contains_harmful_content": bool(result.get("contains_harmful_content", True)),
            "contains_contact_info": bool(result.get("contains_contact_info", True)),
        }

    except Exception as e:
        logger.error(f"BiteBids image moderation error: {e}", exc_info=True)
        return {
            "contains_harmful_content": True,
            "contains_contact_info": True,
        }


# -----------------------------
# BiteBids Ultra-Strict Chat Message Moderation
# -----------------------------
async def openai_chat_moderation(message: str) -> dict:
    """
    Enhanced chat message moderation using OpenAI GPT-4o-mini
    
    Returns:
        {
            "is_safe": bool,
            "contains_harmful_content": bool,
            "contains_contact_info": bool,
            "violations": list,
            "reason": str (if not safe)
        }
    """
    
    if not openai_client:
        logger.error("OpenAI client not initialized")
        return {
            "is_safe": False,
            "contains_harmful_content": True,
            "contains_contact_info": True,
            "violations": ["openai_unavailable"],
            "reason": "Moderation service unavailable"
        }
    
    try:
        # Step 1: Use OpenAI's Moderation API (free and fast)
        moderation_response = openai_client.moderations.create(
            model="omni-moderation-latest",
            input=message
        )
        
        moderation_result = moderation_response.results[0]
        
        # Check if content is flagged by moderation API
        if moderation_result.flagged:
            flagged_categories = [
                category for category, flagged in moderation_result.categories.model_dump().items() 
                if flagged
            ]
            
            logger.warning(f"⚠️ Message flagged by OpenAI Moderation: {flagged_categories}")
            
            return {
                "is_safe": False,
                "contains_harmful_content": True,
                "contains_contact_info": False,
                "violations": [{
                    "type": "openai_moderation",
                    "categories": flagged_categories,
                    "details": "Content violates OpenAI moderation policies"
                }],
                "reason": f"Message contains inappropriate content: {', '.join(flagged_categories)}"
            }
        
        # Step 2: Use GPT-4o-mini for contact info and bypass detection
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a chat moderation system for BiteBids, a private project-bidding platform.\n\n"
                        "PLATFORM RULES:\n"
                        "- All communication must stay within the platform\n"
                        "- Users cannot share external contact methods\n"
                        "- Users cannot attempt to move conversations off-platform\n\n"
                        "DISALLOWED content (flag as contact_info):\n"
                        "- Phone numbers (any format: digits, words, symbols)\n"
                        "- Email addresses (including obfuscated: 'john at gmail dot com')\n"
                        "- Social media handles (Twitter, Instagram, LinkedIn, Facebook, etc.)\n"
                        "- Messaging app usernames (WhatsApp, Telegram, Discord, Skype, etc.)\n"
                        "- URLs or website addresses (including without http://)\n"
                        "- QR codes or shortened links (bit.ly, tinyurl, etc.)\n"
                        "- Instructions to contact elsewhere ('DM me', 'text me', 'call me')\n"
                        "- Obfuscated contact info (spaces, dots, 'at', 'dot', emojis)\n"
                        "- Meeting coordination outside platform\n"
                        "- Payment coordination outside platform (Venmo, PayPal, Cash App, etc.)\n\n"
                        "BYPASS ATTEMPTS (flag as bypass_attempt):\n"
                        "- Using coded language to share contact info\n"
                        "- Splitting information across messages\n"
                        "- Using emojis or symbols to hide contact details\n"
                        "- Asking to move to another platform\n"
                        "- Sharing personal identifiable information\n\n"
                        "ALLOWED content:\n"
                        "- Project discussions, requirements, technical details\n"
                        "- Code snippets, documentation links (for the project)\n"
                        "- Timeline discussions, milestone planning\n"
                        "- Professional questions about the project\n"
                        "- File sharing within platform\n\n"
                        "Be strict but not overly paranoid. Words like 'contact', 'call', 'message' in project context are OK.\n"
                        "Example OK: 'I'll contact you through the platform when ready'\n"
                        "Example NOT OK: 'Contact me at john@email.com'\n\n"
                        "Reply ONLY in valid JSON:\n"
                        '{\n'
                        '  "contains_contact_info": true/false,\n'
                        '  "contains_bypass_attempt": true/false,\n'
                        '  "reason": "brief explanation if flagged",\n'
                        '  "detected_items": ["list of detected violations"]\n'
                        '}'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Analyze this message strictly:\n\n{message}",
                },
            ],
            max_tokens=150,
            temperature=0.0,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        import re
        match = re.search(r'\{[\s\S]*\}', content)
        
        if not match:
            logger.error(f"Invalid JSON response from OpenAI: {content}")
            # Fail closed - assume unsafe
            return {
                "is_safe": False,
                "contains_harmful_content": False,
                "contains_contact_info": True,
                "violations": [{
                    "type": "parsing_error",
                    "details": "Unable to parse moderation response"
                }],
                "reason": "Moderation check failed"
            }
        
        result = json.loads(match.group())
        
        contains_contact = bool(result.get("contains_contact_info", False))
        contains_bypass = bool(result.get("contains_bypass_attempt", False))
        
        violations = []
        
        if contains_contact:
            violations.append({
                "type": "contact_info",
                "details": result.get("reason", "Contains contact information"),
                "detected_items": result.get("detected_items", [])
            })
        
        if contains_bypass:
            violations.append({
                "type": "bypass_attempt",
                "details": result.get("reason", "Attempt to bypass platform rules"),
                "detected_items": result.get("detected_items", [])
            })
        
        is_safe = not (contains_contact or contains_bypass)
        
        if not is_safe:
            logger.warning(f"⚠️ Message flagged: {violations}")
        
        return {
            "is_safe": is_safe,
            "contains_harmful_content": False,
            "contains_contact_info": contains_contact,
            "violations": violations,
            "reason": result.get("reason", "") if not is_safe else ""
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in chat moderation: {e}")
        return {
            "is_safe": False,
            "contains_harmful_content": False,
            "contains_contact_info": True,
            "violations": [{"type": "parsing_error", "details": str(e)}],
            "reason": "Moderation check failed"
        }
    except Exception as e:
        logger.error(f"BiteBids chat moderation error: {e}", exc_info=True)
        # Fail closed - assume unsafe
        return {
            "is_safe": False,
            "contains_harmful_content": True,
            "contains_contact_info": True,
            "violations": [{"type": "error", "details": str(e)}],
            "reason": "Moderation service error"
        }


def is_text_file(file_path: str) -> bool:
    """Check if file is a text file that can be read"""
    text_extensions = {
        '.txt', '.md', '.py', '.js', '.jsx', '.ts', '.tsx', '.css', '.html', 
        '.json', '.xml', '.yaml', '.yml', '.ini', '.conf', '.sh', '.bat',
        '.c', '.cpp', '.h', '.java', '.php', '.rb', '.go', '.rs', '.swift',
        '.sql', '.r', '.scala', '.kt', '.dart'
    }
    
    ext = os.path.splitext(file_path)[1].lower()
    return ext in text_extensions


async def read_file_content(file_path: str, max_size: int = 1024 * 1024) -> dict:
    """Read file content with size limit (1MB default)"""
    try:
        file_size = os.path.getsize(file_path)
        
        if file_size > max_size:
            return {
                "success": False,
                "error": "File too large to display",
                "size": file_size
            }
        
        if not is_text_file(file_path):
            return {
                "success": False,
                "error": "File type not supported for preview",
                "extension": os.path.splitext(file_path)[1]
            }
        
        # Try to read as text
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            return {
                "success": True,
                "content": content,
                "size": file_size,
                "lines": len(content.split('\n'))
            }
        except UnicodeDecodeError:
            # Try with different encoding
            async with aiofiles.open(file_path, 'r', encoding='latin-1') as f:
                content = await f.read()
            
            return {
                "success": True,
                "content": content,
                "size": file_size,
                "lines": len(content.split('\n')),
                "encoding": "latin-1"
            }
            
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# ============================================
# Github Helper Funcitons
# ============================================

async def check_if_repo_is_private(owner: str, repo: str, access_token: str = None) -> bool:
    """Check if a GitHub repository is private"""
    try:
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "BiteBids-App"
        }
        
        # Use provided token or system token
        token = access_token or GITHUB_TOKEN
        if token:
            headers["Authorization"] = f"token {token}"
        
        url = f"https://api.github.com/repos/{owner}/{repo}"
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 404:
            # Could be private or doesn't exist - assume private
            logger.info(f"Repository {owner}/{repo} returned 404 - likely private or doesn't exist")
            return True
        
        if response.status_code == 200:
            data = response.json()
            is_private = data.get('private', False)
            logger.info(f"Repository {owner}/{repo} is {'private' if is_private else 'public'}")
            return is_private
        
        # On other errors, assume public to allow attempt
        return False
    except Exception as e:
        logger.error(f"Error checking if repo is private: {e}")
        return False

def parse_github_url(repo_url: str) -> dict:
    """
    Parse GitHub repository URL to extract owner and repo name
    Example: https://github.com/facebook/react -> {'owner': 'facebook', 'repo': 'react'}
    """
    import re
    pattern = r'github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$'
    match = re.search(pattern, repo_url)
    
    if not match:
        raise ValueError("Invalid GitHub repository URL")
    
    return {
        'owner': match.group(1),
        'repo': match.group(2)
    }

def get_github_headers() -> dict:
    """Get headers for GitHub API requests with authentication"""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "BiteBids-App"  # GitHub requires User-Agent
    }
    
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
        logger.info("✅ Using authenticated GitHub API requests")
    else:
        logger.warning("⚠️ Using unauthenticated GitHub API (60 req/hour limit)")
    
    return headers

def check_github_rate_limit() -> Optional[Dict]:
    """Check GitHub API rate limit status"""
    try:
        headers = get_github_headers()
        response = requests.get("https://api.github.com/rate_limit", headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            core = data['resources']['core']
            
            logger.info(f"📊 GitHub API Rate Limit: {core['remaining']}/{core['limit']}")
            
            if core['remaining'] < 10:
                logger.warning(f"⚠️ GitHub API rate limit low: {core['remaining']} requests remaining")
            
            return core
        
        return None
    except Exception as e:
        logger.error(f"Error checking rate limit: {e}")
        return None

def build_tree_structure(items: List[Dict]) -> List[Dict]:
    """
    Build a hierarchical tree structure from GitHub API flat list
    Remove duplicates and create nested structure
    """
    # Separate files and folders
    files_dict = {}
    folders_dict = {}
    
    for item in items:
        path = item['path']
        
        if item['type'] == 'blob':
            files_dict[path] = {
                'path': path,
                'type': 'blob',
                'size': item.get('size', 0)
            }
        elif item['type'] == 'tree':
            if path not in folders_dict:
                folders_dict[path] = {
                    'path': path,
                    'type': 'tree',
                    'size': item.get('size', 0),
                    'children': []
                }
    
    # Build hierarchy
    root_items = []
    
    for path, folder in folders_dict.items():
        folder_path_prefix = path + '/'
        
        # Add child files
        for file_path, file_data in files_dict.items():
            if file_path.startswith(folder_path_prefix):
                relative_path = file_path[len(folder_path_prefix):]
                if '/' not in relative_path:
                    folder['children'].append(file_data)
        
        # Add child folders
        for child_path, child_folder in folders_dict.items():
            if child_path.startswith(folder_path_prefix) and child_path != path:
                relative_path = child_path[len(folder_path_prefix):]
                if '/' not in relative_path:
                    folder['children'].append(child_folder)
        
        # Root level folder
        if '/' not in path:
            root_items.append(folder)
    
    # Add root level files
    for file_path, file_data in files_dict.items():
        if '/' not in file_path:
            root_items.append(file_data)
    
    return root_items

def fetch_github_tree(owner: str, repo: str, branch: str = "main", encrypted_token: str = None) -> Optional[List[Dict]]:
    """
    Fetch repository tree structure from GitHub API
    Supports private repositories via encrypted access token
    """
    try:
        # Check rate limit before making request
        rate_limit = check_github_rate_limit()
        if rate_limit and rate_limit['remaining'] < 5:
            logger.error("❌ GitHub API rate limit too low, skipping request")
            return None
        
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        headers = get_github_headers()
        
        # ✅ Use developer's token if provided (for private repos)
        if encrypted_token:
            try:
                token = decrypt_token(encrypted_token)
                headers["Authorization"] = f"token {token}"
                logger.info(f"🔒 Using developer's token for private repo: {owner}/{repo}")
            except Exception as e:
                logger.error(f"Error decrypting token: {e}")
                return None
        
        logger.info(f"📡 Fetching tree for {owner}/{repo} (branch: {branch})")
        response = requests.get(url, headers=headers, timeout=15)
        
        # Try master branch if main doesn't exist
        if response.status_code == 404:
            logger.info(f"Branch '{branch}' not found, trying 'master'")
            url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/master?recursive=1"
            response = requests.get(url, headers=headers, timeout=15)
        
        # Handle authentication errors
        if response.status_code == 401:
            logger.error("❌ Invalid or expired GitHub token")
            return None
        
        if response.status_code == 404:
            logger.error(f"❌ Repository not found or no access: {owner}/{repo}")
            return None
        
        # Handle rate limit
        if response.status_code == 403:
            rate_remaining = response.headers.get('X-RateLimit-Remaining', 'unknown')
            rate_reset = response.headers.get('X-RateLimit-Reset', 'unknown')
            
            logger.error(f"❌ GitHub API rate limit exceeded!")
            logger.error(f"   Remaining: {rate_remaining}")
            logger.error(f"   Resets at: {rate_reset}")
            
            if not GITHUB_TOKEN and not encrypted_token:
                logger.error("💡 Add GITHUB_TOKEN to .env to increase rate limit from 60 to 5000 req/hour")
            
            return None
        
        response.raise_for_status()
        data = response.json()
        
        # Get tree items
        items = data.get('tree', [])
        
        if not items:
            logger.warning(f"⚠️ No items found in repository tree")
            return []
        
        # Build deduplicated hierarchical structure
        tree_structure = build_tree_structure(items)
        
        logger.info(f"✅ Fetched {len(items)} items, built tree with {len(tree_structure)} root items")
        
        return tree_structure
        
    except requests.exceptions.Timeout:
        logger.error(f"⏱️ Timeout fetching GitHub tree for {owner}/{repo}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ GitHub API error: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected error in fetch_github_tree: {e}", exc_info=True)
        return None

def fetch_github_file_content(owner: str, repo: str, file_path: str, branch: str = "main", encrypted_token: str = None) -> str:
    """
    Fetch file content from GitHub repository
    Supports private repositories via encrypted access token
    """
    try:
        # Check rate limit
        rate_limit = check_github_rate_limit()
        if rate_limit and rate_limit['remaining'] < 5:
            return "Error: GitHub API rate limit exceeded. Please try again later."
        
        # Encode file path for URL
        encoded_path = requests.utils.quote(file_path)
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{encoded_path}?ref={branch}"
        headers = get_github_headers()
        
        # ✅ Use developer's token if provided (for private repos)
        if encrypted_token:
            try:
                token = decrypt_token(encrypted_token)
                headers["Authorization"] = f"token {token}"
                logger.info(f"🔒 Using developer's token for private file: {file_path}")
            except Exception as e:
                logger.error(f"Error decrypting token: {e}")
                return "Error: Failed to decrypt access token"
        
        logger.info(f"📄 Fetching file: {file_path}")
        response = requests.get(url, headers=headers, timeout=15)
        
        # Try master branch if main doesn't exist
        if response.status_code == 404:
            url = f"https://api.github.com/repos/{owner}/{repo}/contents/{encoded_path}?ref=master"
            response = requests.get(url, headers=headers, timeout=15)
        
        # Handle authentication errors
        if response.status_code == 401:
            return "Error: Invalid or expired access token. Please update your repository access token."
        
        # Handle rate limit
        if response.status_code == 403:
            logger.error(f"❌ GitHub API rate limit exceeded while fetching file")
            return "Error: GitHub API rate limit exceeded. Please add GITHUB_TOKEN to .env file."
        
        if response.status_code == 404:
            return "Error: File not found or no access to this repository."
        
        response.raise_for_status()
        data = response.json()
        
        # Decode content from base64
        content = base64.b64decode(data['content']).decode('utf-8')
        
        logger.info(f"✅ Successfully fetched file: {file_path} ({len(content)} bytes)")
        
        return content
        
    except requests.exceptions.Timeout:
        logger.error(f"⏱️ Timeout fetching file: {file_path}")
        return "Error: Request timeout. Please try again."
    except UnicodeDecodeError:
        logger.error(f"❌ Binary file detected: {file_path}")
        return "Error: This file appears to be binary and cannot be displayed as text."
    except Exception as e:
        logger.error(f"❌ Error fetching file content: {e}", exc_info=True)
        return f"Error loading file: {str(e)}"



# ============================================
# PYDANTIC MODELS
# ============================================

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    role: str  # developer, investor, admin
    status: Optional[str] = "active"
    name: str
    company: Optional[str] = None
    address: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class OAuthValidation(BaseModel):
    session_id: str


class RoleUpdate(BaseModel):
    role: str


class ProjectCreate(BaseModel):
    title: str
    status: str
    description: str
    tech_stack: List[str]
    requirements: str = ""
    budget: float  # NEW - Direct budget field
    lowest_bid: float
    budget_range: Optional[str] = None
    deadline: Optional[str] = None
    location: Optional[str] = "Remote"  # NEW
    category: str = "Machine Learning"  # NEW - Default to Machine Learning
    images: Optional[List[str]] = []  # ✅ NEW FIELD


class ProjectUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    tech_stack: Optional[List[str]] = None
    requirements: Optional[str] = None
    budget: Optional[float] = None
    lowest_bid: Optional[float] = None
    budget_range: Optional[str] = None
    deadline: Optional[str] = None
    location: Optional[str] = None
    category: Optional[str] = None
    status: Optional[str] = None
    images: Optional[List[str]] = None  # ✅ ADD THIS LINE


class BidCreate(BaseModel):
    amount: float


class PricingRequest(BaseModel):
    title: str
    description: str
    tech_stack: List[str]
    requirements: str
    complexity: Optional[str] = "medium"


class PaymentRequest(BaseModel):
    order_type: str  # 'auction' or 'fixed'
    item_id: str
    customer_email: str
    customer_name: str
    billing_address: Dict[str, str]
    payment_method: str
    amount: float
    auction_id: Optional[str] = None


class CheckoutSessionResponse(BaseModel):
    session_id: str
    order_reference: str
    payment_url: str
    expires_at: datetime


class PaymentResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    payment_url: Optional[str] = None
    order_reference: Optional[str] = None
    message: str


class OAuthConfig(BaseModel):
    client_id: str
    client_secret: str


class OAuthSetupResponse(BaseModel):
    provider: str
    redirect_url: str
    configured: bool
    message: str


class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    company: Optional[str] = None
    address: Optional[str] = None
    avatar: Optional[str] = None
    bio: Optional[str] = None
    skills: Optional[List[str]] = None
    profile: Optional[Dict[str, Any]] = None


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    role: str
    company: Optional[str] = None
    address: Optional[str] = None
    avatar: Optional[str] = None
    bio: Optional[str] = None
    skills: Optional[List[str]] = None
    verified: bool
    projects_completed: int
    total_earnings: float
    total_spent: float
    avg_rating: float
    total_reviews: int
    response_rate: int
    on_time_delivery: int
    reputation_score: int
    profile: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    pending_email: Optional[str] = None


class UserListResponse(BaseModel):
    users: List[UserResponse]
    total: int
    page: int
    page_size: int


class StripeCheckoutRequest(BaseModel):
    order_type: str  # 'auction' or 'fixed'
    item_id: str
    customer_email: str
    customer_name: str
    billing_address: dict
    payment_method: str
    amount: float
    auction_id: Optional[str] = None
    winner_bid_id: Optional[str] = None
    project_id: Optional[str] = None
    notification_id: Optional[str] = None


class ChatMessageCreate(BaseModel):
    message: str
    message_type: Optional[str] = "text"
    file_url: Optional[str] = None
    file_name: Optional[str] = None
    file_type: Optional[str] = None
    file_size: Optional[int] = None


class DeliverySubmitRequest(BaseModel):
    delivery_url: Optional[str] = None
    delivery_notes: str
    

class DeliveryApprovalRequest(BaseModel):
    approved: bool
    feedback: Optional[str] = None


class DisputeCreateRequest(BaseModel):
    reason: str
    notes: Optional[str] = None
    evidence: Optional[List[str]] = None


class DisputeResolveRequest(BaseModel):
    resolution: str
    admin_notes: str
    developer_payout: Optional[float] = None
    investor_refund: Optional[float] = None


# ============================================
# DEVELOPER PAYOUT SCHEMAS
# ============================================
class PayoutPreferencesUpdate(BaseModel):
    payout_method: str  # paypal, wise, bank_transfer, crypto, other
    payout_email: Optional[str] = None
    payout_details: Optional[dict] = None  # Bank details, crypto wallet, etc.
    payout_currency: Optional[str] = 'USD'


class PayoutProcessRequest(BaseModel):
    transaction_id: Optional[str] = None  # External transaction reference
    transaction_notes: Optional[str] = None  # Admin notes


class PayoutCompleteRequest(BaseModel):
    transaction_id: str  # Required - proof of payment
    transaction_notes: Optional[str] = None


class PayoutFailRequest(BaseModel):
    failure_reason: str


class ContactFormSubmission(BaseModel):
    name: str
    email: EmailStr
    subject: str
    category: str
    message: str
    
    # @validator('name')
    # def validate_name(cls, v):
    #     if not v or len(v.strip()) < 2:
    #         raise ValueError('Name must be at least 2 characters')
    #     if len(v) > 100:
    #         raise ValueError('Name must be less than 100 characters')
    #     return v.strip()
    
    # @validator('subject')
    # def validate_subject(cls, v):
    #     if not v or len(v.strip()) < 3:
    #         raise ValueError('Subject must be at least 3 characters')
    #     if len(v) > 200:
    #         raise ValueError('Subject must be less than 200 characters')
    #     return v.strip()
    
    # @validator('message')
    # def validate_message(cls, v):
    #     if not v or len(v.strip()) < 10:
    #         raise ValueError('Message must be at least 10 characters')
    #     if len(v) > 5000:
    #         raise ValueError('Message must be less than 5000 characters')
    #     return v.strip()
    
    # @validator('category')
    # def validate_category(cls, v):
    #     allowed_categories = ['general', 'technical', 'billing', 'partnership', 'feedback']
    #     if v not in allowed_categories:
    #         raise ValueError(f'Category must be one of: {", ".join(allowed_categories)}')
    #     return v

class ContactFormRecord(Base):
    __tablename__ = 'contact_form_submissions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    email = Column(String(255), nullable=False)
    subject = Column(String(200), nullable=False)
    category = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    submitted_at = Column(TIMESTAMP, server_default=sql_func.now())
    responded = Column(Boolean, default=False)
    responded_at = Column(TIMESTAMP)
    
    __table_args__ = (
        CheckConstraint("category IN ('general', 'technical', 'billing', 'partnership', 'feedback')"),
    )
    
# ============================================
# AUTHENTICATION HELPERS
# ============================================

def hash_password(password: str) -> str:
    """Hash a password"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against a hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


def create_jwt_token(user_dict: dict) -> str:
    """Create JWT token for user"""
    user_id = user_dict.get("_id") or user_dict.get("id")

    # ✅ Ensure UUIDs are converted to strings
    if isinstance(user_id, UUID):
        user_id = str(user_id)

    payload = {
        "user_id": user_id,
        "email": user_dict["email"],
        "role": user_dict["role"],
        # ✅ Use timezone-aware datetime (fixes DeprecationWarning)
        "exp": datetime.now(timezone.utc) + timedelta(days=7)
    }

        # 🔥 Add admin access flag if user is admin
    if user_dict["role"] == "admin":
        payload["admin"] = True

    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def send_verification_email(to_email: str, token: str):
    """Send email verification link to the user."""
    try:
        verification_link = f"{FRONTEND_URL}/verify-email?token={token}"

        subject = "Verify your BiteBids account 🚀"
        
        html_content = f"""\
        <!DOCTYPE html>
        <html lang="en">

        <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Email Verification</title>

        <style>
            @media only screen and (max-width: 600px) {{
            .container {{
                width: 100% !important;
                padding: 20px !important;
            }}
            .card {{
                padding: 25px !important;
            }}
            .btn-primary {{
                padding: 14px 26px !important;
                font-size: 16px !important;
            }}
            }}
        </style>
        </head>

        <body style="margin:0; padding:0; background:#f8fafc; font-family:'Segoe UI', Arial, sans-serif;">

        <table width="100%" border="0" cellspacing="0" cellpadding="0"
            style="background:#f8fafc; padding:40px 0;">
            <tr>
            <td align="center">

                <table class="container card" width="600" border="0" cellspacing="0" cellpadding="0"
                style="background:white; border-radius:16px; box-shadow:0 4px 20px rgba(0,0,0,0.06); padding:40px;">

                <tr>
                    <td align="center" style="padding-bottom:25px;">
                    <h1 style="margin:0; font-size:28px; font-weight:700; color:#4f46e5;">
                        BiteBids
                    </h1>
                    <p style="margin:8px 0 0; color:#6b7280; font-size:14px;">
                        Verify your account to get started
                    </p>
                    </td>
                </tr>

                <tr>
                    <td align="center" style="padding-bottom:15px;">
                    <h2 style="margin:0; font-size:22px; color:#111827; font-weight:600;">
                        Welcome to BiteBids! 🎉
                    </h2>
                    </td>
                </tr>

                <tr>
                    <td style="padding-bottom:25px;">
                    <p style="margin:0; color:#374151; font-size:15px; line-height:1.6; text-align:center;">
                        Thank you for joining our platform!<br>
                        Please verify your email address to continue.
                    </p>
                    </td>
                </tr>

                <tr>
                    <td align="center" style="padding-bottom:30px;">
                    <a href="{verification_link}" target="_blank"
                        style="display:inline-block; background:#4f46e5; color:white;
                        padding:14px 40px; font-size:17px; font-weight:600;
                        border-radius:10px; text-decoration:none;">
                        Verify Email
                    </a>
                    </td>
                </tr>

                <tr>
                    <td style="padding-bottom:25px;">
                    <p style="color:#6b7280; font-size:13px; line-height:1.5;">
                        If the button doesn't work, copy the link below:
                    </p>
                    <p style="word-break:break-all;">
                        <a href="{verification_link}" style="color:#4f46e5; font-size:13px;">
                        {verification_link}
                        </a>
                    </p>
                    </td>
                </tr>

                <tr>
                    <td style="padding:20px 0;">
                    <hr style="border:0; border-top:1px solid #e5e7eb;" />
                    </td>
                </tr>

                <tr>
                    <td align="center" style="padding-bottom:15px;">
                    <p style="margin:0; font-size:13px; color:#6b7280;">
                        Need help? Contact us:
                        <a href="mailto:bitebids@gmail.com" style="color:#4f46e5;">
                        bitebids@gmail.com
                        </a>
                    </p>
                    </td>
                </tr>

                <tr>
                    <td align="center">
                    <p style="margin:0; font-size:11px; color:#9ca3af;">
                        © 2024 BiteBids. All rights reserved.
                    </p>
                    </td>
                </tr>

                </table>
            </td>
            </tr>
        </table>

        </body>
        </html>
        """



        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM
        msg["To"] = to_email

        msg.attach(MIMEText(html_content, "html"))

        # Send email
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(EMAIL_FROM, to_email, msg.as_string())
        server.quit()

        print(f"📧 Verification email sent to {to_email}")

    except Exception as e:
        print("❌ Email sending failed:", str(e))


async def send_payment_confirmation_email(
    to_email: str,
    project_title: str,
    amount: float,
    role: str = "investor"
):
    """
    Send a payment confirmation email.
    role: 'investor' or 'developer'
    """
    try:
        subject = (
            f"✅ Payment successful for '{project_title}'"
            if role == "investor"
            else f"💰 Client payment received for '{project_title}'"
        )

        role_message = (
            "Your payment was successfully processed and held in BiteBids escrow."
            if role == "investor"
            else "Your client has completed payment. Funds are now held in BiteBids escrow."
        )

        safe_amount = float(amount or 0)

        html_content = f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <title>BiteBids Payment Confirmation</title>
        </head>
        <body style="margin:0; padding:0; background:#f8fafc; font-family:'Segoe UI', Arial, sans-serif;">
            <table width="100%" border="0" cellspacing="0" cellpadding="0"
                   style="background:#f8fafc; padding:40px 0;">
                <tr>
                    <td align="center">
                        <table width="600" border="0" cellspacing="0" cellpadding="0"
                               style="background:white; border-radius:16px; box-shadow:0 4px 20px rgba(0,0,0,0.06); padding:40px;">
                            <tr>
                                <td align="center" style="padding-bottom:20px;">
                                    <h1 style="margin:0; font-size:26px; font-weight:700; color:#4f46e5;">
                                        BiteBids
                                    </h1>
                                    <p style="margin:8px 0 0; color:#6b7280; font-size:14px;">
                                        Payment confirmation
                                    </p>
                                </td>
                            </tr>

                            <tr>
                                <td style="padding-bottom:20px;">
                                    <h2 style="margin:0; font-size:20px; color:#111827; font-weight:600;">
                                        {subject}
                                    </h2>
                                </td>
                            </tr>

                            <tr>
                                <td style="padding-bottom:15px;">
                                    <p style="margin:0; color:#374151; font-size:15px; line-height:1.6;">
                                        {role_message}
                                    </p>
                                </td>
                            </tr>

                            <tr>
                                <td style="padding-bottom:20px;">
                                    <p style="margin:0; color:#111827; font-size:15px; font-weight:600;">
                                        Project: <span style="color:#4f46e5;">{project_title}</span><br/>
                                        Amount: <span style="color:#059669;">${safe_amount:,.2f}</span>
                                    </p>
                                </td>
                            </tr>

                            <tr>
                                <td style="padding-top:10px; border-top:1px solid #e5e7eb;">
                                    <p style="margin:10px 0 0; font-size:12px; color:#9ca3af;">
                                        This payment is processed securely and held in escrow by BiteBids.
                                    </p>
                                </td>
                            </tr>

                            <tr>
                                <td align="center" style="padding-top:20px;">
                                    <p style="margin:0; font-size:11px; color:#9ca3af;">
                                        © 2024 BiteBids. All rights reserved.
                                    </p>
                                </td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
        </body>
        </html>
        """

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM
        msg["To"] = to_email

        msg.attach(MIMEText(html_content, "html"))

        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(EMAIL_FROM, to_email, msg.as_string())
        server.quit()

        print(f"📧 Payment confirmation email sent to {to_email} ({role})")

    except Exception as e:
        print("❌ Payment email sending failed:", str(e))


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """Get current user from JWT token"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("user_id")
        
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Convert user_id to UUID if it's a string
        if isinstance(user_id, str):
            user_id = uuid.UUID(user_id)
        
        # Query user from database
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        
        # Ensure we return a dictionary with proper UUID handling
        user_dict = model_to_dict(user)
        
        # Ensure user_id is a string in the returned dict
        if 'id' in user_dict and isinstance(user_dict['id'], uuid.UUID):
            user_dict['id'] = str(user_dict['id'])
        if '_id' in user_dict and isinstance(user_dict['_id'], uuid.UUID):
            user_dict['_id'] = str(user_dict['_id'])
            
        return user_dict
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except ValueError as e:
        raise HTTPException(status_code=401, detail=f"Invalid user ID format: {str(e)}")

async def get_current_admin(
    current_user: dict = Depends(get_current_user)
):
    """Verify current user is admin"""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=403, 
            detail="Admin access required"
        )
    return current_user

async def get_user_or_self(
    user_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user if admin or if user is requesting their own data"""
    current_user_id = current_user.get("_id") or current_user.get("id")
    
    # Allow access if user is requesting their own data or is admin
    if user_id == current_user_id or current_user.get("role") == "admin":
        result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return user
    
    raise HTTPException(status_code=403, detail="Access denied")


def model_to_dict(model_instance):
    """Convert SQLAlchemy model instance to dictionary"""
    if model_instance is None:
        return None
    
    result = {}
    for column in model_instance.__table__.columns:
        value = getattr(model_instance, column.name)
        
        # Handle UUID
        if isinstance(value, uuid.UUID):
            result[column.name] = str(value)
        # Handle asyncpg UUID (pgproto.UUID)
        elif hasattr(value, '__class__') and 'pgproto.UUID' in str(value.__class__):
            result[column.name] = str(value)
        # Handle datetime
        elif isinstance(value, datetime):
            # Append 'Z' to indicate UTC timezone for naive datetimes
            result[column.name] = value.isoformat() + 'Z'
        # Handle Decimal
        elif isinstance(value, Decimal):
            result[column.name] = float(value)
        else:
            result[column.name] = value
    
    # Map 'id' to '_id' for backward compatibility with frontend
    if 'id' in result:
        result['_id'] = result['id']
    
    return result


async def send_admin_project_notification(
    action: str,  # "created", "updated", or "deleted"
    project_id: str,
    project_title: str,
    developer_name: str,
    developer_email: str,
    project_data: dict = None
):
    """
    Send email notification to BiteBids admin when project is created/updated/deleted
    """
    try:
        admin_email = "bitebids@gmail.com"
        
        # Action-specific configuration
        action_config = {
            "created": {
                "emoji": "🎉",
                "color": "#22c55e",
                "title": "New Project Posted",
                "message": "A new project has been posted on BiteBids!"
            },
            "updated": {
                "emoji": "✏️",
                "color": "#3b82f6",
                "title": "Project Updated",
                "message": "A project has been updated on BiteBids."
            },
            "deleted": {
                "emoji": "🗑️",
                "color": "#ef4444",
                "title": "Project Deleted",
                "message": "A project has been deleted from BiteBids."
            }
        }
        
        config = action_config.get(action, action_config["created"])
        
        # Admin dashboard link
        admin_link = f"{FRONTEND_URL}/admin-dashboard?project={project_id}"
        
        subject = f"{config['emoji']} {config['title']} - {project_title}"
        
        # Build project details section
        project_details = ""
        if project_data and action != "deleted":
            project_details = f"""
            <tr>
                <td style="padding: 20px 0;">
                    <table width="100%" border="0" cellspacing="0" cellpadding="0" 
                           style="background: #f8fafc; border-radius: 8px; padding: 20px; border-left: 4px solid {config['color']};">
                        <tr>
                            <td>
                                <h3 style="margin: 0 0 15px 0; font-size: 16px; color: #1f2937; font-weight: 600;">
                                    Project Details:
                                </h3>
                                
                                <table width="100%" border="0" cellspacing="0" cellpadding="0">
                                    <tr>
                                        <td style="padding: 8px 0; color: #6b7280; font-size: 14px; width: 140px;">
                                            <strong>Category:</strong>
                                        </td>
                                        <td style="padding: 8px 0; color: #111827; font-size: 14px;">
                                            {project_data.get('category', 'N/A')}
                                        </td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 8px 0; color: #6b7280; font-size: 14px;">
                                            <strong>Budget:</strong>
                                        </td>
                                        <td style="padding: 8px 0; color: #111827; font-size: 14px;">
                                            ${project_data.get('budget', 0):,.2f}
                                        </td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 8px 0; color: #6b7280; font-size: 14px;">
                                            <strong>Status:</strong>
                                        </td>
                                        <td style="padding: 8px 0; color: #111827; font-size: 14px;">
                                            {project_data.get('status', 'open').upper()}
                                        </td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 8px 0; color: #6b7280; font-size: 14px;">
                                            <strong>Location:</strong>
                                        </td>
                                        <td style="padding: 8px 0; color: #111827; font-size: 14px;">
                                            {project_data.get('location', 'Remote')}
                                        </td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 8px 0; color: #6b7280; font-size: 14px; vertical-align: top;">
                                            <strong>Tech Stack:</strong>
                                        </td>
                                        <td style="padding: 8px 0; color: #111827; font-size: 14px;">
                                            {', '.join(project_data.get('tech_stack', [])) if project_data.get('tech_stack') else 'N/A'}
                                        </td>
                                    </tr>
                                </table>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
            """
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <title>Admin Notification</title>
        </head>
        
        <body style="margin:0; padding:0; background:#f8fafc; font-family:'Segoe UI', Arial, sans-serif;">
            <table width="100%" border="0" cellspacing="0" cellpadding="0" style="background:#f8fafc; padding:40px 0;">
                <tr>
                    <td align="center">
                        <table class="container" width="600" border="0" cellspacing="0" cellpadding="0"
                               style="background:white; border-radius:16px; box-shadow:0 4px 20px rgba(0,0,0,0.06); padding:40px;">
                            
                            <!-- Header -->
                            <tr>
                                <td align="center" style="padding-bottom: 30px; border-bottom: 2px solid #f3f4f6;">
                                    <h1 style="margin:0; font-size:28px; font-weight:700; color:#4f46e5;">
                                        BiteBids Admin
                                    </h1>
                                    <p style="margin:8px 0 0; color:#6b7280; font-size:14px;">
                                        Project Management Notification
                                    </p>
                                </td>
                            </tr>
                            
                            <!-- Action Badge -->
                            <tr>
                                <td align="center" style="padding: 30px 0 20px 0;">
                                    <div style="display: inline-block; background: {config['color']}; color: white; 
                                                padding: 12px 24px; border-radius: 8px; font-size: 16px; font-weight: 600;">
                                        {config['emoji']} {config['title']}
                                    </div>
                                </td>
                            </tr>
                            
                            <!-- Main Message -->
                            <tr>
                                <td align="center" style="padding-bottom: 20px;">
                                    <h2 style="margin:0; font-size:22px; color:#111827; font-weight:600;">
                                        {project_title}
                                    </h2>
                                    <p style="margin:10px 0 0; color:#6b7280; font-size:16px;">
                                        {config['message']}
                                    </p>
                                </td>
                            </tr>
                            
                            <!-- Developer Info -->
                            <tr>
                                <td style="padding: 20px 0;">
                                    <table width="100%" border="0" cellspacing="0" cellpadding="0" 
                                           style="background: #fef3c7; border-radius: 8px; padding: 16px; border-left: 4px solid #f59e0b;">
                                        <tr>
                                            <td>
                                                <p style="margin: 0 0 8px 0; font-size: 14px; color: #92400e; font-weight: 600;">
                                                    👤 Developer Information:
                                                </p>
                                                <p style="margin: 0; font-size: 14px; color: #78350f;">
                                                    <strong>Name:</strong> {developer_name}<br/>
                                                    <strong>Email:</strong> {developer_email}
                                                </p>
                                            </td>
                                        </tr>
                                    </table>
                                </td>
                            </tr>
                            
                            <!-- Project Details -->
                            {project_details}
                            
                            <!-- Project ID -->
                            <tr>
                                <td style="padding: 20px 0;">
                                    <p style="margin: 0; font-size: 13px; color: #9ca3af; text-align: center;">
                                        Project ID: <code style="background: #f3f4f6; padding: 4px 8px; border-radius: 4px; color: #6b7280;">{project_id}</code>
                                    </p>
                                </td>
                            </tr>
                            
                            <!-- Action Button -->
                            <tr>
                                <td align="center" style="padding: 20px 0 30px 0;">
                                    <a href="{admin_link}" 
                                       style="display:inline-block; background:{config['color']}; color:white; 
                                              padding:16px 40px; border-radius:8px; text-decoration:none; 
                                              font-weight:600; font-size:16px; box-shadow:0 4px 12px rgba(0,0,0,0.15);">
                                        View in Admin Dashboard →
                                    </a>
                                </td>
                            </tr>
                            
                            <!-- Footer -->
                            <tr>
                                <td align="center" style="padding-top: 30px; border-top: 2px solid #f3f4f6;">
                                    <p style="margin:0; color:#9ca3af; font-size:13px;">
                                        This is an automated notification from BiteBids<br/>
                                        <a href="{FRONTEND_URL}" style="color:#4f46e5; text-decoration:none;">Visit BiteBids</a>
                                    </p>
                                </td>
                            </tr>
                            
                        </table>
                    </td>
                </tr>
            </table>
        </body>
        </html>
        """
        
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM
        msg["To"] = admin_email
        
        msg.attach(MIMEText(html_content, "html"))
        
        # Send email
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(EMAIL_FROM, admin_email, msg.as_string())
        server.quit()
        
        logger.info(f"📧 Admin notification sent for {action} project: {project_title}")
        
    except Exception as e:
        logger.error(f"❌ Failed to send admin notification: {str(e)}")

async def send_developer_edit_notification(
    developer_email: str,
    developer_name: str,
    project_title: str,
    project_id: str,
    admin_name: str,
    changes: dict
):
    """
    Send email notification to developer when admin edits their project
    
    Args:
        developer_email: Developer's email
        developer_name: Developer's name
        project_title: Project title
        project_id: Project ID
        admin_name: Admin who made the edit
        changes: Dictionary of changes made (field -> old_value, new_value)
    """
    try:
        subject = f"⚠️ Admin Updated Your Project: {project_title}"
        
        # Build changes HTML
        changes_html = ""
        for field, change_data in changes.items():
            old_value = change_data.get('old')
            new_value = change_data.get('new')
            reason = change_data.get('reason', '')
            
            # Format field name nicely
            field_name = field.replace('_', ' ').title()
            
            # Handle different value types
            if field == 'images':
                old_count = len(old_value) if old_value else 0
                new_count = len(new_value) if new_value else 0
                
                if old_count > new_count:
                    changes_html += f"""
                    <tr>
                        <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                            <strong style="color: #ef4444;">🗑️ {field_name}</strong>
                        </td>
                        <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                            <span style="color: #6b7280;">{old_count} images</span>
                        </td>
                        <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                            <span style="color: #059669;">{new_count} images</span>
                        </td>
                    </tr>
                    """
                    if reason:
                        changes_html += f"""
                        <tr>
                            <td colspan="3" style="padding: 8px 12px; background: #fef3c7; border-bottom: 1px solid #e5e7eb;">
                                <em style="color: #92400e; font-size: 0.875rem;">Reason: {reason}</em>
                            </td>
                        </tr>
                        """
            elif field == 'tech_stack':
                old_stack = ', '.join(old_value) if old_value else 'None'
                new_stack = ', '.join(new_value) if new_value else 'None'
                changes_html += f"""
                <tr>
                    <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                        <strong style="color: #3b82f6;">🔧 {field_name}</strong>
                    </td>
                    <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                        <span style="color: #6b7280; font-size: 0.875rem;">{old_stack}</span>
                    </td>
                    <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                        <span style="color: #059669; font-size: 0.875rem;">{new_stack}</span>
                    </td>
                </tr>
                """
            elif field == 'budget':
                changes_html += f"""
                <tr>
                    <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                        <strong style="color: #10b981;">💰 {field_name}</strong>
                    </td>
                    <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                        <span style="color: #6b7280;">${float(old_value or 0):,.2f}</span>
                    </td>
                    <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                        <span style="color: #059669;">${float(new_value or 0):,.2f}</span>
                    </td>
                </tr>
                """
            else:
                # Generic field
                changes_html += f"""
                <tr>
                    <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                        <strong style="color: #6366f1;">📝 {field_name}</strong>
                    </td>
                    <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                        <span style="color: #6b7280;">{old_value or 'N/A'}</span>
                    </td>
                    <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                        <span style="color: #059669;">{new_value or 'N/A'}</span>
                    </td>
                </tr>
                """
                if reason:
                    changes_html += f"""
                    <tr>
                        <td colspan="3" style="padding: 8px 12px; background: #fef3c7; border-bottom: 1px solid #e5e7eb;">
                            <em style="color: #92400e; font-size: 0.875rem;">Reason: {reason}</em>
                        </td>
                    </tr>
                    """
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <title>Project Updated by Admin</title>
        </head>
        
        <body style="margin:0; padding:0; background:#f8fafc; font-family:'Segoe UI', Arial, sans-serif;">
            <table width="100%" border="0" cellspacing="0" cellpadding="0" style="background:#f8fafc; padding:40px 0;">
                <tr>
                    <td align="center">
                        <table class="container" width="600" border="0" cellspacing="0" cellpadding="0"
                               style="background:white; border-radius:16px; box-shadow:0 4px 20px rgba(0,0,0,0.06); padding:40px;">
                            
                            <!-- Header -->
                            <tr>
                                <td align="center" style="padding-bottom: 30px; border-bottom: 2px solid #f3f4f6;">
                                    <h1 style="margin:0; font-size:28px; font-weight:700; color:#4f46e5;">
                                        BiteBids
                                    </h1>
                                    <p style="margin:8px 0 0; color:#6b7280; font-size:14px;">
                                        Project Update Notification
                                    </p>
                                </td>
                            </tr>
                            
                            <!-- Alert Badge -->
                            <tr>
                                <td align="center" style="padding: 30px 0 20px 0;">
                                    <div style="display: inline-block; background: #fbbf24; color: white; 
                                                padding: 12px 24px; border-radius: 8px; font-size: 16px; font-weight: 600;">
                                        ⚠️ Admin Updated Your Project
                                    </div>
                                </td>
                            </tr>
                            
                            <!-- Greeting -->
                            <tr>
                                <td style="padding-bottom: 20px;">
                                    <p style="margin:0; font-size:16px; color:#111827;">
                                        Hi <strong>{developer_name}</strong>,
                                    </p>
                                </td>
                            </tr>
                            
                            <!-- Main Message -->
                            <tr>
                                <td style="padding-bottom: 20px;">
                                    <p style="margin:0; font-size:15px; color:#374151; line-height:1.6;">
                                        A BiteBids administrator (<strong>{admin_name}</strong>) has made changes to your project 
                                        <strong>"{project_title}"</strong>. Please review the changes below:
                                    </p>
                                </td>
                            </tr>
                            
                            <!-- Changes Table -->
                            <tr>
                                <td style="padding: 20px 0;">
                                    <table width="100%" border="0" cellspacing="0" cellpadding="0" 
                                           style="border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden;">
                                        <thead>
                                            <tr style="background: #f3f4f6;">
                                                <th style="padding: 12px; text-align: left; font-size: 14px; color: #6b7280; font-weight: 600;">
                                                    Field
                                                </th>
                                                <th style="padding: 12px; text-align: left; font-size: 14px; color: #6b7280; font-weight: 600;">
                                                    Previous Value
                                                </th>
                                                <th style="padding: 12px; text-align: left; font-size: 14px; color: #6b7280; font-weight: 600;">
                                                    New Value
                                                </th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {changes_html}
                                        </tbody>
                                    </table>
                                </td>
                            </tr>
                            
                            <!-- Important Notice -->
                            <tr>
                                <td style="padding: 20px 0;">
                                    <table width="100%" border="0" cellspacing="0" cellpadding="0" 
                                           style="background: #fef3c7; border-left: 4px solid #f59e0b; border-radius: 8px; padding: 16px;">
                                        <tr>
                                            <td>
                                                <p style="margin: 0 0 8px 0; font-size: 14px; color: #92400e; font-weight: 600;">
                                                    ℹ️ What This Means:
                                                </p>
                                                <p style="margin: 0; font-size: 13px; color: #78350f; line-height: 1.5;">
                                                    These changes were made by our admin team to ensure compliance with BiteBids platform 
                                                    policies. If you believe this was done in error, please contact support.
                                                </p>
                                            </td>
                                        </tr>
                                    </table>
                                </td>
                            </tr>
                            
                            <!-- Project Link -->
                            <tr>
                                <td style="padding: 20px 0;">
                                    <p style="margin: 0 0 10px 0; font-size: 13px; color: #9ca3af;">
                                        Project ID: <code style="background: #f3f4f6; padding: 4px 8px; border-radius: 4px; color: #6b7280;">{project_id}</code>
                                    </p>
                                </td>
                            </tr>
                            
                            <!-- Action Button -->
                            <tr>
                                <td align="center" style="padding: 20px 0 30px 0;">
                                    <a href="{FRONTEND_URL}/dashboard" 
                                       style="display:inline-block; background:#4f46e5; color:white; 
                                              padding:16px 40px; border-radius:8px; text-decoration:none; 
                                              font-weight:600; font-size:16px; box-shadow:0 4px 12px rgba(0,0,0,0.15);">
                                        View Your Dashboard →
                                    </a>
                                </td>
                            </tr>
                            
                            <!-- Footer -->
                            <tr>
                                <td align="center" style="padding-top: 30px; border-top: 2px solid #f3f4f6;">
                                    <p style="margin:0 0 10px 0; color:#9ca3af; font-size:13px;">
                                        Questions? Contact us at 
                                        <a href="mailto:bitebids@gmail.com" style="color:#4f46e5; text-decoration:none;">
                                            bitebids@gmail.com
                                        </a>
                                    </p>
                                    <p style="margin:0; color:#9ca3af; font-size:13px;">
                                        This is an automated notification from BiteBids<br/>
                                        © 2024 BiteBids. All rights reserved.
                                    </p>
                                </td>
                            </tr>
                            
                        </table>
                    </td>
                </tr>
            </table>
        </body>
        </html>
        """
        
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM
        msg["To"] = developer_email
        
        msg.attach(MIMEText(html_content, "html"))
        
        # Send email
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(EMAIL_FROM, developer_email, msg.as_string())
        server.quit()
        
        logger.info(f"📧 Developer edit notification sent to {developer_email} for project: {project_title}")
        
    except Exception as e:
        logger.error(f"❌ Failed to send developer edit notification: {str(e)}")


# Helper function to send notification to a user (both DB and WebSocket)
async def send_notification_to_user(user_id: str, notification_data: dict, db: AsyncSession):
    """Helper function to send notification to a user (both DB and WebSocket)"""
    try:
        # Save notification to database
        notification = Notification(
            user_id=uuid.UUID(user_id),
            type=notification_data.get("type", "general"),
            title=notification_data.get("title"),
            message=notification_data.get("message"),
            link=notification_data.get("link"),
            details=notification_data.get("details")
        )
        
        db.add(notification)
        await db.commit()
        await db.refresh(notification)
        
        # Send via WebSocket if user is connected
        notif_dict = model_to_dict(notification)
        await manager.send_personal_notification(user_id, notif_dict)
        
        return notif_dict
        
    except Exception as e:
        logger.error(f"Error sending notification: {e}")
        return None


async def get_total_unread_chat_count(db: AsyncSession, user_id: uuid.UUID) -> int:
    """Return total unread chat messages for a user."""
    result = await db.execute(
        select(func.count(ChatMessage.id))
        .select_from(ChatMessage)
        .join(ChatRoom, ChatMessage.room_id == ChatRoom.id)
        .where(
            and_(
                ChatMessage.sender_id != user_id,
                ChatMessage.read == False,
                or_(
                    ChatRoom.developer_id == user_id,
                    ChatRoom.investor_id == user_id
                )
            )
        )
    )
    return result.scalar() or 0


# ============================================
# 2CHECKOUT INTEGRATION CLASSES
# ============================================

class TwoCheckoutAuth:
    def __init__(self, merchant_code: str, secret_key: str, environment: str = "sandbox"):
        self.merchant_code = merchant_code
        self.secret_key = secret_key
        self.environment = environment
        # Use different base URLs for sandbox vs production
        if environment == "sandbox":
            self.base_url = "https://api.sandbox.2checkout.com"
        else:
            self.base_url = "https://api.2checkout.com"
        self.logger = logging.getLogger(__name__)
        
    def generate_auth_header(self) -> Dict[str, str]:
        """Generate authentication header for 2Checkout API requests"""
        import time
        import uuid
        
        # Generate timestamp in milliseconds
        timestamp = int(time.time() * 1000)
        
        # Generate unique nonce
        nonce = uuid.uuid4().hex
        
        # Create message string: nonce + timestamp + merchant_code
        message = nonce + str(timestamp) + self.merchant_code
        
        # Generate HMAC-SHA256 signature
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Construct authentication header
        auth_header = f'code="{self.merchant_code}" date="{timestamp}" nonce="{nonce}" sig="{signature}"'
        
        return {
            "X-Avangate-Authentication": auth_header,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    def make_authenticated_request(self, method: str, endpoint: str, data: Dict[Any, Any] = None) -> requests.Response:
        """Make authenticated request to 2Checkout API"""
        url = f"{self.base_url}/{endpoint}"
        headers = self.generate_auth_header()
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, params=data, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=data, timeout=30)
            elif method.upper() == "PUT":
                response = requests.put(url, headers=headers, json=data, timeout=30)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            self.logger.info(f"2Checkout API {method} {endpoint}: {response.status_code}")
            return response
            
        except requests.RequestException as e:
            self.logger.error(f"2Checkout API request failed: {str(e)}")
            raise


class PaymentMethod(Enum):
    CREDIT_CARD = "credit_card"
    PAYPAL = "paypal"
    GOOGLE_PAY = "google_pay"
    APPLE_PAY = "apple_pay"


class PaymentProcessor:
    def __init__(self, auth: TwoCheckoutAuth):
        self.auth = auth
        self.marketplace_fee_percentage = Decimal('0.06')  # 6%
        self.marketplace_fixed_fee = Decimal('30.00')     # $30
        self.logger = logging.getLogger(__name__)
    
    def calculate_fees(self, developer_price: Decimal) -> Dict[str, Decimal]:
        """Calculate marketplace fees for BiteBids transactions"""
        percentage_fee = developer_price * self.marketplace_fee_percentage
        total_marketplace_fee = percentage_fee + self.marketplace_fixed_fee
        customer_total = developer_price + total_marketplace_fee
        developer_payout = developer_price - total_marketplace_fee
        
        return {
            'developer_price': developer_price,
            'percentage_fee': percentage_fee,
            'fixed_fee': self.marketplace_fixed_fee,
            'total_marketplace_fee': total_marketplace_fee,
            'customer_total': customer_total,
            'developer_payout': developer_payout
        }
    
    def create_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create order in 2Checkout system"""
        try:
            print(f"Creating 2Checkout order with data: {order_data}")
            response = self.auth.make_authenticated_request(
                "POST", 
                "rest/6.0/orders/", 
                order_data
            )
            
            if response.status_code == 200:
                result = response.json()
                self.logger.info(f"Order created successfully: {result.get('RefNo')}")
                print(f"2Checkout API Response: {result}")
                return result
            else:
                self.logger.error(f"Order creation failed: {response.status_code} - {response.text}")
                print(f"2Checkout API Error: {response.status_code} - {response.text}")
                
                # For demo purposes, return a mock response when 2Checkout API fails
                mock_response = {
                    "RefNo": f"DEMO_{uuid.uuid4().hex[:8].upper()}",
                    "Status": "PENDING",
                    "Message": "Demo order created (2Checkout API not available)",
                    "payment_url": None
                }
                self.logger.info("Using demo mode due to API unavailability")
                return mock_response
                
        except Exception as e:
            self.logger.error(f"Order creation error: {str(e)}")
            print(f"Order creation error: {str(e)}")
            # Return mock response for demo
            return {
                "RefNo": f"DEMO_{uuid.uuid4().hex[:8].upper()}",
                "Status": "PENDING", 
                "Message": f"Demo order created (API Error: {str(e)})",
                "payment_url": None
            }
    
    def get_return_url(self) -> str:
        """Get base return URL for payment redirects"""
        return os.getenv("FRONTEND_URL", "http://localhost:3000")


class WebhookHandler:
    def __init__(self, secret_key: str, secret_word: str, merchant_code: str):
        self.secret_key = secret_key
        self.secret_word = secret_word
        self.merchant_code = merchant_code
        self.logger = logging.getLogger(__name__)
    
    def validate_webhook_signature(self, payload: Dict[str, Any]) -> bool:
        """Validate webhook signature from 2Checkout"""
        try:
            received_hash = payload.get('HASH', '')
            
            if 'ORDERNO' in payload or 'IPN_PID' in payload:
                # Build parameter string for signature validation
                params_to_hash = []
                
                # Handle array parameters
                ipn_pid = payload.get('IPN_PID', '')
                if isinstance(ipn_pid, list) and len(ipn_pid) > 0:
                    ipn_pid = ipn_pid[0]
                
                ipn_pname = payload.get('IPN_PNAME', '')
                if isinstance(ipn_pname, list) and len(ipn_pname) > 0:
                    ipn_pname = ipn_pname[0]
                
                ipn_date = payload.get('IPN_DATE', '')
                if isinstance(ipn_date, list) and len(ipn_date) > 0:
                    ipn_date = ipn_date[0]
                
                orderno = payload.get('ORDERNO', '')
                if isinstance(orderno, list) and len(orderno) > 0:
                    orderno = orderno[0]
                
                # Build hash string
                params_to_hash = [
                    str(ipn_pid),
                    str(ipn_pname),
                    str(ipn_date), 
                    str(orderno),
                    str(self.secret_word)
                ]
                
                string_to_hash = ''.join(params_to_hash)
                calculated_hash = hashlib.md5(string_to_hash.encode('utf-8')).hexdigest()
                
                is_valid = (calculated_hash == received_hash)
                if not is_valid:
                    self.logger.warning(f"Webhook signature mismatch. Expected: {calculated_hash}, Received: {received_hash}")
                
                return is_valid
            
            return False
            
        except Exception as e:
            self.logger.error(f"Webhook signature validation error: {str(e)}")
            return False
    
    async def process_payment_notification(self, payload: Dict[str, Any]):
        """Process payment notification from webhook"""
        try:
            order_ref = payload.get('ORDERNO') or payload.get('order_reference')
            payment_status = payload.get('PAYMENT_STATUS', 'unknown')
            
            self.logger.info(f"Processing payment notification for order {order_ref}: {payment_status}")
            
            # Here you would update the order status in your database
            # For example, mark as completed if payment successful
            
        except Exception as e:
            self.logger.error(f"Error processing payment notification: {str(e)}")


# Initialize payment processor if credentials are available
payment_processor = None
webhook_handler = None

if CHECKOUT_MERCHANT_CODE and CHECKOUT_SECRET_KEY:
    auth = TwoCheckoutAuth(
        merchant_code=CHECKOUT_MERCHANT_CODE,
        secret_key=CHECKOUT_SECRET_KEY,
        environment=CHECKOUT_ENVIRONMENT
    )
    payment_processor = PaymentProcessor(auth)
    
if CHECKOUT_SECRET_KEY and CHECKOUT_INS_SECRET_WORD:
    webhook_handler = WebhookHandler(
        secret_key=CHECKOUT_SECRET_KEY,
        secret_word=CHECKOUT_INS_SECRET_WORD,
        merchant_code=CHECKOUT_MERCHANT_CODE
    )





# ============================================
# WEBSOCKET CONNECTION MANAGER
# ============================================

class ConnectionManager:
    """Manages WebSocket connections for live notifications and chat"""
    
    def __init__(self):
        # Store active connections per user
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # Store chat room connections (project_id -> list of WebSockets)
        self.chat_rooms: Dict[str, List[WebSocket]] = {}
        # Map WebSocket to user_id for easy lookup
        self.websocket_to_user: Dict[WebSocket, str] = {}
        
    async def connect(self, websocket: WebSocket, user_id: str):
        """Connect a user's WebSocket"""
        await websocket.accept()
        
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        
        self.active_connections[user_id].append(websocket)
        self.websocket_to_user[websocket] = user_id
        logger.info(f"User {user_id} connected via WebSocket")
        
    def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket"""
        user_id = self.websocket_to_user.get(websocket)
        
        if user_id and user_id in self.active_connections:
            self.active_connections[user_id].remove(websocket)
            
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        
        if websocket in self.websocket_to_user:
            del self.websocket_to_user[websocket]
            
        # Remove from any chat rooms
        for room_id in list(self.chat_rooms.keys()):
            if websocket in self.chat_rooms[room_id]:
                self.chat_rooms[room_id].remove(websocket)
                if not self.chat_rooms[room_id]:
                    del self.chat_rooms[room_id]
        
        logger.info(f"User {user_id} disconnected from WebSocket")
    
    async def join_chat_room(self, websocket: WebSocket, room_id: str):
        """Join a chat room for a project"""
        if room_id not in self.chat_rooms:
            self.chat_rooms[room_id] = []
        
        if websocket not in self.chat_rooms[room_id]:
            self.chat_rooms[room_id].append(websocket)
        
        user_id = self.websocket_to_user.get(websocket)
        logger.info(f"User {user_id} joined chat room {room_id}")
    
    async def leave_chat_room(self, websocket: WebSocket, room_id: str):
        """Leave a chat room"""
        if room_id in self.chat_rooms and websocket in self.chat_rooms[room_id]:
            self.chat_rooms[room_id].remove(websocket)
            
            if not self.chat_rooms[room_id]:
                del self.chat_rooms[room_id]
        
        user_id = self.websocket_to_user.get(websocket)
        logger.info(f"User {user_id} left chat room {room_id}")
    
    def is_user_in_room(self, room_id: str, user_id: str) -> bool:
        """Check if a user has any active WebSocket in a chat room."""
        connections = self.chat_rooms.get(room_id, [])
        for connection in connections:
            if self.websocket_to_user.get(connection) == user_id:
                return True
        return False
    
    async def send_personal_notification(self, user_id: str, notification: dict):
        """Send notification to a specific user"""
        if user_id in self.active_connections:
            disconnected = []
            
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json({
                        "type": "notification",
                        "data": notification
                    })
                except Exception as e:
                    logger.error(f"Error sending notification to user {user_id}: {e}")
                    disconnected.append(connection)
            
            # Clean up disconnected connections
            for conn in disconnected:
                self.disconnect(conn)
    
    async def send_ws_event(self, user_id: str, event: dict):
        """Send a raw WebSocket event to a specific user"""
        if user_id in self.active_connections:
            disconnected = []
            
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json(event)
                except Exception as e:
                    logger.error(f"Error sending ws event to user {user_id}: {e}")
                    disconnected.append(connection)
            
            for conn in disconnected:
                self.disconnect(conn)
    
    async def send_chat_message(self, room_id: str, message: dict):
        """Send message to all users in a chat room"""
        if room_id in self.chat_rooms:
            disconnected = []
            
            for connection in self.chat_rooms[room_id]:
                try:
                    await connection.send_json({
                        "type": "chat_message",
                        "data": message
                    })
                except Exception as e:
                    logger.error(f"Error sending chat message to room {room_id}: {e}")
                    disconnected.append(connection)
            
            # Clean up disconnected connections
            for conn in disconnected:
                self.disconnect(conn)
    
    async def broadcast_to_all(self, message: dict):
        """Broadcast message to all connected users"""
        for user_id, connections in list(self.active_connections.items()):
            disconnected = []
            
            for connection in connections:
                try:
                    await connection.send_json({
                        "type": "broadcast",
                        "data": message
                    })
                except Exception as e:
                    logger.error(f"Error broadcasting to user {user_id}: {e}")
                    disconnected.append(connection)
            
            # Clean up disconnected connections
            for conn in disconnected:
                self.disconnect(conn)

    async def broadcast_to_room(self, room_id: str, message: dict, exclude: WebSocket = None):
        """Broadcast message to all users in a chat room, optionally excluding one WebSocket"""
        if room_id in self.chat_rooms:
            disconnected = []
            
            for connection in self.chat_rooms[room_id]:
                # Skip the excluded connection
                if exclude and connection == exclude:
                    continue
                    
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to room {room_id}: {e}")
                    disconnected.append(connection)
            
            # Clean up disconnected connections
            for conn in disconnected:
                self.disconnect(conn)


# Initialize connection manager
manager = ConnectionManager()

# ============================================
# ADMIN: GET ALL CHAT ROOMS
# ============================================

# ============================================
# FIXED ADMIN CHAT MANAGEMENT ENDPOINTS
# Replace the previous endpoints with these fixed versions
# ============================================
# Add this Pydantic model with your other models
class AdminChatFilter(BaseModel):
    """Filter options for admin chat viewing"""
    search: Optional[str] = None
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    has_unread: Optional[bool] = None

@app.get("/api/admin/chat/rooms")
async def admin_get_all_chat_rooms(
    search: Optional[str] = None,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """
    Admin endpoint to get all chat rooms with filters
    Compatible with any ChatRoom schema
    """
    try:
        # Base query
        query = select(ChatRoom)
        
        # Apply filters
        if project_id:
            query = query.where(ChatRoom.project_id == uuid.UUID(project_id))
        
        if user_id:
            query = query.where(
                or_(
                    ChatRoom.developer_id == uuid.UUID(user_id),
                    ChatRoom.investor_id == uuid.UUID(user_id)
                )
            )
        
        # Try to order by updated_at if it exists, otherwise by created_at
        try:
            query = query.order_by(ChatRoom.updated_at.desc())
        except AttributeError:
            try:
                query = query.order_by(ChatRoom.created_at.desc())
            except AttributeError:
                # If neither exists, order by id
                query = query.order_by(ChatRoom.id.desc())
        
        # Pagination
        query = query.offset(skip).limit(limit)
        
        result = await db.execute(query)
        rooms = result.scalars().all()
        
        # Get additional details for each room
        rooms_data = []
        for room in rooms:
            # Fetch project details separately
            project_query = select(Project).where(Project.id == room.project_id)
            project_result = await db.execute(project_query)
            project = project_result.scalar_one_or_none()
            
            # Apply search filter if provided
            if search:
                search_pattern = search.lower()
                project_title = (project.title if project else "").lower()
                if search_pattern not in project_title:
                    continue  # Skip this room if search doesn't match
            
            # Get message count
            msg_count_query = select(func.count(ChatMessage.id)).where(
                ChatMessage.room_id == room.id
            )
            msg_count_result = await db.execute(msg_count_query)
            message_count = msg_count_result.scalar()
            
            # Get unread counts for both users
            dev_unread_query = select(func.count(ChatMessage.id)).where(
                and_(
                    ChatMessage.room_id == room.id,
                    ChatMessage.sender_id != room.developer_id,
                    ChatMessage.read == False
                )
            )
            dev_unread_result = await db.execute(dev_unread_query)
            developer_unread = dev_unread_result.scalar()
            
            inv_unread_query = select(func.count(ChatMessage.id)).where(
                and_(
                    ChatMessage.room_id == room.id,
                    ChatMessage.sender_id != room.investor_id,
                    ChatMessage.read == False
                )
            )
            inv_unread_result = await db.execute(inv_unread_query)
            investor_unread = inv_unread_result.scalar()
            
            # Get developer details
            dev_query = select(User).where(User.id == room.developer_id)
            dev_result = await db.execute(dev_query)
            developer = dev_result.scalar_one_or_none()
            
            # Get investor details
            inv_query = select(User).where(User.id == room.investor_id)
            inv_result = await db.execute(inv_query)
            investor = inv_result.scalar_one_or_none()
            
            # Get last message to determine "updated_at" time
            last_msg_query = select(ChatMessage).where(
                ChatMessage.room_id == room.id
            ).order_by(ChatMessage.created_at.desc()).limit(1)
            last_msg_result = await db.execute(last_msg_query)
            last_message = last_msg_result.scalar_one_or_none()
            
            # Determine updated_at - use last message time or room created_at
            updated_at = None
            if last_message:
                updated_at = last_message.created_at
            elif hasattr(room, 'updated_at'):
                updated_at = room.updated_at
            elif hasattr(room, 'created_at'):
                updated_at = room.created_at
            else:
                updated_at = datetime.utcnow()
            
            # Determine created_at
            created_at = room.created_at if hasattr(room, 'created_at') else datetime.utcnow()
            
            rooms_data.append({
                "id": str(room.id),
                "project_id": str(room.project_id),
                "project_title": project.title if project else "Unknown Project",
                "developer_id": str(room.developer_id),
                "developer_name": developer.name if developer else "Unknown",
                "developer_email": developer.email if developer else "Unknown",
                "investor_id": str(room.investor_id),
                "investor_name": investor.name if investor else "Unknown",
                "investor_email": investor.email if investor else "Unknown",
                "created_at": created_at.isoformat(),
                "updated_at": updated_at.isoformat(),
                "message_count": message_count,
                "developer_unread_count": developer_unread,
                "investor_unread_count": investor_unread,
                "last_message": {
                    "message": last_message.message if last_message else None,
                    "sender_id": str(last_message.sender_id) if last_message else None,
                    "created_at": last_message.created_at.isoformat() if last_message else None
                } if last_message else None
            })
        
        # Get total count for pagination
        count_query = select(func.count(ChatRoom.id))
        if project_id:
            count_query = count_query.where(ChatRoom.project_id == uuid.UUID(project_id))
        if user_id:
            count_query = count_query.where(
                or_(
                    ChatRoom.developer_id == uuid.UUID(user_id),
                    ChatRoom.investor_id == uuid.UUID(user_id)
                )
            )
        
        count_result = await db.execute(count_query)
        total = count_result.scalar()
        
        # If search is applied, use filtered results count
        if search:
            total = len(rooms_data)
        
        return {
            "rooms": rooms_data,
            "total": total,
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Error fetching admin chat rooms: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/chat/rooms/{room_id}/messages")
async def admin_get_room_messages(
    room_id: str,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """
    Admin endpoint to get all messages in a specific chat room
    """
    try:
        # Verify room exists
        room_query = select(ChatRoom).where(ChatRoom.id == uuid.UUID(room_id))
        room_result = await db.execute(room_query)
        room = room_result.scalar_one_or_none()
        
        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")
        
        # Get messages with sender information
        query = select(ChatMessage).where(
            ChatMessage.room_id == uuid.UUID(room_id)
        ).order_by(ChatMessage.created_at.asc()).offset(skip).limit(limit)
        
        result = await db.execute(query)
        messages = result.scalars().all()
        
        # Get sender details for each message
        messages_data = []
        for msg in messages:
            sender_query = select(User).where(User.id == msg.sender_id)
            sender_result = await db.execute(sender_query)
            sender = sender_result.scalar_one_or_none()
            
            # Handle optional updated_at field
            updated_at = msg.updated_at if hasattr(msg, 'updated_at') else msg.created_at
            
            messages_data.append({
                "id": str(msg.id),
                "room_id": str(msg.room_id),
                "sender_id": str(msg.sender_id),
                "sender_name": sender.name if sender else "Unknown",
                "sender_email": sender.email if sender else "Unknown",
                "sender_role": sender.role if sender else "Unknown",
                "message": msg.message,
                "file_url": msg.file_url if hasattr(msg, 'file_url') else None,
                "file_name": msg.file_name if hasattr(msg, 'file_name') else None,
                "read": msg.read if hasattr(msg, 'read') else False,
                "created_at": msg.created_at.isoformat(),
                "updated_at": updated_at.isoformat(),
                # Moderation fields
                "flagged": getattr(msg, 'flagged', False),
                "moderation_status": getattr(msg, 'moderation_status', 'approved'),
                "moderation_reason": getattr(msg, 'moderation_reason', None)
            })
        
        # Get total message count
        count_query = select(func.count(ChatMessage.id)).where(
            ChatMessage.room_id == uuid.UUID(room_id)
        )
        count_result = await db.execute(count_query)
        total = count_result.scalar()
        
        return {
            "messages": messages_data,
            "total": total,
            "skip": skip,
            "limit": limit
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid room ID format")
    except Exception as e:
        logger.error(f"Error fetching admin room messages: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/chat/statistics")
async def admin_get_chat_statistics(
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """
    Admin endpoint to get overall chat statistics
    """
    try:
        # Total chat rooms
        total_rooms_query = select(func.count(ChatRoom.id))
        total_rooms_result = await db.execute(total_rooms_query)
        total_rooms = total_rooms_result.scalar()
        
        # Total messages
        total_messages_query = select(func.count(ChatMessage.id))
        total_messages_result = await db.execute(total_messages_query)
        total_messages = total_messages_result.scalar()
        
        # Active rooms (rooms with messages in last 7 days)
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        active_rooms_query = select(func.count(func.distinct(ChatMessage.room_id))).where(
            ChatMessage.created_at >= seven_days_ago
        )
        active_rooms_result = await db.execute(active_rooms_query)
        active_rooms = active_rooms_result.scalar()
        
        # Total unread messages (if read field exists)
        try:
            unread_messages_query = select(func.count(ChatMessage.id)).where(
                ChatMessage.read == False
            )
            unread_messages_result = await db.execute(unread_messages_query)
            unread_messages = unread_messages_result.scalar()
        except AttributeError:
            unread_messages = 0
        
        # Messages sent today
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_messages_query = select(func.count(ChatMessage.id)).where(
            ChatMessage.created_at >= today_start
        )
        today_messages_result = await db.execute(today_messages_query)
        today_messages = today_messages_result.scalar()
        
        # Average messages per room
        avg_messages_per_room = total_messages / total_rooms if total_rooms > 0 else 0
        
        return {
            "total_rooms": total_rooms,
            "total_messages": total_messages,
            "active_rooms_last_7_days": active_rooms,
            "unread_messages": unread_messages,
            "messages_today": today_messages,
            "avg_messages_per_room": round(avg_messages_per_room, 2)
        }
        
    except Exception as e:
        logger.error(f"Error fetching chat statistics: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/chat/search-messages")
async def admin_search_messages(
    query: str,
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """
    Admin endpoint to search all messages across all chat rooms
    """
    try:
        search_pattern = f"%{query.lower()}%"
        
        # Search messages
        search_query = select(ChatMessage).where(
            func.lower(ChatMessage.message).like(search_pattern)
        ).order_by(ChatMessage.created_at.desc()).offset(skip).limit(limit)
        
        result = await db.execute(search_query)
        messages = result.scalars().all()
        
        # Get details for each message
        messages_data = []
        for msg in messages:
            # Get room details
            room_query = select(ChatRoom).where(ChatRoom.id == msg.room_id)
            room_result = await db.execute(room_query)
            room = room_result.scalar_one_or_none()
            
            # Get project details separately
            project = None
            if room:
                project_query = select(Project).where(Project.id == room.project_id)
                project_result = await db.execute(project_query)
                project = project_result.scalar_one_or_none()
            
            # Get sender details
            sender_query = select(User).where(User.id == msg.sender_id)
            sender_result = await db.execute(sender_query)
            sender = sender_result.scalar_one_or_none()
            
            messages_data.append({
                "id": str(msg.id),
                "room_id": str(msg.room_id),
                "project_title": project.title if project else "Unknown",
                "sender_id": str(msg.sender_id),
                "sender_name": sender.name if sender else "Unknown",
                "sender_role": sender.role if sender else "Unknown",
                "message": msg.message,
                "created_at": msg.created_at.isoformat()
            })
        
        # Get total count
        count_query = select(func.count(ChatMessage.id)).where(
            func.lower(ChatMessage.message).like(search_pattern)
        )
        count_result = await db.execute(count_query)
        total = count_result.scalar()
        
        return {
            "messages": messages_data,
            "total": total,
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Error searching messages: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# API ROUTES
# ============================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "BiteBids API", "database": "PostgreSQL"}


# ============================================
# Github ROUTES
# ============================================

@app.post("/api/chat/rooms/{room_id}/submit-github-repo")
async def submit_github_repository(
    room_id: str,
    request: dict,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Submit GitHub repository URL for project
    Developer only
    Supports private repositories via access token
    """
    try:
        # ✅ FIX: null-safe request parsing
        repo_url = (request.get("repo_url") or "").strip()
        access_token = (request.get("access_token") or "").strip()

        if not repo_url:
            raise HTTPException(status_code=400, detail="Repository URL is required")

        # Validate GitHub URL format
        try:
            parsed = parse_github_url(repo_url)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid GitHub repository URL")

        # Check if repo is private (token optional at this stage)
        is_private = await check_if_repo_is_private(
            parsed["owner"],
            parsed["repo"],
            access_token
        )

        # If private and no token provided → error
        if is_private and not access_token:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "PRIVATE_REPO_TOKEN_REQUIRED",
                    "message": (
                        "This repository is private. Please provide a GitHub "
                        "Personal Access Token with 'repo' scope."
                    )
                }
            )

        # Verify room exists
        room_query = select(ChatRoom).where(ChatRoom.id == uuid.UUID(room_id))
        room_result = await db.execute(room_query)
        room = room_result.scalar_one_or_none()

        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")

        # Verify developer permissions
        user_id = uuid.UUID(current_user["id"])
        if user_id != room.developer_id:
            raise HTTPException(
                status_code=403,
                detail="Only the developer can submit repository"
            )

        # Encrypt token if provided
        encrypted_token = None
        if access_token:
            try:
                encrypted_token = encrypt_token(access_token)
                logger.info("🔒 Encrypted access token for private repo")
            except Exception as e:
                logger.error(f"Error encrypting token: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail="Failed to encrypt access token"
                )

        # Check if repository already exists for this room
        existing_query = select(ProjectGithubRepo).where(
            ProjectGithubRepo.room_id == uuid.UUID(room_id)
        )
        existing_result = await db.execute(existing_query)
        existing = existing_result.scalar_one_or_none()

        if existing:
            existing.repo_url = repo_url
            existing.is_private = is_private
            existing.encrypted_access_token = encrypted_token
            existing.submitted_at = datetime.utcnow()
        else:
            github_repo = ProjectGithubRepo(
                id=uuid.uuid4(),
                room_id=uuid.UUID(room_id),
                repo_url=repo_url,
                is_private=is_private,
                encrypted_access_token=encrypted_token,
                submitted_by=user_id,
                submitted_at=datetime.utcnow()
            )
            db.add(github_repo)

        await db.commit()

        logger.info(
            f"✅ GitHub repository submitted for room {room_id} "
            f"(private: {is_private})"
        )

        return {
            "success": True,
            "message": "Repository submitted successfully",
            "repo_url": repo_url,
            "is_private": is_private
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error submitting repository", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


@app.get("/api/chat/rooms/{room_id}/github-repo")
async def get_github_repository(
    room_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get GitHub repository for a room
    ✅ Now supports admin access
    """
    try:
        # Verify room exists
        room_query = select(ChatRoom).where(ChatRoom.id == uuid.UUID(room_id))
        room_result = await db.execute(room_query)
        room = room_result.scalar_one_or_none()
        
        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")
        
        user_id = uuid.UUID(current_user["id"])
        user_role = current_user.get("role", "")
        
        # ✅ NEW: Allow admins to access any room
        if user_role != "admin":
            # Regular users must be participants
            if user_id not in [room.developer_id, room.investor_id]:
                raise HTTPException(status_code=403, detail="Access denied")
        
        # Get repository
        repo_query = select(ProjectGithubRepo).where(
            ProjectGithubRepo.room_id == uuid.UUID(room_id)
        )
        repo_result = await db.execute(repo_query)
        repo = repo_result.scalar_one_or_none()
        
        if not repo:
            return {
                "exists": False,
                "message": "No repository submitted yet"
            }
        
        logger.info(f"✅ GitHub repo fetched for room {room_id} by {'admin' if user_role == 'admin' else 'user'} {user_id}")
        
        return {
            "exists": True,
            "repo_url": repo.repo_url,
            "submitted_at": repo.submitted_at.isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching repository: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/github/repo-structure")
async def get_repo_structure(
    request: dict,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get GitHub repository structure
    Supports private repositories
    ✅ Now supports admin access
    """
    try:
        repo_url = request.get('repo_url')
        
        if not repo_url:
            raise HTTPException(status_code=400, detail="Repository URL is required")
        
        user_role = current_user.get("role", "")
        user_id = uuid.UUID(current_user["id"])
        
        # Parse GitHub URL
        try:
            parsed = parse_github_url(repo_url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Check if we have stored token for this repo
        repo_query = select(ProjectGithubRepo).where(
            ProjectGithubRepo.repo_url == repo_url
        )
        repo_result = await db.execute(repo_query)
        repo_record = repo_result.scalar_one_or_none()
        
        # ✅ NEW: Admins can access any repo
        if repo_record and user_role != "admin":
            # Verify non-admin users have access to this room
            room_query = select(ChatRoom).where(ChatRoom.id == repo_record.room_id)
            room_result = await db.execute(room_query)
            room = room_result.scalar_one_or_none()
            
            if room and user_id not in [room.developer_id, room.investor_id]:
                raise HTTPException(status_code=403, detail="Access denied")
        
        encrypted_token = repo_record.encrypted_access_token if repo_record else None
        
        # Fetch tree (will use token if available)
        tree = fetch_github_tree(parsed['owner'], parsed['repo'], encrypted_token=encrypted_token)
        
        if tree is None:
            raise HTTPException(status_code=404, detail="Could not fetch repository structure. Repository may be private or rate limit exceeded.")
        
        logger.info(f"✅ Repo structure fetched for {repo_url} by {'admin' if user_role == 'admin' else 'user'} {user_id}")
        
        return {
            "success": True,
            "owner": parsed['owner'],
            "repo": parsed['repo'],
            "tree": tree
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching repo structure: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/github/file-content")
async def get_file_content(
    request: dict,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get file content from GitHub repository
    Supports private repositories
    ✅ Now supports admin access
    """
    try:
        repo_url = request.get('repo_url')
        file_path = request.get('file_path')
        
        if not repo_url or not file_path:
            raise HTTPException(status_code=400, detail="Repository URL and file path are required")
        
        user_role = current_user.get("role", "")
        user_id = uuid.UUID(current_user["id"])
        
        # Parse GitHub URL
        try:
            parsed = parse_github_url(repo_url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Check if we have stored token for this repo
        repo_query = select(ProjectGithubRepo).where(
            ProjectGithubRepo.repo_url == repo_url
        )
        repo_result = await db.execute(repo_query)
        repo_record = repo_result.scalar_one_or_none()
        
        # ✅ NEW: Admins can access any file
        if repo_record and user_role != "admin":
            # Verify non-admin users have access to this room
            room_query = select(ChatRoom).where(ChatRoom.id == repo_record.room_id)
            room_result = await db.execute(room_query)
            room = room_result.scalar_one_or_none()
            
            if room and user_id not in [room.developer_id, room.investor_id]:
                raise HTTPException(status_code=403, detail="Access denied")
        
        encrypted_token = repo_record.encrypted_access_token if repo_record else None
        
        # Fetch file content (will use token if available)
        content = fetch_github_file_content(
            parsed['owner'], 
            parsed['repo'], 
            file_path,
            encrypted_token=encrypted_token
        )
        
        logger.info(f"✅ File content fetched: {file_path} from {repo_url} by {'admin' if user_role == 'admin' else 'user'} {user_id}")
        
        return {
            "success": True,
            "content": content,
            "file_path": file_path
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching file content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/github/download-repo/{room_id}")
async def download_github_repo(
    room_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Download GitHub repository as ZIP file
    - Only accessible after project is confirmed (completed status)
    - Supports private repositories using stored access tokens
    - Admin can access any repository
    """
    try:
        user_role = current_user.get("role", "")
        user_id = uuid.UUID(current_user["id"])
        room_uuid = uuid.UUID(room_id)

        # 1. Get the chat room
        room_query = select(ChatRoom).where(ChatRoom.id == room_uuid)
        room_result = await db.execute(room_query)
        room = room_result.scalar_one_or_none()

        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")

        # 2. Authorization check (admin bypass)
        if user_role != "admin":
            if user_id not in [room.developer_id, room.investor_id]:
                raise HTTPException(status_code=403, detail="Access denied to this chat room")

        # 3. Get the project and check status
        project_query = select(Project).where(Project.id == room.project_id)
        project_result = await db.execute(project_query)
        project = project_result.scalar_one_or_none()

        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # 4. Check if project is completed (investor confirmed)
        # Allow download for completed, fixed_price (after at least one confirmation), or admin
        is_investor = user_id == room.investor_id

        if user_role != "admin":
            # Check if there's a completed payout for this investor
            payout_query = select(DeveloperPayout).where(
                DeveloperPayout.project_id == project.id,
                DeveloperPayout.investor_id == user_id
            )
            payout_result = await db.execute(payout_query)
            payout = payout_result.scalar_one_or_none()

            if is_investor and not payout:
                raise HTTPException(
                    status_code=403,
                    detail="You must confirm the project before downloading. Please click 'Confirm' first."
                )

        # 5. Get GitHub repo for this room
        repo_query = select(ProjectGithubRepo).where(ProjectGithubRepo.room_id == room_uuid)
        repo_result = await db.execute(repo_query)
        repo_record = repo_result.scalar_one_or_none()

        if not repo_record or not repo_record.repo_url:
            raise HTTPException(status_code=404, detail="No GitHub repository found for this project")

        # 6. Parse GitHub URL
        try:
            parsed = parse_github_url(repo_record.repo_url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid GitHub URL: {str(e)}")

        owner = parsed['owner']
        repo = parsed['repo']

        # 7. Prepare headers for GitHub API
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "BiteBids-Platform"
        }

        # Add auth token if available (for private repos)
        if repo_record.encrypted_access_token:
            try:
                token = decrypt_token(repo_record.encrypted_access_token)
                headers["Authorization"] = f"Bearer {token}"
            except Exception as e:
                logger.error(f"Failed to decrypt token: {e}")

        # 8. Get default branch first
        branch = "main"
        try:
            repo_info_url = f"https://api.github.com/repos/{owner}/{repo}"
            repo_response = requests.get(repo_info_url, headers=headers, timeout=10)
            if repo_response.status_code == 200:
                repo_info = repo_response.json()
                branch = repo_info.get("default_branch", "main")
        except Exception as e:
            logger.warning(f"Could not get default branch, using 'main': {e}")

        # 9. Download ZIP from GitHub (using codeload for faster direct download)
        # codeload.github.com is GitHub's CDN - typically 2-3x faster than api.github.com
        zip_url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"

        logger.info(f"📦 Downloading repo ZIP: {owner}/{repo} branch:{branch} for user:{user_id}")

        # Use longer timeout for large repos (3 minutes)
        response = requests.get(zip_url, headers=headers, stream=True, timeout=180)

        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="Repository not found or access denied")
        elif response.status_code == 401:
            raise HTTPException(status_code=401, detail="Authentication required for this private repository")
        elif response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"GitHub API error: {response.text}")

        # 10. Get filename from Content-Disposition header or generate one
        content_disposition = response.headers.get('Content-Disposition', '')
        if 'filename=' in content_disposition:
            filename = content_disposition.split('filename=')[-1].strip('"')
        else:
            filename = f"{repo}-{branch}.zip"

        # 11. Stream the response back
        logger.info(f"✅ Serving download: {filename} for user {user_id}")

        # Build headers - only include Content-Length if available and valid
        response_headers = {
            "Content-Disposition": f'attachment; filename="{filename}"',
        }

        content_length = response.headers.get('Content-Length')
        if content_length and content_length.strip():
            response_headers["Content-Length"] = content_length

        return StreamingResponse(
            iter(response.iter_content(chunk_size=65536)),  # 64KB chunks for faster transfer
            media_type="application/zip",
            headers=response_headers
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading repository: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download repository: {str(e)}")


# ============================================
# PROJECT UPLOAD (R2 CLOUD STORAGE) ENDPOINTS
# ============================================

@app.post("/api/upload/presigned-url/{room_id}")
async def get_upload_presigned_url(
    room_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate a presigned URL for direct upload to R2
    - Only developers can upload
    - Returns presigned URL valid for 1 hour
    """
    try:
        if not r2_client:
            raise HTTPException(status_code=503, detail="Cloud storage not configured")

        user_id = uuid.UUID(current_user["id"])
        room_uuid = uuid.UUID(room_id)

        # Get chat room
        room_query = select(ChatRoom).where(ChatRoom.id == room_uuid)
        room_result = await db.execute(room_query)
        room = room_result.scalar_one_or_none()

        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")

        # Only developer can upload
        if user_id != room.developer_id:
            raise HTTPException(status_code=403, detail="Only the developer can upload project files")

        # Check if there's already a GitHub repo or upload for this room
        github_query = select(ProjectGithubRepo).where(ProjectGithubRepo.room_id == room_uuid)
        github_result = await db.execute(github_query)
        existing_github = github_result.scalar_one_or_none()

        if existing_github:
            raise HTTPException(status_code=400, detail="A GitHub repository has already been submitted for this project")

        # Get request body
        body = await request.json()
        file_name = body.get("file_name", "project.zip")
        file_size = body.get("file_size", 0)
        content_type = body.get("content_type", "application/zip")

        # Validate file size (max 5GB)
        max_size = 5 * 1024 * 1024 * 1024  # 5GB
        if file_size > max_size:
            raise HTTPException(status_code=400, detail="File size exceeds 5GB limit")

        # Generate unique file key
        file_key = f"projects/{room_id}/{uuid.uuid4()}/{file_name}"

        # Generate presigned URL for upload (valid for 1 hour)
        presigned_url = r2_client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': R2_BUCKET_NAME,
                'Key': file_key,
                'ContentType': content_type,
            },
            ExpiresIn=3600  # 1 hour
        )

        logger.info(f"📤 Generated upload URL for room {room_id} by user {user_id}")

        return {
            "presigned_url": presigned_url,
            "file_key": file_key,
            "expires_in": 3600
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating upload URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload/complete/{room_id}")
async def complete_upload(
    room_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Complete the upload process by saving file tree metadata
    - Called after file is uploaded to R2
    - Saves file tree structure for investor preview
    """
    try:
        user_id = uuid.UUID(current_user["id"])
        room_uuid = uuid.UUID(room_id)

        # Get chat room
        room_query = select(ChatRoom).where(ChatRoom.id == room_uuid)
        room_result = await db.execute(room_query)
        room = room_result.scalar_one_or_none()

        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")

        # Only developer can complete upload
        if user_id != room.developer_id:
            raise HTTPException(status_code=403, detail="Only the developer can upload project files")

        body = await request.json()
        file_key = body.get("file_key")
        file_name = body.get("file_name")
        file_size = body.get("file_size")
        file_tree = body.get("file_tree")

        if not all([file_key, file_name, file_tree]):
            raise HTTPException(status_code=400, detail="Missing required fields")

        # Check if upload already exists for this room
        existing_query = select(ProjectUpload).where(ProjectUpload.room_id == room_uuid)
        existing_result = await db.execute(existing_query)
        existing = existing_result.scalar_one_or_none()

        if existing:
            # Update existing upload
            existing.file_key = file_key
            existing.file_name = file_name
            existing.file_size = file_size
            existing.file_tree = file_tree
            existing.uploaded_at = datetime.utcnow()
        else:
            # Create new upload record
            upload = ProjectUpload(
                room_id=room_uuid,
                file_key=file_key,
                file_name=file_name,
                file_size=file_size,
                file_tree=file_tree,
                uploaded_by=user_id,
                status='pending'
            )
            db.add(upload)

        await db.commit()

        logger.info(f"✅ Upload completed for room {room_id} by user {user_id}")

        return {
            "success": True,
            "message": "Project uploaded successfully",
            "file_name": file_name,
            "file_size": file_size
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing upload: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/upload/info/{room_id}")
async def get_upload_info(
    room_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get upload information including file tree for preview
    - Both developer and investor can view
    """
    try:
        user_id = uuid.UUID(current_user["id"])
        user_role = current_user.get("role", "")
        room_uuid = uuid.UUID(room_id)

        # Get chat room
        room_query = select(ChatRoom).where(ChatRoom.id == room_uuid)
        room_result = await db.execute(room_query)
        room = room_result.scalar_one_or_none()

        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")

        # Authorization check
        if user_role != "admin" and user_id not in [room.developer_id, room.investor_id]:
            raise HTTPException(status_code=403, detail="Access denied")

        # Get upload record
        upload_query = select(ProjectUpload).where(ProjectUpload.room_id == room_uuid)
        upload_result = await db.execute(upload_query)
        upload = upload_result.scalar_one_or_none()

        if not upload:
            return {"upload": None}

        return {
            "upload": {
                "id": str(upload.id),
                "file_name": upload.file_name,
                "file_size": upload.file_size,
                "file_tree": upload.file_tree,
                "uploaded_at": upload.uploaded_at.isoformat() if upload.uploaded_at else None,
                "status": upload.status
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting upload info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/upload/download-url/{room_id}")
async def get_download_presigned_url(
    room_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate a presigned URL for downloading from R2
    - Only accessible after project is confirmed
    """
    try:
        if not r2_client:
            raise HTTPException(status_code=503, detail="Cloud storage not configured")

        user_id = uuid.UUID(current_user["id"])
        user_role = current_user.get("role", "")
        room_uuid = uuid.UUID(room_id)

        # Get chat room
        room_query = select(ChatRoom).where(ChatRoom.id == room_uuid)
        room_result = await db.execute(room_query)
        room = room_result.scalar_one_or_none()

        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")

        # Authorization check
        if user_role != "admin" and user_id not in [room.developer_id, room.investor_id]:
            raise HTTPException(status_code=403, detail="Access denied")

        # Check if project is confirmed (for investor)
        is_investor = user_id == room.investor_id

        if user_role != "admin" and is_investor:
            # Check for completed payout
            payout_query = select(DeveloperPayout).where(
                DeveloperPayout.project_id == room.project_id,
                DeveloperPayout.investor_id == user_id
            )
            payout_result = await db.execute(payout_query)
            payout = payout_result.scalar_one_or_none()

            if not payout:
                raise HTTPException(
                    status_code=403,
                    detail="You must confirm the project before downloading"
                )

        # Get upload record
        upload_query = select(ProjectUpload).where(ProjectUpload.room_id == room_uuid)
        upload_result = await db.execute(upload_query)
        upload = upload_result.scalar_one_or_none()

        if not upload:
            raise HTTPException(status_code=404, detail="No uploaded project found")

        # Generate presigned download URL (valid for 1 hour)
        presigned_url = r2_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': R2_BUCKET_NAME,
                'Key': upload.file_key,
            },
            ExpiresIn=3600  # 1 hour
        )

        # Update status
        upload.status = 'downloaded'
        await db.commit()

        logger.info(f"📥 Generated download URL for room {room_id} by user {user_id}")

        return {
            "download_url": presigned_url,
            "file_name": upload.file_name,
            "file_size": upload.file_size,
            "expires_in": 3600
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating download URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# WEBSOCKET ENDPOINTS
# ============================================

@app.websocket("/ws/notifications/{user_id}")
async def websocket_notifications(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for live notifications"""
    await manager.connect(websocket, user_id)
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "user_id": user_id
        })
        
        while True:
            # Keep connection alive and listen for messages
            data = await websocket.receive_json()
            
            # Handle different message types
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            
            elif data.get("type") == "mark_read":
                # Handle marking notification as read
                notif_id = data.get("notification_id")
                # Update notification in database (handled by existing endpoint)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        manager.disconnect(websocket)


@app.websocket("/ws/chat/{room_id}/{user_id}")
async def websocket_chat(websocket: WebSocket, room_id: str, user_id: str):
    """WebSocket endpoint for chat rooms"""
    await manager.connect(websocket, user_id)
    
    try:
        user_uuid = uuid.UUID(user_id)
        room_uuid = uuid.UUID(room_id)
    except ValueError:
        await websocket.close(code=1008)
        manager.disconnect(websocket)
        return
    
    async with async_session_maker() as db:
        room_result = await db.execute(
            select(ChatRoom).where(ChatRoom.id == room_uuid)
        )
        room = room_result.scalar_one_or_none()
        
        if not room or user_uuid not in [room.developer_id, room.investor_id]:
            await websocket.close(code=1008)
            manager.disconnect(websocket)
            return
        
        await db.execute(
            update(ChatMessage)
            .where(
                and_(
                    ChatMessage.room_id == room_uuid,
                    ChatMessage.sender_id != user_uuid,
                    ChatMessage.read == False
                )
            )
            .values(read=True, read_at=func.now())
        )
        await db.commit()
        total_unread = await get_total_unread_chat_count(db, user_uuid)
        await manager.send_ws_event(
            user_id,
            {"type": "chat_unread_count", "total_unread_count": total_unread}
        )
    
    await manager.join_chat_room(websocket, room_id)
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "room_id": room_id,
            "user_id": user_id
        })
        
        # Send notification to room about user joining
        await manager.broadcast_to_room(room_id, {
            "type": "user_joined",
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, exclude=websocket)
        
        while True:
            # Listen for incoming messages
            data = await websocket.receive_json()
            
            if data.get("type") == "message":
                message_text = data.get("message", "").strip()
                
                # ✅ CONTENT FILTERING
                filter_result = ContentFilter.check_message(message_text)
                
                if not filter_result['is_safe']:
                    # Send error back to user
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Message blocked: Contains prohibited content",
                        "violations": [v['type'] for v in filter_result['violations']],
                        "detail": "Please do not share phone numbers, emails, or social media. All communication must happen through BiteBids."
                    }))
                    
                    # Log violation
                    async with async_session_maker() as db:
                        log_entry = ContentFilterLog(
                            chat_room_id=uuid.UUID(room_id),
                            user_id=uuid.UUID(user_id),
                            original_message=message_text,
                            filtered_content=[v['type'] for v in filter_result['violations']],
                            action_taken='blocked'
                        )
                        db.add(log_entry)
                        await db.commit()
                    
                    return  # Don't process message further
                
                # Message is safe - handle incoming chat message
                # Note: Messages should be sent via REST API for proper persistence
                # This is kept for backward compatibility
                message_data = {
                    "id": str(uuid.uuid4()),
                    "room_id": room_id,
                    "sender_id": user_id,
                    "message": message_text,
                    "message_type": data.get("message_type", "text"),
                    "read": False,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                # Broadcast to room
                await manager.send_chat_message(room_id, message_data)
            
            elif data.get("type") == "typing":
                # Broadcast typing indicator to others (not to sender)
                await manager.broadcast_to_room(room_id, {
                    "type": "typing",
                    "user_id": user_id,
                    "is_typing": data.get("is_typing", True)
                }, exclude=websocket)
            
            elif data.get("type") == "message_read":
                # ✅ NEW: Handle message read notification
                message_id = data.get("message_id")
                if message_id:
                    try:
                        message_uuid = uuid.UUID(message_id)
                        user_uuid = uuid.UUID(user_id)
                        async with async_session_maker() as db:
                            await db.execute(
                                update(ChatMessage)
                                .where(
                                    and_(
                                        ChatMessage.id == message_uuid,
                                        ChatMessage.room_id == uuid.UUID(room_id),
                                        ChatMessage.sender_id != user_uuid,
                                        ChatMessage.read == False
                                    )
                                )
                                .values(read=True, read_at=func.now())
                            )
                            await db.commit()
                            total_unread = await get_total_unread_chat_count(db, user_uuid)
                            await manager.send_ws_event(
                                user_id,
                                {"type": "chat_unread_count", "total_unread_count": total_unread}
                            )
                    except ValueError:
                        pass
                    except Exception as e:
                        logger.error(f"Error updating message read status: {e}")
                    
                    # Broadcast read notification to others
                    await manager.broadcast_to_room(room_id, {
                        "type": "message_read",
                        "message_id": message_id,
                        "user_id": user_id,
                        "read_at": datetime.now(timezone.utc).isoformat()
                    }, exclude=websocket)
            
            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        # Notify room about user leaving
        await manager.broadcast_to_room(room_id, {
            "type": "user_left",
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, exclude=websocket)
        await manager.leave_chat_room(websocket, room_id)
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket chat error for room {room_id}, user {user_id}: {e}")
        await manager.leave_chat_room(websocket, room_id)
        manager.disconnect(websocket)


# ============================================
# CHAT REST API ENDPOINTS
# ============================================

# Get all chat rooms for current user
@app.get("/api/chat/rooms")
async def get_user_chat_rooms(
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get all chat rooms for current user"""
    try:
        user_id = uuid.UUID(current_user["id"])
        
        result = await db.execute(
            select(ChatRoom)
            .where(
                or_(
                    ChatRoom.developer_id == user_id,
                    ChatRoom.investor_id == user_id
                )
            )
            .order_by(ChatRoom.created_at.desc())
        )
        rooms = result.scalars().all()
        
        return models_to_list(rooms)
        
    except Exception as e:
        logger.error(f"Error getting user chat rooms: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/chat/rooms/create/{project_id}")
async def create_chat_room(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Create a chat room for a project after payment is completed"""
    try:
        # Verify project exists
        project_result = await db.execute(
            select(Project).where(Project.id == uuid.UUID(project_id))
        )
        project = project_result.scalar_one_or_none()
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Check if payment is completed for this project by this investor
        payment_result = await db.execute(
            select(CheckoutSession)
            .where(
                and_(
                    CheckoutSession.project_id == uuid.UUID(project_id),
                    CheckoutSession.customer_id == uuid.UUID(current_user['id']),
                    CheckoutSession.status == "completed"
                )
            )
        )
        payment = payment_result.scalar_one_or_none()
        
        if not payment:
            raise HTTPException(
                status_code=403, 
                detail="Payment must be completed before creating chat room"
            )
        
        # Get investor_id from payment
        investor_id = payment.customer_id
        
        # ✅ UPDATED: Check if chat room already exists for THIS developer-investor pair
        # For fixed_price projects, multiple chat rooms can exist (one per investor)
        room_result = await db.execute(
            select(ChatRoom).where(
                and_(
                    ChatRoom.project_id == uuid.UUID(project_id),
                    ChatRoom.investor_id == investor_id
                )
            )
        )
        existing_room = room_result.scalar_one_or_none()
        
        if existing_room:
            return {
                "room_id": str(existing_room.id),
                "message": "Chat room already exists for this investor"
            }
        
        # Create new chat room for this developer-investor pair
        chat_room = ChatRoom(
            project_id=uuid.UUID(project_id),
            developer_id=project.developer_id,
            investor_id=investor_id,
            status="active"
        )
        
        db.add(chat_room)
        await db.commit()
        await db.refresh(chat_room)
        
        # Create system message
        system_message = ChatMessage(
            room_id=chat_room.id,
            sender_id=project.developer_id,
            message="Chat room created. Payment completed successfully. You can now communicate about the project.",
            message_type="system"
        )
        
        db.add(system_message)
        await db.commit()
        
        # Send notifications to both parties
        await send_notification_to_user(
            str(project.developer_id),
            {
                "type": "chat_room_created",
                "title": "Chat Room Created",
                "message": f"Chat room for project '{project.title}' is now active",
                "link": f"/chat/{chat_room.id}",
                "details": {
                    "project_id": project_id,
                    "room_id": str(chat_room.id)
                }
            },
            db
        )
        
        await send_notification_to_user(
            str(investor_id),
            {
                "type": "chat_room_created",
                "title": "Chat Room Created",
                "message": f"Chat room for project '{project.title}' is now active",
                "link": f"/chat/{chat_room.id}",
                "details": {
                    "project_id": project_id,
                    "room_id": str(chat_room.id)
                }
            },
            db
        )
        
        return {
            "room_id": str(chat_room.id),
            "project_id": project_id,
            "developer_id": str(project.developer_id),
            "investor_id": str(investor_id),
            "status": "active",
            "created_at": chat_room.created_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating chat room: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Get chat room details
@app.get("/api/chat/rooms/{room_id}")
async def get_chat_room(
    room_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get chat room details"""
    try:
        result = await db.execute(
            select(ChatRoom).where(ChatRoom.id == uuid.UUID(room_id))
        )
        room = result.scalar_one_or_none()
        
        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")
        
        # Verify user is part of this chat
        user_id = uuid.UUID(current_user["id"])
        if user_id not in [room.developer_id, room.investor_id]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return model_to_dict(room)
        
    except Exception as e:
        logger.error(f"Error getting chat room: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def moderate_message_async(message_id: str, message_text: str, room_id: str, sender_id: str, user_email: str):
    """
    Background task to moderate a message after it's been sent
    If violations found, flag the message and notify users
    """
    try:
        logger.info(f"🔍 Background moderation for message {message_id}")

        # Run OpenAI moderation
        moderation_result = await openai_chat_moderation(message_text)

        # Create new database session for background task
        async with async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)() as db:
            # Fetch the message
            msg_query = select(ChatMessage).where(ChatMessage.id == uuid.UUID(message_id))
            result = await db.execute(msg_query)
            message = result.scalar_one_or_none()

            if not message:
                logger.warning(f"Message {message_id} not found for moderation")
                return

            if not moderation_result['is_safe']:
                # Message violates policy - flag it
                logger.warning(f"⚠️ Message {message_id} flagged after moderation")
                logger.warning(f"Violations: {moderation_result['violations']}")

                message.flagged = True
                message.moderation_status = 'rejected'

                # Create user-friendly reason
                first_violation = moderation_result['violations'][0] if moderation_result['violations'] else {}
                violation_type = first_violation.get('type', 'unknown')

                if violation_type == 'contact_info':
                    reason = "Contains contact information"
                elif violation_type == 'bypass_attempt':
                    reason = "Attempting to move conversation off-platform"
                elif violation_type == 'openai_moderation':
                    categories = first_violation.get('categories', [])
                    reason = f"Violates community guidelines: {categories[0] if categories else 'inappropriate content'}"
                else:
                    reason = "Violates community guidelines"

                message.moderation_reason = reason

                # Log the violation
                try:
                    log_entry = ContentFilterLog(
                        chat_room_id=uuid.UUID(room_id),
                        user_id=uuid.UUID(sender_id),
                        original_message=message_text,
                        filtered_content=[reason],
                        action_taken='flagged'
                    )
                    db.add(log_entry)
                except Exception as log_error:
                    logger.error(f"Failed to log violation: {log_error}")

                await db.commit()

                # Notify both users via WebSocket that message was flagged
                # Send directly to room connections without wrapping
                logger.info(f"📡 Sending message_flagged WebSocket to room {room_id}")
                if room_id in manager.chat_rooms:
                    connection_count = len(manager.chat_rooms[room_id])
                    logger.info(f"📡 Found {connection_count} connections in room")
                    for connection in manager.chat_rooms[room_id]:
                        try:
                            await connection.send_json({
                                "type": "message_flagged",
                                "message_id": message_id,
                                "reason": reason
                            })
                            logger.info(f"✅ Sent message_flagged to connection")
                        except Exception as e:
                            logger.error(f"Error sending flagged notification: {e}")
                else:
                    logger.warning(f"⚠️ Room {room_id} not found in chat_rooms: {list(manager.chat_rooms.keys())}")

                logger.info(f"🚫 Message {message_id} flagged and users notified")
            else:
                # Message is safe - mark as approved
                message.moderation_status = 'approved'
                await db.commit()
                logger.info(f"✅ Message {message_id} approved")

    except Exception as e:
        logger.error(f"❌ Error in background moderation: {e}", exc_info=True)


# Send chat message
@app.post("/api/chat/rooms/{room_id}/messages")
async def send_chat_message(
    room_id: str,
    message_data: ChatMessageCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Send chat message with OPTIMISTIC SENDING + ASYNC moderation:
    1. Save message immediately (instant UX)
    2. Moderate in background
    3. Flag/delete if violations found
    """

    try:
        user_id = uuid.UUID(current_user['id'])
        message_text = message_data.message.strip() if message_data.message else ""

        if not message_text and not message_data.file_url:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Verify room exists and user is a participant
        room_query = select(ChatRoom).where(ChatRoom.id == uuid.UUID(room_id))
        result = await db.execute(room_query)
        room = result.scalar_one_or_none()
        
        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")
        
        if user_id not in [room.developer_id, room.investor_id]:
            raise HTTPException(status_code=403, detail="Not a participant in this chat")
        
        # Create message with pending moderation status
        new_message = ChatMessage(
            room_id=uuid.UUID(room_id),
            sender_id=user_id,
            message=message_text if message_text else None,
            file_url=message_data.file_url,
            file_name=message_data.file_name,
            file_size=message_data.file_size,
            file_type=message_data.file_type,
            message_type=message_data.message_type,
            moderation_status='pending',
            flagged=False
        )
        db.add(new_message)

        recipient_id = room.investor_id if user_id == room.developer_id else room.developer_id

        # Update room's last message timestamp
        room.last_message_at = datetime.utcnow()

        # Commit everything in one transaction
        await db.commit()
        await db.refresh(new_message)

        message_dict = model_to_dict(new_message)

        # Send via WebSocket IMMEDIATELY (optimistic send)
        await manager.send_chat_message(room_id, message_dict)

        # Schedule background moderation ONLY for text messages
        if message_text:
            background_tasks.add_task(
                moderate_message_async,
                str(new_message.id),
                message_text,
                room_id,
                str(user_id),
                current_user['email']
            )
            logger.info(f"📋 Scheduled background moderation for message {new_message.id}")

        if recipient_id:
            total_unread = await get_total_unread_chat_count(db, recipient_id)
            room_unread_result = await db.execute(
                select(func.count(ChatMessage.id))
                .where(
                    and_(
                        ChatMessage.room_id == uuid.UUID(room_id),
                        ChatMessage.sender_id != recipient_id,
                        ChatMessage.read == False
                    )
                )
            )
            room_unread = room_unread_result.scalar() or 0
            await manager.send_ws_event(
                str(recipient_id),
                {
                    "type": "chat_unread_count",
                    "total_unread_count": total_unread,
                    "room_id": room_id,
                    "room_unread_count": room_unread
                }
            )
        
        logger.info(f"✅ Message sent in room {room_id}")
        
        return message_dict
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error sending message: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# # Get Room Chat Messages 
# @app.get("/api/chat/rooms/{room_id}/messages")
# async def get_chat_messages(
#     room_id: str,
#     limit: int = Query(50, le=100),
#     before: Optional[str] = None,
#     db: AsyncSession = Depends(get_db),
#     current_user = Depends(get_current_user)
# ):
#     """Get messages from a chat room"""
#     try:
#         # Verify room exists and user has access
#         room_result = await db.execute(
#             select(ChatRoom).where(ChatRoom.id == uuid.UUID(room_id))
#         )
#         room = room_result.scalar_one_or_none()
        
#         if not room:
#             raise HTTPException(status_code=404, detail="Chat room not found")
        
#         user_id = uuid.UUID(current_user["id"])
#         if user_id not in [room.developer_id, room.investor_id]:
#             raise HTTPException(status_code=403, detail="Access denied")
        
#         # Build query
#         query = select(ChatMessage).where(ChatMessage.room_id == uuid.UUID(room_id))
        
#         if before:
#             query = query.where(ChatMessage.created_at < datetime.fromisoformat(before))
        
#         query = query.order_by(ChatMessage.created_at.desc()).limit(limit)
        
#         result = await db.execute(query)
#         messages = list(reversed(result.scalars().all()))
        
#         # Mark messages as read
#         unread_ids = [msg.id for msg in messages if not msg.read and msg.sender_id != user_id]
        
#         if unread_ids:
#             await db.execute(
#                 update(ChatMessage)
#                 .where(ChatMessage.id.in_(unread_ids))
#                 .values(read=True, read_at=func.now())
#             )
#             await db.commit()
#             total_unread = await get_total_unread_chat_count(db, user_id)
#             await manager.send_ws_event(
#                 str(user_id),
#                 {"type": "chat_unread_count", "total_unread_count": total_unread}
#             )
        
#         return models_to_list(messages)
        
#     except Exception as e:
#         logger.error(f"Error getting chat messages: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


# Get Room Chat Messages 
@app.get("/api/chat/rooms/{room_id}/messages")
async def get_chat_messages(
    room_id: str,
    before: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get messages from a chat room (no limit - returns all messages)"""
    try:
        # Verify room exists and user has access
        room_result = await db.execute(
            select(ChatRoom).where(ChatRoom.id == uuid.UUID(room_id))
        )
        room = room_result.scalar_one_or_none()

        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")

        user_id = uuid.UUID(current_user["id"])
        if user_id not in [room.developer_id, room.investor_id]:
            raise HTTPException(status_code=403, detail="Access denied")

        # Build query - ✅ ORDER BY created_at ASC for chronological order
        query = select(ChatMessage).where(ChatMessage.room_id == uuid.UUID(room_id))

        if before:
            query = query.where(ChatMessage.created_at < datetime.fromisoformat(before))

        # ✅ Order by created_at ascending (oldest first) - NO LIMIT
        # This ensures all messages are returned in chronological order
        query = query.order_by(ChatMessage.created_at.asc())
        
        result = await db.execute(query)
        messages = list(result.scalars().all())  # ✅ No need to reverse

        logger.info(f"📬 Fetching messages for room {room_id}: Found {len(messages)} messages")

        # Mark messages as read
        unread_ids = [msg.id for msg in messages if not msg.read and msg.sender_id != user_id]
        
        if unread_ids:
            await db.execute(
                update(ChatMessage)
                .where(ChatMessage.id.in_(unread_ids))
                .values(read=True, read_at=func.now())
            )
            await db.commit()
            total_unread = await get_total_unread_chat_count(db, user_id)
            await manager.send_ws_event(
                str(user_id),
                {"type": "chat_unread_count", "total_unread_count": total_unread}
            )
        
        # ✅ IMPORTANT: Ensure created_at is properly serialized as ISO string
        serialized_messages = []
        for msg in messages:
            msg_dict = {
                'id': str(msg.id),
                'room_id': str(msg.room_id),
                'sender_id': str(msg.sender_id) if msg.sender_id else None,
                'message': msg.message,
                'message_type': msg.message_type,
                'file_url': msg.file_url,
                'file_name': msg.file_name,
                'file_size': msg.file_size,
                'read': msg.read,
                'read_at': msg.read_at.isoformat() + 'Z' if msg.read_at else None,
                'created_at': msg.created_at.isoformat() + 'Z' if msg.created_at else None,
                'sender_type': getattr(msg, 'sender_type', None),
                'sender_role': getattr(msg, 'sender_role', None),
                'is_admin': getattr(msg, 'is_admin', False),
                # Moderation fields
                'flagged': getattr(msg, 'flagged', False),
                'moderation_status': getattr(msg, 'moderation_status', 'approved'),
                'moderation_reason': getattr(msg, 'moderation_reason', None)
            }
            serialized_messages.append(msg_dict)

        logger.info(f"📤 Returning {len(serialized_messages)} serialized messages to frontend")

        return serialized_messages
        
    except Exception as e:
        logger.error(f"Error getting chat messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Mark a chat message as read
@app.put("/api/chat/messages/{message_id}/read")
async def mark_message_as_read(
    message_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Mark a chat message as read"""
    try:
        message_uuid = uuid.UUID(message_id)
        user_id = uuid.UUID(current_user["id"])
        
        # Get the message
        result = await db.execute(
            select(ChatMessage).where(ChatMessage.id == message_uuid)
        )
        message = result.scalar_one_or_none()
        
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        
        # Get the chat room to verify user is a participant
        room_result = await db.execute(
            select(ChatRoom).where(ChatRoom.id == message.room_id)
        )
        room = room_result.scalar_one_or_none()
        
        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")
        
        # Verify user is a participant in the chat room
        if user_id not in [room.developer_id, room.investor_id]:
            raise HTTPException(status_code=403, detail="Not authorized to access this chat")
        
        # Only allow marking if you're NOT the sender
        if message.sender_id == user_id:
            raise HTTPException(status_code=403, detail="Cannot mark own message as read")
        
        # Update read status
        message.read = True
        message.read_at = datetime.utcnow()
        await db.commit()
        
        total_unread = await get_total_unread_chat_count(db, user_id)
        await manager.send_ws_event(
            str(user_id),
            {"type": "chat_unread_count", "total_unread_count": total_unread}
        )
        
        logger.info(f"Message {message_id} marked as read by user {user_id}")
        
        return {
            "success": True,
            "message_id": str(message_id),
            "read_at": message.read_at.isoformat()
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid message ID format")
    except Exception as e:
        logger.error(f"Error marking message as read: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# Get unread message count for a chat room
@app.get("/api/chat/rooms/{room_id}/unread-count")
async def get_unread_count(
    room_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get unread message count for a chat room"""
    try:
        user_id = uuid.UUID(current_user["id"])
        
        room_result = await db.execute(
            select(ChatRoom).where(ChatRoom.id == uuid.UUID(room_id))
        )
        room = room_result.scalar_one_or_none()
        
        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")
        
        if user_id not in [room.developer_id, room.investor_id]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        result = await db.execute(
            select(func.count(ChatMessage.id))
            .where(
                and_(
                    ChatMessage.room_id == uuid.UUID(room_id),
                    ChatMessage.sender_id != user_id,
                    ChatMessage.read == False
                )
            )
        )
        count = result.scalar()
        
        return {"unread_count": count}
        
    except Exception as e:
        logger.error(f"Error getting unread count: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get total unread message count across all chat rooms
@app.get("/api/chat/unread-count/total")
async def get_total_unread_count(
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get total unread message count across all chat rooms"""
    try:
        user_id = uuid.UUID(current_user["id"])
        count = await get_total_unread_chat_count(db, user_id)
        
        return {"total_unread_count": count or 0}
        
    except Exception as e:
        logger.error(f"Error getting total unread count: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/rooms/{room_id}/upload")
async def upload_chat_file(
    room_id: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    temp_path = None

    try:
        logger.info(f"📂 Upload request received: {file.filename}, Room: {room_id}, User: {current_user.get('id')}")

        # STEP 1: AUTHORIZATION
        logger.info("🔑 Checking authorization...")
        room_query = await db.execute(select(ChatRoom).where(ChatRoom.id == uuid.UUID(room_id)))
        room = room_query.scalar_one_or_none()
        if not room:
            logger.info("❌ Chat room not found")
            raise HTTPException(status_code=404, detail="Chat room not found")

        user_id = uuid.UUID(current_user["id"])
        if user_id not in [room.developer_id, room.investor_id]:
            logger.info("❌ Access denied for user")
            raise HTTPException(status_code=403, detail="Access denied")
        logger.info("✅ Authorization passed")

        # STEP 2: FILE VALIDATION
        logger.info("📄 Validating file...")
        filename = file.filename
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext not in ALLOWED_IMAGE_EXTENSIONS:
            logger.info("❌ Invalid file type")
            raise HTTPException(status_code=400, detail="Only PNG and JPEG images are allowed.")

        content = await file.read()
        if len(content) == 0:
            logger.info("❌ File is empty")
            raise HTTPException(status_code=400, detail="Cannot upload empty file.")

        max_size = 5 * 1024 * 1024
        if len(content) > max_size:
            logger.info("❌ File too large")
            raise HTTPException(status_code=400, detail=f"File too large. Max size: {max_size / 1024 / 1024} MB")
        logger.info("✅ File validation passed")

        # STEP 3: SAVE TEMPORARY FILE
        logger.info("💾 Saving temporary file...")
        temp_dir = "uploads/temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_filename = f"{uuid.uuid4()}_{filename}"
        temp_path = os.path.join(temp_dir, temp_filename)
        with open(temp_path, 'wb') as f:
            f.write(content)
        logger.info(f"✅ Temporary file saved at {temp_path}")

        # STEP 4: OPENAI MODERATION
        logger.info("🛡️ Running OpenAI moderation...")
        openai_result = await openai_image_moderation(temp_path)
        logger.info(f"🛡️ Moderation result: {openai_result}")

        if openai_result["contains_harmful_content"]:
            logger.info("❌ Image contains harmful content")
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail="This image contains harmful content.")

        if openai_result["contains_contact_info"]:
            logger.info("❌ Image contains contact info")
            os.remove(temp_path)
            raise HTTPException(
                status_code=400,
                detail="This image contains contact information (phone, email, social media, or URLs)."
            )
        logger.info("✅ Image passed moderation")

        # STEP 5: SAVE TO PERMANENT LOCATION
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_filename = os.path.basename(filename).replace("..", "")
        safe_filename = f"{user_id}_{timestamp}_{clean_filename}"
        final_path = os.path.join(UPLOAD_DIR, safe_filename)
        shutil.move(temp_path, final_path)
        temp_path = None
        logger.info(f"💾 File moved to permanent location: {final_path}")

        # STEP 6: CREATE DATABASE MESSAGE
        logger.info("📝 Creating database message...")
        new_message = ChatMessage(
            id=uuid.uuid4(),
            room_id=uuid.UUID(room_id),
            sender_id=user_id,
            message=f"🖼️ Shared an image: {filename}",
            message_type='file',
            file_url=f"/uploads/chat_files/{safe_filename}",
            file_name=filename,
            file_type=file_ext,
            file_size=len(content),
            created_at=datetime.utcnow()
        )
        db.add(new_message)
        await db.commit()
        await db.refresh(new_message)
        logger.info(f"✅ Database message created: {new_message.id}")

        # STEP 7: BROADCAST VIA WEBSOCKET
        logger.info("📡 Broadcasting message via WebSocket...")
        message_dict = {
            "id": str(new_message.id),
            "room_id": room_id,
            "sender_id": str(user_id),
            "message": new_message.message,
            "message_type": 'file',
            "file_url": new_message.file_url,
            "file_name": new_message.file_name,
            "file_type": new_message.file_type,
            "file_size": new_message.file_size,
            "read": False,
            "created_at": new_message.created_at.isoformat() + 'Z',
            "timestamp": new_message.created_at.isoformat() + 'Z'
        }

        await manager.send_chat_message(room_id, message_dict)
        logger.info("✅ Broadcast complete")

        return {
            "success": True,
            "message_id": str(new_message.id),
            "file_url": new_message.file_url,
            "file_name": new_message.file_name,
            "file_size": len(content),
            "message": message_dict
        }

    except HTTPException:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/uploads/chat_files/{filename}")
async def download_chat_file(
    filename: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Secure file download with full authentication and authorization
    ✅ Requires valid JWT token in Authorization header
    ✅ Verifies user has access to the chat room
    ✅ Allows admin users to access all files
    """
    
    try:
        # 1. SANITIZE FILENAME - Prevent path traversal
        safe_filename = os.path.basename(filename)
        file_path = os.path.join(UPLOAD_DIR, safe_filename)
        
        # 2. CHECK FILE EXISTS ON DISK
        if not os.path.exists(file_path):
            logger.warning(f"File not found on disk: {safe_filename}")
            raise HTTPException(status_code=404, detail="File not found")
        
        # 3. GET CURRENT USER ID AND ROLE
        user_id = uuid.UUID(current_user["id"])
        user_role = current_user.get("role", "")
        
        # ✅ NEW: Allow admins to access all files without further checks
        if user_role == "admin":
            logger.info(f"📥 Admin file access: {safe_filename} by admin user {user_id}")
            
            # Determine content type
            content_type, _ = mimetypes.guess_type(file_path)
            if content_type is None:
                content_type = 'application/octet-stream'
            
            # Return file for admin
            return FileResponse(
                path=file_path,
                media_type=content_type,
                filename=safe_filename,
                headers={
                    "Content-Disposition": f"attachment; filename={safe_filename}",
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
        
        # 4. VERIFY FILE IS IN DATABASE (for non-admin users)
        # Find the message that contains this file
        message_query = select(ChatMessage).where(
            ChatMessage.file_url == f"/uploads/chat_files/{safe_filename}"
        )
        message_result = await db.execute(message_query)
        message = message_result.scalar_one_or_none()
        
        if not message:
            logger.warning(f"File not found in database: {safe_filename}")
            raise HTTPException(status_code=404, detail="File not found in system")
        
        # 5. AUTHORIZATION - Verify user has access to this chat room
        room_query = select(ChatRoom).where(ChatRoom.id == message.room_id)
        room_result = await db.execute(room_query)
        room = room_result.scalar_one_or_none()
        
        if not room:
            logger.warning(f"Chat room not found for file: {safe_filename}")
            raise HTTPException(status_code=404, detail="Chat room not found")
        
        # Check if current user is a participant in this chat room
        if user_id not in [room.developer_id, room.investor_id]:
            logger.warning(f"Unauthorized access attempt by user {user_id} (role: {user_role}) for file: {safe_filename}")
            raise HTTPException(status_code=403, detail="You don't have access to this file")
        
        # 6. DETERMINE CONTENT TYPE
        content_type, _ = mimetypes.guess_type(file_path)
        if content_type is None:
            content_type = 'application/octet-stream'
        
        # 7. LOG SUCCESSFUL DOWNLOAD
        logger.info(f"📥 File download: {safe_filename} by user {user_id}")
        
        # 8. RETURN FILE
        return FileResponse(
            path=file_path,
            media_type=content_type,
            filename=safe_filename,
            headers={
                "Content-Disposition": f"attachment; filename={safe_filename}",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error downloading file {filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to download file")


# Get metadata for a file attachment
@app.get("/api/chat/files/{message_id}")
async def get_file_metadata(
    message_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get metadata for a file attachment"""
    
    try:
        # Get message
        result = await db.execute(
            select(ChatMessage).where(ChatMessage.id == uuid.UUID(message_id))
        )
        message = result.scalar_one_or_none()
        
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        
        if message.message_type != 'file':
            raise HTTPException(status_code=400, detail="Message is not a file")
        
        # Verify user has access to the room
        room_result = await db.execute(
            select(ChatRoom).where(ChatRoom.id == message.room_id)
        )
        room = room_result.scalar_one_or_none()
        
        user_id = uuid.UUID(current_user["id"])
        if user_id not in [room.developer_id, room.investor_id]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return {
            "id": str(message.id),
            "file_name": message.file_name,
            "file_url": message.file_url,
            "file_type": message.file_type,
            "file_size": message.file_size,
            "uploaded_by": str(message.sender_id),
            "uploaded_at": message.created_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting file metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/notifications/unread-total")
async def get_total_unread_notifications_and_chats(
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get combined count of unread notifications and chat messages"""
    try:
        user_id = uuid.UUID(current_user["id"])
        
        # Count unread notifications
        notif_result = await db.execute(
            select(func.count(Notification.id))
            .where(
                and_(
                    Notification.user_id == user_id,
                    Notification.read == False
                )
            )
        )
        notif_count = notif_result.scalar() or 0
        
        # Count unread chat messages
        # Get all rooms where user is a participant
        rooms_result = await db.execute(
            select(ChatRoom).where(
                or_(
                    ChatRoom.developer_id == user_id,
                    ChatRoom.investor_id == user_id
                )
            )
        )
        rooms = rooms_result.scalars().all()
        room_ids = [room.id for room in rooms]
        
        # Count unread messages in all rooms
        if room_ids:
            chat_result = await db.execute(
                select(func.count(ChatMessage.id))
                .where(
                    and_(
                        ChatMessage.room_id.in_(room_ids),
                        ChatMessage.sender_id != user_id,
                        ChatMessage.read == False
                    )
                )
            )
            chat_count = chat_result.scalar() or 0
        else:
            chat_count = 0
        
        return {
            "total_unread": notif_count + chat_count,
            "notification_count": notif_count,
            "chat_count": chat_count
        }
        
    except Exception as e:
        logger.error(f"Error getting total unread: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/notifications/mark-all-read")
async def mark_all_notifications_and_chats_read(
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Mark all notifications and chat messages as read"""
    try:
        user_id = uuid.UUID(current_user["id"])
        
        # Mark all notifications as read
        await db.execute(
            update(Notification)
            .where(
                and_(
                    Notification.user_id == user_id,
                    Notification.read == False
                )
            )
            .values(read=True, read_at=datetime.utcnow())
        )
        
        # Mark all chat messages as read
        # Get all rooms where user is a participant
        rooms_result = await db.execute(
            select(ChatRoom).where(
                or_(
                    ChatRoom.developer_id == user_id,
                    ChatRoom.investor_id == user_id
                )
            )
        )
        rooms = rooms_result.scalars().all()
        room_ids = [room.id for room in rooms]
        
        # Mark all unread messages in these rooms as read
        if room_ids:
            await db.execute(
                update(ChatMessage)
                .where(
                    and_(
                        ChatMessage.room_id.in_(room_ids),
                        ChatMessage.sender_id != user_id,
                        ChatMessage.read == False
                    )
                )
                .values(read=True, read_at=datetime.utcnow())
            )
        
        await db.commit()
        
        logger.info(f"User {user_id} marked all notifications and chats as read")
        
        return {
            "success": True,
            "message": "All notifications and messages marked as read"
        }
        
    except Exception as e:
        logger.error(f"Error marking all as read: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

        

# ============================================
# AUTHENTICATION ROUTES
# ============================================

@app.post("/api/auth/register")
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new user with email verification"""

    # Check if user exists
    result = await db.execute(select(User).where(User.email == user_data.email))
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Hash password
    hashed_password = hash_password(user_data.password)

    # Generate verification token
    verification_token = token_urlsafe(48)

    # Create user (NO BANK DETAILS)
    new_user = User(
        email=user_data.email,
        password_hash=hashed_password,
        role=user_data.role,
        status="pending",
        name=user_data.name,
        company=user_data.company,
        email_verified=False,
        verification_token=verification_token,
        verification_sent_at=datetime.utcnow(),
        reputation_score=0,
        profile={"cosmic_theme": "default", "avatar": None, "bio": ""}
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    # Send verification email
    await send_verification_email(new_user.email, verification_token)

    # DO NOT AUTO LOGIN — do NOT return token
    return {
        "message": "Registration successful! Please check your email to verify your account."
    }


@app.post("/api/auth/login")
async def login(login_data: UserLogin, db: AsyncSession = Depends(get_db)):
    """Login user only if email is verified"""
    result = await db.execute(select(User).where(User.email == login_data.email))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Check email verified
    if not user.email_verified:
        raise HTTPException(
            status_code=403,
            detail="Please verify your email before logging in."
        )

    # Check password
    if not user.password_hash or not verify_password(login_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Login time
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    user.last_login = now_utc

    db.add(user)
    await db.commit()
    await db.refresh(user)

    # Prepare user dict
    user_dict = {
        "id": str(user.id),
        "email": user.email,
        "role": user.role,
        "name": user.name,
        "reputation_score": user.reputation_score,
    }

    token = create_jwt_token(user_dict)

    return {"token": token, "user": user_dict}


@app.get("/api/auth/verify/{token}")
async def verify_email(token: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.verification_token == token))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=400, detail="Invalid verification token")

    user.email_verified = True
    user.verification_token = None
    user.status = "active"
    user.verified = True  # Set user as verified when email is verified

    await db.commit()

    return {"message": "Email verified successfully! You may now login."}


@app.get("/api/auth/me")
async def get_me(current_user = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Get current user profile"""
    return current_user


@app.put("/api/auth/role")
async def update_role(
    role_update: RoleUpdate,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update user role"""
    if role_update.role not in ["developer", "investor"]:
        raise HTTPException(status_code=400, detail="Invalid role")
    
    user_id = uuid.UUID(current_user["_id"])
    await db.execute(
        update(User).where(User.id == user_id).values(role=role_update.role)
    )
    await db.commit()
    
    return {"message": "Role updated successfully", "role": role_update.role}




# ============================================
# USER ENDPOINTS
# ============================================

@app.get("/api/users/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current user's full profile"""
    user_id = current_user.get("_id") or current_user.get("id")
    
    # Handle both string UUID and UUID object
    if isinstance(user_id, str):
        user_id = uuid.UUID(user_id)
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return model_to_dict(user)


@app.put("/api/users/me", response_model=UserResponse)
async def update_current_user_profile(
    user_update: UserUpdate,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    """Update current user's profile"""
    user_id = current_user.get("_id") or current_user.get("id")
    
    # Check if email is being updated
    if user_update.email and user_update.email != current_user["email"]:
        # Check if new email is already taken
        result = await db.execute(
            select(User).where(
                User.email == user_update.email,
                User.id != uuid.UUID(user_id)
            )
        )
        existing_user = result.scalar_one_or_none()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Generate email change verification token
        email_change_token = token_urlsafe(32)
        
        # Store pending email and verification token
        await db.execute(
            update(User)
            .where(User.id == uuid.UUID(user_id))
            .values(
                pending_email=user_update.email,
                email_change_token=email_change_token,
                email_change_sent_at=datetime.now(timezone.utc).replace(tzinfo=None)
            )
        )
        await db.commit()
        
        # Send verification email
        verification_link = f"{FRONTEND_URL}/verify-email-change?token={email_change_token}"
        try:
            subject = "Verify Your New Email Address"
            
            html_content = f"""\
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1.0" />
                <title>Email Change Verification</title>
            </head>
            <body style="margin:0; padding:0; background:#f8fafc; font-family:'Segoe UI', Arial, sans-serif;">
                <table width="100%" border="0" cellspacing="0" cellpadding="0"
                    style="background:#f8fafc; padding:40px 0;">
                    <tr>
                        <td align="center">
                            <table class="container card" width="600" border="0" cellspacing="0" cellpadding="0"
                                style="background:white; border-radius:16px; box-shadow:0 4px 20px rgba(0,0,0,0.06); padding:40px;">
                                <tr>
                                    <td align="center" style="padding-bottom:25px;">
                                        <h1 style="margin:0; font-size:28px; font-weight:700; color:#4f46e5;">
                                            BiteBids
                                        </h1>
                                        <p style="margin:8px 0 0; color:#6b7280; font-size:14px;">
                                            Verify your new email address
                                        </p>
                                    </td>
                                </tr>
                                <tr>
                                    <td align="center" style="padding-bottom:15px;">
                                        <h2 style="margin:0; font-size:22px; color:#111827; font-weight:600;">
                                            Email Change Request
                                        </h2>
                                    </td>
                                </tr>
                                <tr>
                                    <td style="padding-bottom:25px;">
                                        <p style="margin:0; color:#374151; font-size:15px; line-height:1.6; text-align:center;">
                                            Hello {current_user['name']},<br><br>
                                            You requested to change your email address to <strong>{user_update.email}</strong>.<br><br>
                                            Please click the button below to verify your new email address.
                                        </p>
                                    </td>
                                </tr>
                                <tr>
                                    <td align="center" style="padding-bottom:30px;">
                                        <a href="{verification_link}" target="_blank"
                                            style="display:inline-block; background:#4f46e5; color:white;
                                            padding:14px 40px; font-size:17px; font-weight:600;
                                            border-radius:10px; text-decoration:none;">
                                            Verify New Email
                                        </a>
                                    </td>
                                </tr>
                                <tr>
                                    <td style="padding-bottom:25px;">
                                        <p style="color:#6b7280; font-size:13px; line-height:1.5;">
                                            If the button doesn't work, copy the link below:
                                        </p>
                                        <p style="word-break:break-all;">
                                            <a href="{verification_link}" style="color:#4f46e5; font-size:13px;">
                                                {verification_link}
                                            </a>
                                        </p>
                                        <p style="color:#ef4444; font-size:13px; margin-top:15px;">
                                            ⏰ This link will expire in 24 hours.
                                        </p>
                                        <p style="color:#6b7280; font-size:13px; margin-top:15px;">
                                            If you didn't request this change, please ignore this email.
                                        </p>
                                    </td>
                                </tr>
                                <tr>
                                    <td style="padding:20px 0;">
                                        <hr style="border:0; border-top:1px solid #e5e7eb;" />
                                    </td>
                                </tr>
                                <tr>
                                    <td align="center">
                                        <p style="margin:0; font-size:11px; color:#9ca3af;">
                                            © 2024 BiteBids. All rights reserved.
                                        </p>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                </table>
            </body>
            </html>
            """
            
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = EMAIL_FROM
            msg["To"] = user_update.email
            
            msg.attach(MIMEText(html_content, "html"))
            
            # Send email
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(EMAIL_FROM, user_update.email, msg.as_string())
            server.quit()
            
            logger.info(f"📧 Email change verification sent to {user_update.email}")
            
        except Exception as e:
            logger.error(f"Failed to send email change verification: {e}")
            raise HTTPException(status_code=500, detail="Failed to send verification email")
        
        # Remove email from update_data as it's pending verification
        user_update.email = None
        
        # Return message about verification
        result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
        user = result.scalar_one_or_none()
        response = model_to_dict(user)
        return {**response, "message": "Verification email sent to your new email address"}
    
    # Prepare update data (NO BANK FIELDS)
    update_data = {}
    for field, value in user_update.dict(exclude_unset=True).items():
        if value is not None:
            update_data[field] = value
    
    if update_data:
        update_data['updated_at'] = datetime.now(timezone.utc).replace(tzinfo=None)
        
        await db.execute(
            update(User)
            .where(User.id == uuid.UUID(user_id))
            .values(**update_data)
        )
        await db.commit()
    
    # Return updated user
    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    updated_user = result.scalar_one_or_none()
    
    return model_to_dict(updated_user)


@app.get("/api/users/{user_id}", response_model=UserResponse)
async def get_user_by_id(
    user: User = Depends(get_user_or_self),
    current_user: dict = Depends(get_current_user)
):
    """Get user by ID (admin or self)"""
    return model_to_dict(user)

@app.put("/api/users/{user_id}", response_model=UserResponse)
async def update_user_by_id(
    user_update: UserUpdate,
    user: User = Depends(get_user_or_self),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update user by ID (admin or self)"""
    user_id = str(user.id)
    
    # Check if email is being updated and if it's already taken
    if user_update.email and user_update.email != user.email:
        result = await db.execute(
            select(User).where(
                User.email == user_update.email,
                User.id != uuid.UUID(user_id)
            )
        )
        existing_user = result.scalar_one_or_none()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
    
    # Prepare update data (NO BANK FIELDS)
    update_data = {}
    for field, value in user_update.dict(exclude_unset=True).items():
        if value is not None:
            update_data[field] = value
    
    if update_data:
        update_data['updated_at'] = datetime.now(timezone.utc).replace(tzinfo=None)
        
        await db.execute(
            update(User)
            .where(User.id == uuid.UUID(user_id))
            .values(**update_data)
        )
        await db.commit()
    
    # Return updated user
    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    updated_user = result.scalar_one_or_none()
    
    return model_to_dict(updated_user)


@app.patch("/api/users/{user_id}/verify")
async def verify_user(
    user_id: str,
    current_user: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Verify a user (admin only)"""
    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.verified:
        raise HTTPException(status_code=400, detail="User already verified")
    
    # Update verification status
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    await db.execute(
        update(User)
        .where(User.id == uuid.UUID(user_id))
        .values(
            verified=True,
            verification_date=now_utc,
            updated_at=now_utc
        )
    )
    await db.commit()
    
    return {"message": "User verified successfully"}


@app.patch("/api/users/{user_id}/role")
async def update_user_role(
    user_id: str,
    role_update: RoleUpdate,
    current_user: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Update user role (admin only)"""
    if role_update.role not in ["developer", "investor", "admin"]:
        raise HTTPException(status_code=400, detail="Invalid role")
    
    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Prevent changing own role away from admin
    current_user_id = current_user.get("_id") or current_user.get("id")
    if user_id == current_user_id and role_update.role != "admin":
        raise HTTPException(status_code=400, detail="Cannot remove admin role from yourself")
    
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    await db.execute(
        update(User)
        .where(User.id == uuid.UUID(user_id))
        .values(role=role_update.role, updated_at=now_utc)
    )
    await db.commit()
    
    return {"message": "User role updated successfully"}

# ============================================
# USER CREDIT ENDPOINTS
# ============================================
@app.get("/api/projects/check-posting-status")
async def check_posting_status(
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Check if user has available posting credits or needs to pay"""
    try:
        user_id = uuid.UUID(current_user["id"])
        
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if user.role != "developer":
            raise HTTPException(status_code=403, detail="Only developers can post projects")
        
        credits = user.posting_credits or 0
        has_credit = credits > 0
        
        return {
            "success": True,
            "has_credit": has_credit,
            "credits": credits,
            "needs_payment": not has_credit,
            "posting_fee": PROJECT_POSTING_FEE
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking posting status: {e}")
        raise HTTPException(status_code=500, detail="Failed to check posting status")


@app.get("/api/users/me/posting-credits")
async def get_posting_credits(
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current user's posting credits"""
    try:
        user_id = uuid.UUID(current_user["id"])
        
        result = await db.execute(
            select(User.posting_credits).where(User.id == user_id)
        )
        credits = result.scalar_one_or_none() or 0
        
        return {
            "success": True,
            "credits": credits,
            "has_credit": credits > 0
        }
        
    except Exception as e:
        logger.error(f"Error getting posting credits: {e}")
        raise HTTPException(status_code=500, detail="Failed to get posting credits")


@app.post("/api/admin/users/{user_id}/add-credits")
async def admin_add_posting_credits(
    user_id: str,
    credits: int = 1,
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Admin manually adds posting credits to a user"""
    try:
        user_uuid = uuid.UUID(user_id)
        
        result = await db.execute(
            select(User).where(User.id == user_uuid)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        user.posting_credits = (user.posting_credits or 0) + credits
        await db.commit()
        await db.refresh(user)
        
        # Send notification
        notif = Notification(
            user_id=user_uuid,
            type="credits_added",
            title="Posting Credits Added! 🎉",
            message=f"Admin has added {credits} posting credit(s) to your account.",
            link="/dashboard",
            details={
                "credits_added": credits,
                "total_credits": user.posting_credits
            }
        )
        db.add(notif)
        await db.commit()
        
        return {
            "success": True,
            "message": f"Added {credits} credits to user",
            "new_total": user.posting_credits
        }
        
    except Exception as e:
        logger.error(f"Error adding credits: {e}")
        raise HTTPException(status_code=500, detail="Failed to add credits")


# ============================================
# PAYMENT ENDPOINT FOR PROJECT POSTING
# ============================================

@app.post("/api/payments/create-post-project-session")
async def create_post_project_payment_session(
    request: dict,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Create a Stripe Checkout Session for project posting fee ($0.99)
    """
    
    if not STRIPE_SECRET_KEY:
        raise HTTPException(
            status_code=500, 
            detail="Stripe is not configured. Please contact support."
        )
    
    try:
        customer_email = request.get('customer_email')
        customer_name = request.get('customer_name')
        
        # Convert to cents for Stripe
        amount_cents = int(PROJECT_POSTING_FEE * 100)
        
        # Build metadata
        metadata = {
            'user_id': current_user.get('id'),
            'user_email': customer_email,
            'payment_type': 'project_posting',
            'amount': str(PROJECT_POSTING_FEE)
        }
        
        # Calculate expiry time (30 minutes)
        expiry = datetime.utcnow() + timedelta(minutes=30)
        expiry_ts = int(expiry.replace(tzinfo=timezone.utc).timestamp())

        # Create Stripe Checkout Session
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'unit_amount': amount_cents,
                    'product_data': {
                        'name': 'BiteBids Project Posting Fee',
                        'description': 'One-time fee to post a project on BiteBids marketplace',
                    },
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=f"{FRONTEND_URL}/marketplace?session_id={{CHECKOUT_SESSION_ID}}&payment_status=success",
            cancel_url=f"{FRONTEND_URL}/marketplace?payment_status=cancelled",
            customer_email=customer_email,
            metadata=metadata,
            payment_intent_data={
                'metadata': metadata,
                'description': 'BiteBids Project Posting Fee',
            },
            expires_at=expiry_ts,
        )
        
        # Create database record
        customer_id_uuid = uuid.UUID(current_user.get('id'))
        
        checkout_record = CheckoutSession(
            id=uuid.uuid4(),
            session_id=checkout_session.id,
            order_reference=f"post_project_{uuid.uuid4()}",
            external_reference=checkout_session.id,
            project_id=None,  # No project yet
            amount=Decimal(str(PROJECT_POSTING_FEE)),
            total_with_fees=Decimal(str(PROJECT_POSTING_FEE)),
            fees=Decimal('0'),
            payment_method='stripe_card',
            customer_id=customer_id_uuid,
            status='pending',
            payment_url=checkout_session.url,
            payment_method_used='stripe',
            expires_at=datetime.utcnow() + timedelta(minutes=30),
            extra_data={
                'stripe_session_id': checkout_session.id,
                'payment_type': 'project_posting',
                'metadata': metadata
            }
        )
        
        db.add(checkout_record)
        await db.commit()
        await db.refresh(checkout_record)
        
        logger.info(f"Project posting payment session created: {checkout_session.id} for user {current_user.get('email')}")
        
        return {
            "success": True,
            "session_id": checkout_session.id,
            "checkout_url": checkout_session.url,
            "order_reference": checkout_record.order_reference,
            "amount": PROJECT_POSTING_FEE
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Stripe error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error creating project posting payment session: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create payment session: {str(e)}"
        )


@app.post("/api/projects/upload-image")
async def upload_project_image(
    image: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Upload and validate project image with OpenAI moderation
    """
    temp_path = None
    
    try:
        logger.info(f"📷 Image upload request: {image.filename}, User: {current_user.get('id')}")
        
        # STEP 1: VALIDATE FILE TYPE
        filename = image.filename
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext not in ['.jpg', '.jpeg', '.png']:
            raise HTTPException(status_code=400, detail="Only JPEG and PNG images are allowed.")
        
        # STEP 2: READ FILE
        content = await image.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Cannot upload empty file.")
        
        # STEP 3: VALIDATE SIZE
        max_size = 5 * 1024 * 1024  # 5MB
        if len(content) > max_size:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Max size: {max_size / 1024 / 1024} MB"
            )
        
        # STEP 4: SAVE TEMPORARY FILE
        temp_dir = "uploads/temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_filename = f"{uuid.uuid4()}_{filename}"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        with open(temp_path, 'wb') as f:
            f.write(content)
        
        logger.info(f"✅ Temporary file saved: {temp_path}")
        
        # STEP 5: OPENAI MODERATION
        logger.info("🛡️ Running OpenAI image moderation...")
        moderation_result = await openai_image_moderation(temp_path)
        logger.info(f"🛡️ Moderation result: {moderation_result}")
        
        if moderation_result["contains_harmful_content"]:
            logger.warning("❌ Image contains harmful content")
            os.remove(temp_path)
            raise HTTPException(
                status_code=400, 
                detail="This image contains harmful or inappropriate content."
            )
        
        if moderation_result["contains_contact_info"]:
            logger.warning("❌ Image contains contact information")
            os.remove(temp_path)
            raise HTTPException(
                status_code=400, 
                detail="This image contains contact information (phone, email, social media, or URLs). Please remove any contact details from the image."
            )
        
        logger.info("✅ Image passed moderation")

        # STEP 6: OPTIMIZE AND PREPARE IMAGE FOR UPLOAD
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        user_id = current_user.get('id')
        clean_filename = os.path.splitext(os.path.basename(filename).replace("..", ""))[0]

        # Optimize image before upload
        optimized_path = temp_path
        try:
            img = Image.open(temp_path)

            # Convert RGBA to RGB if needed
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background

            # Resize if too large (max 1920px width)
            max_width = 1920
            if img.width > max_width:
                ratio = max_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

            # Save optimized image to temp
            optimized_path = temp_path.replace('.png', '.jpg').replace('.jpeg', '.jpg')
            img.save(optimized_path, format='JPEG', quality=85, optimize=True)
            logger.info(f"💾 Image optimized: {optimized_path}")

        except Exception as e:
            logger.warning(f"Image optimization failed, using original: {e}")
            optimized_path = temp_path

        # STEP 7: UPLOAD TO IMGBB
        logger.info(f"☁️ Uploading to ImgBB...")

        try:
            if not IMGBB_API_KEY:
                raise HTTPException(
                    status_code=500,
                    detail="ImgBB API key not configured. Please contact administrator."
                )

            # Read the optimized image file as base64
            with open(optimized_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')

            # Upload to ImgBB
            upload_data = {
                'key': IMGBB_API_KEY,
                'image': image_data,
                'name': f"{user_id}_{timestamp}_{clean_filename}"
            }

            response = requests.post(IMGBB_UPLOAD_URL, data=upload_data, timeout=30)
            response.raise_for_status()

            result = response.json()

            if not result.get('success'):
                raise Exception(f"ImgBB upload failed: {result.get('error', {}).get('message', 'Unknown error')}")

            image_url = result['data']['url']
            delete_url = result['data'].get('delete_url', '')

            logger.info(f"✅ Image uploaded to ImgBB: {image_url}")

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ ImgBB upload failed: {e}")
            raise HTTPException(status_code=500, detail=f"Image hosting service error: {str(e)}")
        except Exception as e:
            logger.error(f"❌ ImgBB upload failed: {e}")
            raise HTTPException(status_code=500, detail=f"Cloud storage upload failed: {str(e)}")

        finally:
            # Clean up temp files
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            if optimized_path != temp_path and os.path.exists(optimized_path):
                os.remove(optimized_path)

        logger.info(f"✅ Image upload complete: {image_url}")

        return {
            "success": True,
            "image_url": image_url,
            "filename": clean_filename,
            "delete_url": delete_url
        }
        
    except HTTPException:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Unexpected error during image upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/uploads/project_images/{filename}")
async def serve_project_image(filename: str):
    """
    Serve project images - now redirects to Cloudinary or serves old local files
    """
    try:
        # Sanitize filename
        safe_filename = os.path.basename(filename)
        file_path = os.path.join(PROJECT_IMAGES_DIR, safe_filename)

        # Check if file exists locally (for backwards compatibility with old uploads)
        if os.path.exists(file_path):
            logger.info(f"📁 Serving legacy local file: {safe_filename}")
            content_type, _ = mimetypes.guess_type(file_path)
            if content_type is None:
                content_type = 'application/octet-stream'

            return FileResponse(
                path=file_path,
                media_type=content_type,
                filename=safe_filename,
                headers={
                    "Cache-Control": "public, max-age=31536000",
                }
            )

        # File not found locally - all new images should be on ImgBB
        logger.warning(f"❌ Image not found (should be on ImgBB): {safe_filename}")
        raise HTTPException(
            status_code=404,
            detail="Image not found. New images are stored on ImgBB."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving image {filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve image")



# ============================================
# PROJECT ROUTES
# ============================================

@app.get("/api/projects")
async def get_projects(
    skip: int = 0,
    limit: int = 20,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Get projects with developer info"""
    query = select(Project).options(joinedload(Project.developer))

    if status:
        query = query.where(Project.status == status)

    query = query.order_by(Project.created_at.desc()).offset(skip).limit(limit)
    result = await db.execute(query)
    projects = result.scalars().all()

    project_list = []
    for project in projects:
        p = model_to_dict(project)

        if project.developer:
            p["developer"] = {
                "id": str(project.developer.id),
                "name": project.developer.name,
                "company": project.developer.company,
                "avatar": project.developer.avatar,
            }

        project_list.append(p)

    return {
        "projects": project_list,
        "total": len(project_list),
        "skip": skip,
        "limit": limit
    }

@app.post("/api/projects")
async def create_project(
    project_data: ProjectCreate,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    if current_user["role"] != "developer":
        raise HTTPException(403, "Only developers can create projects")


    user_id = uuid.UUID(current_user["id"])
    
    # ✅ CHECK POSTING CREDITS
    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(404, "User not found")
    
    # Check if user has credits
    if (user.posting_credits or 0) <= 0:
        raise HTTPException(
            status_code=402,  # Payment Required
            detail={
                "message": "You need to purchase posting credit before creating a project",
                "code": "NO_POSTING_CREDITS",
                "posting_fee": PROJECT_POSTING_FEE
            }
        )
    
    # ✅ DEDUCT 1 CREDIT
    user.posting_credits = (user.posting_credits or 0) - 1


    deadline = None
    if project_data.deadline:
        try:
            deadline = datetime.fromisoformat(project_data.deadline.replace("Z", "+00:00"))
        except:
            deadline = datetime.utcnow() + timedelta(days=30)

    new_project = Project(
        title=project_data.title,
        status=project_data.status,
        description=project_data.description,
        tech_stack=project_data.tech_stack,
        requirements=project_data.requirements,
        budget_range=project_data.budget_range,
        budget=project_data.budget,
        lowest_bid=project_data.lowest_bid,
        deadline=deadline,
        location=project_data.location or "Remote",
        developer_id=current_user["id"],
        category=project_data.category or "Machine Learning",
        images=project_data.images if hasattr(project_data, 'images') else []  # NEW
    )

    db.add(new_project)
    await db.commit()
    await db.refresh(new_project)

    # ✅ NEW: Send admin notification
    background_tasks.add_task(
        send_admin_project_notification,
        action="created",
        project_id=str(new_project.id),
        project_title=new_project.title,
        developer_name=current_user.get('name', 'Unknown'),
        developer_email=current_user.get('email', 'Unknown'),
        project_data={
            'category': new_project.category,
            'budget': float(new_project.budget),
            'status': new_project.status,
            'location': new_project.location,
            'tech_stack': new_project.tech_stack
        }
    )


    return {
        **model_to_dict(new_project),
        "remaining_credits": user.posting_credits
        }

@app.put("/api/projects/{project_id}")
async def update_project(
    project_id: str,
    project_data: ProjectUpdate,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    project = await db.scalar(select(Project).where(Project.id == uuid.UUID(project_id)))
    if not project:
        raise HTTPException(404, "Project not found")

    # ✅ ALLOW BOTH DEVELOPER AND ADMIN TO EDIT
    is_owner = str(project.developer_id) == str(current_user["id"])
    is_admin = current_user.get("role") == "admin"
    
    if not is_owner and not is_admin:
        raise HTTPException(403, "Only the project owner or admin can edit this project")

    # ✅ NEW: Track changes if admin is editing someone else's project
    changes = {}
    admin_editing_others_project = is_admin and not is_owner
    
    if admin_editing_others_project:
        # Track all changes
        for field, value in project_data.dict(exclude_unset=True).items():
            if field == "deadline" and value:
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            
            old_value = getattr(project, field, None)
            
            # Only track if value actually changed
            if old_value != value:
                # Special handling for different field types
                if field == 'images':
                    old_images = old_value or []
                    new_images = value or []
                    
                    if len(old_images) != len(new_images):
                        changes['images'] = {
                            'old': old_images,
                            'new': new_images,
                            'reason': 'Image(s) removed for policy compliance'
                        }
                elif field == 'tech_stack':
                    changes['tech_stack'] = {
                        'old': old_value or [],
                        'new': value or []
                    }
                elif field == 'budget':
                    changes['budget'] = {
                        'old': float(old_value) if old_value else 0,
                        'new': float(value) if value else 0
                    }
                elif field == 'description':
                    changes['description'] = {
                        'old': old_value or '',
                        'new': value or '',
                        'reason': 'Content updated for policy compliance'
                    }
                elif field == 'title':
                    changes['title'] = {
                        'old': old_value or '',
                        'new': value or ''
                    }
                elif field == 'status':
                    changes['status'] = {
                        'old': old_value or '',
                        'new': value or ''
                    }
                elif field == 'category':
                    changes['category'] = {
                        'old': old_value or '',
                        'new': value or ''
                    }
                elif field == 'location':
                    changes['location'] = {
                        'old': old_value or '',
                        'new': value or ''
                    }

    # Apply updates
    for field, value in project_data.dict(exclude_unset=True).items():
        if field == "deadline" and value:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        setattr(project, field, value)

    await db.commit()
    await db.refresh(project)

    # ✅ Send admin notification
    background_tasks.add_task(
        send_admin_project_notification,
        action="updated",
        project_id=str(project.id),
        project_title=project.title,
        developer_name=current_user.get('name', 'Unknown'),
        developer_email=current_user.get('email', 'Unknown'),
        project_data={
            'category': project.category,
            'budget': float(project.budget),
            'status': project.status,
            'location': project.location,
            'tech_stack': project.tech_stack
        }
    )

    # ✅ NEW: Send email to developer if admin edited their project
    if admin_editing_others_project and changes:
        # Get developer info
        developer = await db.scalar(
            select(User).where(User.id == project.developer_id)
        )
        
        if developer and developer.email:
            background_tasks.add_task(
                send_developer_edit_notification,
                developer_email=developer.email,
                developer_name=developer.name,
                project_title=project.title,
                project_id=str(project.id),
                admin_name=current_user.get('name', 'Admin'),
                changes=changes
            )
            
            # ✅ Also send in-app notification
            notif = Notification(
                user_id=developer.id,
                type="project_edited_by_admin",
                title="⚠️ Admin Updated Your Project",
                message=f"An administrator has made changes to your project '{project.title}'. Check your email for details.",
                link=f"/dashboard",
                details={
                    "project_id": str(project.id),
                    "project_title": project.title,
                    "admin_name": current_user.get('name', 'Admin'),
                    "changes_count": len(changes)
                },
                read=False
            )
            db.add(notif)
            await db.commit()

    return model_to_dict(project)

@app.delete("/api/projects/{project_id}")
async def delete_project(
    project_id: str,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    project = await db.scalar(select(Project).where(Project.id == uuid.UUID(project_id)))

    if not project:
        raise HTTPException(404, "Project not found")

    if str(project.developer_id) != str(current_user["id"]) and current_user["role"] != "admin":
        raise HTTPException(403, "Unauthorized")

    # ✅ CAPTURE DATA BEFORE DELETE
    project_title = project.title
    project_category = project.category
    project_budget = project.budget
    project_tech_stack = project.tech_stack or []

    await db.delete(project)
    await db.commit()

    # ✅ NEW: Send admin notification AFTER deletion
    background_tasks.add_task(
        send_admin_project_notification,
        action="deleted",
        project_id=project_id,
        project_title=project_title,
        developer_name=current_user.get('name', 'Unknown'),
        developer_email=current_user.get('email', 'Unknown'),
        project_data={
            'category': project_category,
            'budget': project_budget,
            'status': 'deleted',
            'location': 'N/A',
            'tech_stack': []
        }
    )


    return {"message": "Project deleted", "project_id": project_id}

@app.get("/api/projects/{project_id}")
async def get_project(project_id: str, db: AsyncSession = Depends(get_db)):
    project = await db.scalar(
        select(Project)
        .options(joinedload(Project.developer))
        .where(Project.id == uuid.UUID(project_id))
    )

    if not project:
        raise HTTPException(404, "Project not found")

    data = model_to_dict(project)
    data["developer"] = {
        "id": str(project.developer.id),
        "name": project.developer.name,
        "company": project.developer.company,
        "avatar": project.developer.avatar,
    }
    return data


@app.get("/api/projects/developer/{developer_id}")
async def get_developer_projects(
    developer_id: str,
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Project)
        .where(Project.developer_id == uuid.UUID(developer_id))
        .order_by(Project.created_at.desc())
        .offset(skip).limit(limit)
    )
    projects = result.scalars().all()

    # Check for chat rooms for each project
    projects_with_chat_info = []
    for project in projects:
        project_dict = model_to_dict(project)

        # Check if project has any chat rooms
        chat_room_result = await db.execute(
            select(ChatRoom).where(ChatRoom.project_id == project.id).limit(1)
        )
        has_chat_room = chat_room_result.scalar() is not None
        project_dict["has_chat_room"] = has_chat_room

        # Count active chat rooms for this project
        active_rooms_result = await db.execute(
            select(func.count(ChatRoom.id)).where(
                ChatRoom.project_id == project.id,
                ChatRoom.status == 'active'
            )
        )
        project_dict["active_rooms_count"] = active_rooms_result.scalar() or 0

        projects_with_chat_info.append(project_dict)

    return {
        "projects": projects_with_chat_info,
        "total": len(projects),
        "skip": skip,
        "limit": limit
    }


# ============================================
# BID ROUTES
# ============================================
@app.post("/api/projects/{project_id}/bids")
async def create_bid(
    project_id: str,
    bid_data: BidCreate,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    if current_user["role"] != "investor":
        raise HTTPException(403, "Only investors can place bids")

    project = await db.scalar(select(Project).where(Project.id == uuid.UUID(project_id)))
    if not project:
        raise HTTPException(404, "Project not found")

    existing_bid = await db.scalar(
        select(Bid).where(
            Bid.project_id == project.id,
            Bid.investor_id == current_user["id"]
        )
    )
    if existing_bid:
        raise HTTPException(400, "You already placed a bid")

    new_bid = Bid(
        project_id=project.id,
        investor_id=current_user["id"],
        amount=bid_data.amount,
        status="pending"
    )

    db.add(new_bid)
    await db.commit()
    await db.refresh(new_bid)

    # ✅ SINGLE notification - handles both DB and WebSocket
    formatted_amount = f"${bid_data.amount:,.2f}"
    
    await send_notification_to_user(
        str(project.developer_id),
        {
            "type": "bid_received",
            "title": "New Bid Received",
            "message": f"New bid of {formatted_amount} placed on your project '{project.title}'.",
            "link": f"/project/{project_id}/bids",
            "details": {
                "bid_id": str(new_bid.id),
                "project_id": project_id,
                "amount": float(bid_data.amount)
            }
        },
        db
    )

    return model_to_dict(new_bid)


@app.put("/api/bids/{bid_id}/accept")
async def accept_bid(
    bid_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    bid = await db.scalar(select(Bid).where(Bid.id == uuid.UUID(bid_id)))
    if not bid:
        raise HTTPException(404, "Bid not found")

    project = await db.scalar(select(Project).where(Project.id == bid.project_id))

    if str(project.developer_id) != str(current_user["id"]):
        raise HTTPException(403, "Only the project owner can accept bids")

    bid.status = "accepted"
    bid.accepted_at = datetime.utcnow()

    # Update project stats
    amount = float(bid.amount)
    project.highest_bid = max(amount, project.highest_bid or amount)
    project.lowest_bid = min(amount, project.lowest_bid or amount)
    project.bids_count = (project.bids_count or 0) + 1

    await db.commit()

    # ✅ SINGLE notification - handles both DB and WebSocket
    formatted_amount = f"${float(bid.amount):,.2f}"
    
    await send_notification_to_user(
        str(bid.investor_id),
        {
            "type": "bid_accepted",
            "title": "Your bid was accepted!",
            "message": f"Your bid of {formatted_amount} on '{project.title}' has been accepted.",
            "link": f"/project/{project.id}",
            "details": {
                "project_id": str(project.id),
                "bid_id": bid_id,
                "amount": float(bid.amount)
            }
        },
        db
    )

    return {"message": "Bid accepted"}


@app.put("/api/bids/{bid_id}/reject")
async def reject_bid(
    bid_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    bid = await db.scalar(select(Bid).where(Bid.id == uuid.UUID(bid_id)))
    if not bid:
        raise HTTPException(404, "Bid not found")

    project = await db.scalar(select(Project).where(Project.id == bid.project_id))

    if str(project.developer_id) != str(current_user["id"]):
        raise HTTPException(403, "Unauthorized")

    bid.status = "rejected"
    await db.commit()

    # ✅ SINGLE notification - handles both DB and WebSocket
    await send_notification_to_user(
        str(bid.investor_id),
        {
            "type": "bid_rejected",
            "title": "Bid Rejected",
            "message": f"Your bid on '{project.title}' was rejected. Don't give up - keep bidding on other projects!",
            "details": {
                "project_id": str(project.id),
                "bid_id": bid_id
            }
        },
        db
    )

    return {
        "message": "Bid rejected successfully",
        "bid_id": str(bid.id),
        "status": "rejected"
    }


@app.put("/api/projects/{project_id}/close-bidding")
async def close_bidding(
    project_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    project = await db.scalar(select(Project).where(Project.id == uuid.UUID(project_id)))
    if not project:
        raise HTTPException(404, "Project not found")

    if str(project.developer_id) != str(current_user["id"]):
        raise HTTPException(403, "Unauthorized")

    # Get accepted bids ordered highest first
    accepted_bids = await db.scalars(
        select(Bid).where(
            Bid.project_id == project.id,
            Bid.status == "accepted"
        ).order_by(Bid.amount.desc())
    )
    bids = list(accepted_bids)

    if not bids:
        raise HTTPException(400, "No accepted bids to finalize")

    # The highest accepted bid wins
    winner = bids[0]

    project.status = "winner_selected"
    project.assigned_to = winner.investor_id
    project.highest_bid = float(winner.amount)

    # Reject all other bids
    await db.execute(
        update(Bid)
        .where(Bid.project_id == project.id, Bid.id != winner.id)
        .values(status="rejected")
    )
    await db.commit()

    # Fetch winner's user profile
    winner_user = await db.scalar(
        select(User).where(User.id == winner.investor_id)
    )

    # ✅ SINGLE notification - handles both DB and WebSocket
    formatted_amount = f"${winner.amount:,.2f}"
    
    await send_notification_to_user(
        str(winner.investor_id),
        {
            "type": "payment_required",
            "title": "🎉 Congratulations! You Won the Project!",
            "message": (
                f"Your bid of {formatted_amount} on '{project.title}' has been selected! "
                f"Please complete the payment to start the project."
            ),
            "link": str(winner.id),
            "details": {
                "bid_id": str(winner.id),
                "project_id": str(project.id),
                "project_title": project.title,
                "amount": float(winner.amount),
                "bid_amount": float(winner.amount),
                "investor_id": str(winner.investor_id),
                "developer_id": str(project.developer_id),
                "payment_type": "project_winner",
                "action_required": "payment"
            }
        },
        db
    )

    # Return detailed info
    return {
        "message": "Bidding closed",
        "winner": str(winner.investor_id),
        "winner_name": winner_user.name if winner_user else None,
        "amount": float(winner.amount),
        "notification_sent": True
    }


@app.get("/api/projects/{project_id}/bids")
async def get_project_bids(project_id: str, db: AsyncSession = Depends(get_db)):
    bids = await db.scalars(
        select(Bid)
        .where(Bid.project_id == uuid.UUID(project_id))
        .order_by(Bid.created_at.desc())
    )
    return models_to_list(bids)


@app.get("/api/bids/investor/{investor_id}")
async def get_investor_bids(investor_id: str, db: AsyncSession = Depends(get_db)):
    bids = await db.scalars(
        select(Bid)
        .where(Bid.investor_id == uuid.UUID(investor_id))
        .order_by(Bid.created_at.desc())
    )
    return models_to_list(bids)


# ============================================
# Contact PAGE ENDPOINTS
# ============================================
async def send_email(
    to_email: str,
    subject: str,
    html_content: str,
    text_content: str = None
):
    """
    Generic function to send emails using SMTP
    
    Args:
        to_email: Recipient email address
        subject: Email subject
        html_content: HTML content of the email
        text_content: Plain text fallback (optional)
    
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM
        msg["To"] = to_email

        # Add plain text version if provided
        if text_content:
            msg.attach(MIMEText(text_content, "plain"))
        
        # Add HTML version
        msg.attach(MIMEText(html_content, "html"))

        # Send email
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(EMAIL_FROM, to_email, msg.as_string())
        server.quit()

        logger.info(f"📧 Email sent successfully to {to_email}")
        return True

    except Exception as e:
        logger.error(f"❌ Email sending failed to {to_email}: {str(e)}")
        return False


@app.post("/api/contact/submit")
async def submit_contact_form(
    form_data: ContactFormSubmission,
    db: AsyncSession = Depends(get_db)
):
    """
    Submit contact form - sends email to admin team
    """
    try:
        logger.info(f"Contact form submission from: {form_data.email}")
        
        # Category display names
        category_names = {
            'general': 'General Inquiry',
            'technical': 'Technical Support',
            'billing': 'Billing & Payments',
            'partnership': 'Partnership',
            'feedback': 'Feedback'
        }
        
        category_display = category_names.get(form_data.category, form_data.category.title())
        
        # Prepare email content for admin
        subject = f"[Contact Form] {category_display}: {form_data.subject}"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px 10px 0 0;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 24px;
                }}
                .content {{
                    background: #f8f9fa;
                    padding: 30px;
                    border: 1px solid #e0e0e0;
                    border-top: none;
                }}
                .field {{
                    margin-bottom: 20px;
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid #667eea;
                }}
                .field-label {{
                    font-weight: bold;
                    color: #667eea;
                    margin-bottom: 5px;
                    font-size: 12px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }}
                .field-value {{
                    color: #333;
                    font-size: 15px;
                }}
                .message-box {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid #667eea;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                }}
                .category-badge {{
                    display: inline-block;
                    background: #667eea;
                    color: white;
                    padding: 5px 15px;
                    border-radius: 20px;
                    font-size: 13px;
                    font-weight: 600;
                }}
                .footer {{
                    background: #f8f9fa;
                    padding: 20px;
                    text-align: center;
                    border-radius: 0 0 10px 10px;
                    border: 1px solid #e0e0e0;
                    border-top: none;
                    color: #666;
                    font-size: 13px;
                }}
                .reply-button {{
                    display: inline-block;
                    background: #667eea;
                    color: white;
                    padding: 12px 30px;
                    text-decoration: none;
                    border-radius: 6px;
                    margin-top: 15px;
                    font-weight: 600;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📧 New Contact Form Submission</h1>
            </div>
            
            <div class="content">
                <div class="field">
                    <div class="field-label">Category</div>
                    <div class="field-value">
                        <span class="category-badge">{category_display}</span>
                    </div>
                </div>
                
                <div class="field">
                    <div class="field-label">From</div>
                    <div class="field-value">{form_data.name}</div>
                </div>
                
                <div class="field">
                    <div class="field-label">Email</div>
                    <div class="field-value">
                        <a href="mailto:{form_data.email}" style="color: #667eea; text-decoration: none;">
                            {form_data.email}
                        </a>
                    </div>
                </div>
                
                <div class="field">
                    <div class="field-label">Subject</div>
                    <div class="field-value">{form_data.subject}</div>
                </div>
                
                <div style="margin-top: 20px;">
                    <div class="field-label" style="margin-bottom: 10px;">Message</div>
                    <div class="message-box">
{form_data.message}
                    </div>
                </div>
                
                <div style="text-align: center; margin-top: 25px;">
                    <a href="mailto:{form_data.email}?subject=Re: {form_data.subject}" class="reply-button">
                        Reply to {form_data.name}
                    </a>
                </div>
            </div>
            
            <div class="footer">
                <p>This message was sent via the BiteBids contact form</p>
                <p style="margin: 5px 0; color: #999; font-size: 12px;">
                    Received on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
                </p>
            </div>
        </body>
        </html>
        """
        
        # Plain text version
        text_content = f"""
New Contact Form Submission
==========================

Category: {category_display}
From: {form_data.name}
Email: {form_data.email}
Subject: {form_data.subject}

Message:
{form_data.message}

---
Reply to: {form_data.email}
Received: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
        """
        
        # Send email to admin team
        # IMPORTANT: Change this to your actual admin email
        admin_email = os.getenv('ADMIN_EMAIL', 'bitebids@gmail.com')
        
        email_sent = await send_email(
            to_email=admin_email,
            subject=subject,
            html_content=html_content,
            text_content=text_content
        )
        
        if not email_sent:
            raise HTTPException(
                status_code=500,
                detail="Failed to send email. Please try again later."
            )
        
        # Send auto-reply to user
        try:
            auto_reply_subject = f"We received your message: {form_data.subject}"
            auto_reply_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 600px;
                        margin: 0 auto;
                        padding: 20px;
                    }}
                    .header {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 30px;
                        border-radius: 10px 10px 0 0;
                        text-align: center;
                    }}
                    .content {{
                        background: #f8f9fa;
                        padding: 30px;
                        border: 1px solid #e0e0e0;
                    }}
                    .footer {{
                        background: #f8f9fa;
                        padding: 20px;
                        text-align: center;
                        color: #666;
                        font-size: 13px;
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>✅ Thank You for Contacting BiteBids!</h1>
                </div>
                <div class="content">
                    <p>Hi {form_data.name},</p>
                    
                    <p>Thank you for reaching out to us! We've received your message and our team will review it shortly.</p>
                    
                    <p><strong>Your submission details:</strong></p>
                    <ul>
                        <li><strong>Category:</strong> {category_display}</li>
                        <li><strong>Subject:</strong> {form_data.subject}</li>
                    </ul>
                    
                    <p>We typically respond within 24 hours during business days (Monday-Friday, 9 AM - 6 PM EST).</p>
                    
                    <p>If your inquiry is urgent, you can also reach us at:</p>
                    <ul>
                        <li>📧 Email: bitebids@gmail.com</li>
                        <li>📞 Phone: +1 (555) 123-4567</li>
                    </ul>
                    
                    <p>Best regards,<br>The BiteBids Team</p>
                </div>
                <div class="footer">
                    <p>This is an automated message. Please do not reply to this email.</p>
                </div>
            </body>
            </html>
            """
            
            auto_reply_text = f"""
Hi {form_data.name},

Thank you for reaching out to BiteBids!

We've received your message and will respond within 24 hours during business days (Monday-Friday, 9 AM - 6 PM EST).

Your submission details:
- Category: {category_display}
- Subject: {form_data.subject}

If your inquiry is urgent, contact us at:
📧 bitebids@gmail.com
📞 +1 (555) 123-4567

Best regards,
The BiteBids Team
            """
            
            await send_email(
                to_email=form_data.email,
                subject=auto_reply_subject,
                html_content=auto_reply_html,
                text_content=auto_reply_text
            )
        except Exception as e:
            logger.error(f"Failed to send auto-reply: {e}")
            # Don't fail the whole request if auto-reply fails
        
        logger.info(f"Contact form processed successfully for {form_data.email}")
        
        return {
            "success": True,
            "message": "Thank you for your message! We'll get back to you within 24 hours."
        }
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing contact form: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your message. Please try again later."
        )


# ============================================
# NOTIFICATION ENDPOINTS
# ============================================

# ✅ UPDATE: API endpoint to return metadata
@app.get("/api/notifications/{user_id}")
async def get_user_notifications(
    user_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get all notifications for a specific user
    Returns notifications sorted by creation date (newest first)
    """
    try:
        user_uuid = uuid.UUID(user_id)
        
        result = await db.execute(
            select(Notification)
            .where(Notification.user_id == user_uuid)
            .order_by(Notification.created_at.desc())
            .limit(50)
        )
        notifications = result.scalars().all()
        
        unread_result = await db.execute(
            select(func.count(Notification.id))
            .where(
                and_(
                    Notification.user_id == user_uuid,
                    Notification.read == False
                )
            )
        )
        unread_count = unread_result.scalar()
        
        # ✅ FIX: Include details in response (renamed from metadata)
        notification_list = [
            {
                "id": str(notif.id),
                "type": notif.type,
                "title": notif.title,
                "message": notif.message,
                "link": notif.link,
                "metadata": notif.details or {},  # ✅ Return as 'metadata' for frontend compatibility
                "read": notif.read,
                "read_at": notif.read_at.isoformat() if notif.read_at else None,
                "created_at": notif.created_at.isoformat() if notif.created_at else None,
            }
            for notif in notifications
        ]
        
        return {
            "success": True,
            "notifications": notification_list,
            "unread_count": unread_count or 0
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    except Exception as e:
        logger.error(f"Error fetching notifications: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch notifications")


@app.put("/api/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Mark a specific notification as read
    """
    try:
        # Convert string to UUID
        notif_uuid = uuid.UUID(notification_id)
        
        # Update notification
        result = await db.execute(
            update(Notification)
            .where(Notification.id == notif_uuid)
            .values(
                read=True,
                read_at=datetime.now()
            )
            .returning(Notification.id)
        )
        
        await db.commit()
        
        updated = result.scalar_one_or_none()
        
        if not updated:
            raise HTTPException(status_code=404, detail="Notification not found")
        
        return {
            "success": True,
            "message": "Notification marked as read"
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid notification ID format")
    except Exception as e:
        logger.error(f"Error marking notification as read: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update notification")


@app.put("/api/notifications/{user_id}/read-all")
async def mark_all_notifications_read(
    user_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Mark all notifications as read for a specific user
    """
    try:
        # Convert string to UUID
        user_uuid = uuid.UUID(user_id)
        
        # Update all unread notifications for user
        await db.execute(
            update(Notification)
            .where(
                and_(
                    Notification.user_id == user_uuid,
                    Notification.read == False
                )
            )
            .values(
                read=True,
                read_at=datetime.now()
            )
        )
        
        await db.commit()
        
        return {
            "success": True,
            "message": "All notifications marked as read"
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    except Exception as e:
        logger.error(f"Error marking all notifications as read: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update notifications")


@app.post("/api/notifications/create")
async def create_notification(
    user_id: str,
    notification_type: str,
    title: str,
    message: str,
    link: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,  # ✅ ADD details parameter
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new notification for a user
    This endpoint can be used internally to create notifications
    """
    try:
        # Convert string to UUID
        user_uuid = uuid.UUID(user_id)
        
        # Create notification
        new_notification = Notification(
            user_id=user_uuid,
            type=notification_type,
            title=title,
            message=message,
            link=link,
            details=details,  # ✅ ADD details
            read=False
        )
        
        db.add(new_notification)
        await db.commit()
        await db.refresh(new_notification)
        
        return {
            "success": True,
            "message": "Notification created successfully",
            "notification_id": str(new_notification.id)
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    except Exception as e:
        logger.error(f"Error creating notification: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create notification")


@app.delete("/api/notifications/{notification_id}")
async def delete_notification(
    notification_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a specific notification
    """
    try:
        # Convert string to UUID
        notif_uuid = uuid.UUID(notification_id)
        
        # Delete notification
        result = await db.execute(
            delete(Notification)
            .where(Notification.id == notif_uuid)
            .returning(Notification.id)
        )
        
        await db.commit()
        
        deleted = result.scalar_one_or_none()
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Notification not found")
        
        return {
            "success": True,
            "message": "Notification deleted successfully"
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid notification ID format")
    except Exception as e:
        logger.error(f"Error deleting notification: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete notification")


# ============================================
# DISPUTE CREATION ENDPOINT
# Add this to your server.py file
# ============================================

# Investor Approved the project email template
async def send_project_approved_email(
    to_email: str,
    developer_name: str,
    project_title: str,
    amount: float,
    platform_fee: float,
    gross_amount: float
):
    """
    Send project approval email to developer
    ✅ FIXED: Correct email specifically for project approval (not "Payment Received")
    Add this function near your other email functions in server.py
    """
    try:
        subject = f"✅ Project Approved: '{project_title}'"

        safe_amount = float(amount or 0)
        safe_fee = float(platform_fee or 0)
        safe_gross = float(gross_amount or 0)

        html_content = f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <title>Project Approved - BiteBids</title>
        </head>
        <body style="margin:0; padding:0; background:#f8fafc; font-family:'Segoe UI', Arial, sans-serif;">
            <table width="100%" border="0" cellspacing="0" cellpadding="0"
                   style="background:#f8fafc; padding:40px 0;">
                <tr>
                    <td align="center">
                        <table width="600" border="0" cellspacing="0" cellpadding="0"
                               style="background:white; border-radius:16px; box-shadow:0 4px 20px rgba(0,0,0,0.06); padding:40px;">
                            
                            <!-- Header -->
                            <tr>
                                <td align="center" style="padding-bottom:20px;">
                                    <h1 style="margin:0; font-size:26px; font-weight:700; color:#4f46e5;">
                                        BiteBids
                                    </h1>
                                    <p style="margin:8px 0 0; color:#6b7280; font-size:14px;">
                                        Project Approval Confirmation
                                    </p>
                                </td>
                            </tr>

                            <!-- Success Icon -->
                            <tr>
                                <td align="center" style="padding:20px 0;">
                                    <div style="width:80px; height:80px; background:#d1fae5; border-radius:50%; 
                                                display:flex; align-items:center; justify-content:center; margin:0 auto;">
                                        <span style="font-size:40px;">🎉</span>
                                    </div>
                                </td>
                            </tr>

                            <!-- Main Message -->
                            <tr>
                                <td style="padding-bottom:20px;">
                                    <h2 style="margin:0; font-size:20px; color:#111827; font-weight:600; text-align:center;">
                                        Congratulations, {developer_name}!
                                    </h2>
                                    <p style="margin:15px 0 0; color:#374151; font-size:15px; line-height:1.6; text-align:center;">
                                        Your project has been approved by the investor!
                                    </p>
                                </td>
                            </tr>

                            <!-- Project Details -->
                            <tr>
                                <td style="padding-bottom:20px;">
                                    <div style="background:#f9fafb; border-radius:8px; padding:20px; border-left:4px solid #10b981;">
                                        <p style="margin:0 0 10px 0; color:#111827; font-size:15px; font-weight:600;">
                                            Project: <span style="color:#4f46e5;">{project_title}</span>
                                        </p>
                                        
                                        <table width="100%" style="margin-top:15px;">
                                            <tr>
                                                <td style="padding:8px 0; color:#6b7280; font-size:14px;">
                                                    <strong>Gross Amount:</strong>
                                                </td>
                                                <td style="padding:8px 0; color:#111827; font-size:14px; text-align:right;">
                                                    ${safe_gross:,.2f}
                                                </td>
                                            </tr>
                                            <tr>
                                                <td style="padding:8px 0; color:#6b7280; font-size:14px;">
                                                    <strong>Platform Fee (6%):</strong>
                                                </td>
                                                <td style="padding:8px 0; color:#ef4444; font-size:14px; text-align:right;">
                                                    -${safe_fee:,.2f}
                                                </td>
                                            </tr>
                                            <tr style="border-top:2px solid #e5e7eb;">
                                                <td style="padding:12px 0 0 0; color:#111827; font-size:16px;">
                                                    <strong>Your Payout:</strong>
                                                </td>
                                                <td style="padding:12px 0 0 0; color:#10b981; font-size:18px; font-weight:700; text-align:right;">
                                                    ${safe_amount:,.2f}
                                                </td>
                                            </tr>
                                        </table>
                                    </div>
                                </td>
                            </tr>

                            <!-- What's Next -->
                            <tr>
                                <td style="padding-bottom:20px;">
                                    <div style="background:#dbeafe; border-radius:8px; padding:16px; border-left:4px solid #3b82f6;">
                                        <p style="margin:0 0 8px 0; font-size:14px; color:#1e40af; font-weight:600;">
                                            💰 What's Next:
                                        </p>
                                        <ul style="margin:0; padding-left:20px; color:#1e3a8a; font-size:13px; line-height:1.6;">
                                            <li>Payment is being processed by BiteBids</li>
                                            <li>Funds will be transferred to your account within 3-5 business days</li>
                                            <li>You'll receive another email once the transfer is complete</li>
                                        </ul>
                                    </div>
                                </td>
                            </tr>

                            <!-- CTA Button -->
                            <tr>
                                <td align="center" style="padding:20px 0;">
                                    <a href="{FRONTEND_URL}/dashboard" 
                                       style="display:inline-block; background:#4f46e5; color:white; 
                                              padding:14px 32px; border-radius:8px; text-decoration:none; 
                                              font-weight:600; font-size:15px;">
                                        View Dashboard →
                                    </a>
                                </td>
                            </tr>

                            <!-- Footer -->
                            <tr>
                                <td align="center" style="padding-top:20px; border-top:1px solid #e5e7eb;">
                                    <p style="margin:0; font-size:12px; color:#9ca3af;">
                                        This is an automated message from BiteBids<br/>
                                        © 2024 BiteBids. All rights reserved.
                                    </p>
                                </td>
                            </tr>

                        </table>
                    </td>
                </tr>
            </table>
        </body>
        </html>
        """

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM
        msg["To"] = to_email

        msg.attach(MIMEText(html_content, "html"))

        # Send email
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(EMAIL_FROM, to_email, msg.as_string())
        server.quit()

        logger.info(f"📧 Project approval email sent to {to_email}")

    except Exception as e:
        logger.error(f"❌ Failed to send project approval email: {str(e)}")


@app.post("/api/projects/{project_id}/simple-approve")
async def simple_approve_project(
    project_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Simplified project approval - investor confirms completion
    ✅ SIMPLIFIED: No backend dispute check (handled by frontend)
    ✅ FIXED: Sends correct "Project Approved" email
    ✅ FIXED: Creates notification and system message properly
    """
    try:
        # Get project
        project_query = await db.execute(
            select(Project).where(Project.id == uuid.UUID(project_id))
        )
        project = project_query.scalar_one_or_none()
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Verify project is active (fixed_price or in_progress)
        if project.status not in ['fixed_price', 'in_progress']:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot confirm project with status '{project.status}'. Project must be in progress."
            )
        
        # Calculate payout
        project_amount = float(project.budget)
        platform_fee_percentage = 6.0  # 6%
        platform_fee = project_amount * (platform_fee_percentage / 100)
        developer_payout = project_amount - platform_fee
        
        logger.info(f"Approving project {project_id}: amount=${project_amount}, fee=${platform_fee}, payout=${developer_payout}")
        
        # Store original status
        original_status = project.status
        
        # ✅ IMPORTANT: Only change status for auction projects (in_progress)
        # For fixed_price projects, keep status as 'fixed_price' (multiple investors can buy)
        if original_status == 'in_progress':
            project.status = 'completed'  # Auction project → completed
            logger.info(f"Auction project {project_id} marked as completed")
        else:
            # Keep fixed_price status unchanged
            logger.info(f"Fixed-price project {project_id} keeps status '{project.status}'")
        
        project.completed_at = datetime.utcnow()
        
        # Get developer user
        developer_query = await db.execute(
            select(User).where(User.id == project.developer_id)
        )
        developer = developer_query.scalar_one()
        
        # Update developer stats
        developer.projects_completed = (developer.projects_completed or 0) + 1
        developer.total_earnings = (developer.total_earnings or 0) + Decimal(str(developer_payout))

        # ✅ CREATE PAYOUT RECORD
        user_uuid = uuid.UUID(current_user['id'])

        # Find the checkout session for this project and investor
        checkout_result = await db.execute(
            select(CheckoutSession).where(
                and_(
                    CheckoutSession.project_id == project.id,
                    CheckoutSession.customer_id == user_uuid,
                    CheckoutSession.status == 'completed'
                )
            ).order_by(CheckoutSession.completed_at.desc())
        )
        checkout_session = checkout_result.scalar_one_or_none()

        # Create payout record
        payout_record = DeveloperPayout(
            developer_id=project.developer_id,
            project_id=project.id,
            checkout_session_id=checkout_session.id if checkout_session else None,
            investor_id=user_uuid,
            gross_amount=Decimal(str(project_amount)),
            platform_fee=Decimal(str(platform_fee)),
            net_amount=Decimal(str(developer_payout)),
            currency='USD',
            payout_method=developer.payout_method,
            payout_email=developer.payout_email,
            payout_details=developer.payout_details,
            status='pending',
            description=f"Payment for project: {project.title}"
        )
        db.add(payout_record)
        logger.info(f"✅ Payout record created for developer {developer.email}: ${developer_payout:.2f}")

        # ✅ FIXED: Create proper notification for developer
        dev_notification = Notification(
            user_id=project.developer_id,
            type='project_approved',
            title='🎉 Project Approved!',
            message=f'Your project "{project.title}" has been approved by the investor. Payment of ${developer_payout:.2f} is being processed.',
            link=f"/projects/{project_id}",
            details={
                'project_id': str(project.id),
                'project_title': project.title,
                'amount': float(developer_payout),
                'platform_fee': float(platform_fee),
                'gross_amount': float(project_amount)
            },
            read=False
        )
        db.add(dev_notification)
        
        # ✅ FIXED: Send system message to chat (with proper room lookup)
        try:
            user_uuid = uuid.UUID(current_user['id'])
            
            # Get the chat room for THIS investor
            room_query = await db.execute(
                select(ChatRoom).where(
                    and_(
                        ChatRoom.project_id == project.id,
                        ChatRoom.investor_id == user_uuid
                    )
                )
            )
            room = room_query.scalar_one_or_none()
            
            if room:
                system_message = ChatMessage(
                    room_id=room.id,
                    sender_id=project.developer_id,
                    message=f"✅ Project Completed!\n\n" +
                           f"Investor has confirmed project completion.\n" +
                           f"Payment Details:\n" +
                           f"• Gross Amount: ${project_amount:.2f}\n" +
                           f"• Platform Fee (6%): ${platform_fee:.2f}\n" +
                           f"• Developer Payout: ${developer_payout:.2f}\n\n" +
                           f"Payment is being processed and will be transferred to the developer's account.",
                    message_type='system',
                    created_at=datetime.utcnow()
                )
                db.add(system_message)
                logger.info(f"✅ System message added to chat room {room.id}")
            else:
                logger.warning(f"⚠️ No chat room found for project {project_id} and investor {user_uuid}")
        except Exception as e:
            logger.error(f"Error adding system message to chat: {e}")
        
        # Commit all changes
        await db.commit()
        await db.refresh(project)
        await db.refresh(dev_notification)
        
        # ✅ FIXED: Send live notification via WebSocket
        try:
            await send_notification_to_user(
                str(project.developer_id),
                {
                    "type": "project_approved",
                    "title": "🎉 Project Approved!",
                    "message": f"Your project \"{project.title}\" has been approved! Payment of ${developer_payout:.2f} is being processed.",
                    "link": f"/projects/{project_id}",
                    "details": {
                        'project_id': str(project.id),
                        'project_title': project.title,
                        'amount': float(developer_payout),
                        'platform_fee': float(platform_fee),
                        'gross_amount': float(project_amount)
                    }
                },
                db
            )
            logger.info(f"✅ Live notification sent to developer {project.developer_id}")
        except Exception as e:
            logger.error(f"Failed to send live notification: {e}")
        
        # ✅ FIXED: Send correct "Project Approved" email (not "Payment Received")
        try:
            asyncio.create_task(
                send_project_approved_email(
                    to_email=developer.email,
                    developer_name=developer.name,
                    project_title=project.title,
                    amount=developer_payout,
                    platform_fee=platform_fee,
                    gross_amount=project_amount
                )
            )
            logger.info(f"✅ Project approval email queued for {developer.email}")
        except Exception as e:
            logger.error(f"Failed to queue approval email: {e}")
        
        # Log activity
        try:
            activity = ActivityLog(
                user_id=current_user['id'],
                user_name=current_user['name'],
                type='project_approval',
                action='project_approved',
                details={
                    'project_id': str(project.id),
                    'project_title': project.title,
                    'developer_id': str(project.developer_id),
                    'amount': float(project_amount),
                    'payout': float(developer_payout)
                }
            )
            db.add(activity)
            await db.commit()
        except Exception as e:
            logger.error(f"Failed to log activity: {e}")
        
        logger.info(f"✅ Project {project_id} approved successfully")
        
        return {
            "success": True,
            "message": "Project approved successfully! Payment is being processed.",
            "project_id": str(project.id),
            "project_status": project.status,
            "developer_payout": float(developer_payout),
            "platform_commission": float(platform_fee),
            "gross_amount": float(project_amount),
            "notification_sent": True,
            "email_sent": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error approving project: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to approve project: {str(e)}"
        )


@app.post("/api/projects/{project_id}/dispute/create")
async def create_project_dispute(
    project_id: str,
    data: dict,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a dispute for a project
    
    Request body:
    - reason: str (required)
    - notes: str (optional)
    - investor_id: str (optional - required if developer has multiple investors)
    """
    try:
        # Get project
        project_query = await db.execute(
            select(Project).where(Project.id == uuid.UUID(project_id))
        )
        project = project_query.scalar_one_or_none()
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Verify project is active
        if project.status not in ['fixed_price', 'in_progress', 'disputed']:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot dispute project with status '{project.status}'"
            )
        
        # ✅ Determine who's opening the dispute
        user_uuid = uuid.UUID(current_user['id'])
        is_developer = user_uuid == project.developer_id
        
        # ✅ CRITICAL FIX: Handle investor_id properly
        if is_developer:
            # Developer opening dispute - need to know which investor
            investor_id_str = data.get('investor_id')
            
            if not investor_id_str:
                # ✅ Check how many chat rooms exist
                rooms_query = await db.execute(
                    select(ChatRoom).where(
                        ChatRoom.project_id == project.id,
                        ChatRoom.developer_id == user_uuid
                    )
                )
                rooms = rooms_query.scalars().all()
                
                if len(rooms) > 1:
                    # Multiple investors - need to specify which one
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "code": "INVESTOR_ID_REQUIRED",
                            "message": "Multiple investors have purchased this project. Please specify which investor.",
                            "investors": [
                                {
                                    "investor_id": str(room.investor_id),
                                    "room_id": str(room.id)
                                }
                                for room in rooms
                            ]
                        }
                    )
                elif len(rooms) == 1:
                    # Only one investor - use that one
                    investor_id = rooms[0].investor_id
                    room = rooms[0]
                else:
                    # No rooms found
                    raise HTTPException(
                        status_code=400,
                        detail="No chat room found. Cannot open dispute."
                    )
            else:
                # investor_id was provided
                try:
                    investor_id = uuid.UUID(investor_id_str)
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid investor_id format")
                
                # Get the specific room for this investor
                room_query = await db.execute(
                    select(ChatRoom).where(
                        and_(
                            ChatRoom.project_id == project.id,
                            ChatRoom.developer_id == user_uuid,
                            ChatRoom.investor_id == investor_id
                        )
                    )
                )
                room = room_query.scalar_one_or_none()
                
                if not room:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No chat room found with investor {investor_id_str}"
                    )
        else:
            # Investor opening dispute - investor_id is current user
            investor_id = user_uuid
            
            # Get their chat room
            room_query = await db.execute(
                select(ChatRoom).where(
                    and_(
                        ChatRoom.project_id == project.id,
                        ChatRoom.investor_id == investor_id
                    )
                )
            )
            room = room_query.scalar_one_or_none()
            
            if not room:
                raise HTTPException(
                    status_code=400,
                    detail="No chat room found. You must have an active chat to open a dispute."
                )
        
        # Verify investor exists
        investor_check_query = await db.execute(
            select(User).where(User.id == investor_id)
        )
        investor_check = investor_check_query.scalar_one_or_none()
        if not investor_check:
            raise HTTPException(status_code=404, detail="Investor not found")
        
        # ✅ Check for existing active dispute with THIS investor
        existing_dispute_query = await db.execute(
            select(ProjectDisputeSimple).where(
                and_(
                    ProjectDisputeSimple.project_id == uuid.UUID(project_id),
                    ProjectDisputeSimple.investor_id == investor_id,
                    ProjectDisputeSimple.resolved == False
                )
            )
        )
        active_dispute = existing_dispute_query.scalar_one_or_none()
        
        if active_dispute:
            raise HTTPException(
                status_code=400,
                detail="An active dispute already exists with this investor."
            )
        
        # Get request data
        reason = data.get('reason', '').strip()
        notes = data.get('notes', '').strip()
        
        if not reason:
            raise HTTPException(status_code=400, detail="Dispute reason is required")
        
        dispute_opener = "developer" if is_developer else "investor"
        logger.info(f"Creating dispute for project {project_id}: investor={investor_id}, opened_by={dispute_opener}")
        
        # Store old status
        old_status = project.status
        
        # Only change status to 'disputed' for auction projects
        if project.status == 'in_progress':
            project.status = 'disputed'
            logger.info(f"Changed auction project status to 'disputed'")
        else:
            logger.info(f"Keeping fixed_price project status as '{project.status}'")
        
        # Create dispute record
        dispute_record = ProjectDisputeSimple(
            project_id=uuid.UUID(project_id),
            reason=reason,
            notes=notes,
            disputed_by=user_uuid,
            investor_id=investor_id,
            disputed_at=datetime.utcnow(),
            previous_status=old_status,
            resolved=False
        )
        db.add(dispute_record)
        
        # Get developer and investor details
        developer_query = await db.execute(
            select(User).where(User.id == project.developer_id)
        )
        developer = developer_query.scalar_one()
        
        investor_query = await db.execute(
            select(User).where(User.id == investor_id)
        )
        investor = investor_query.scalar_one()
        
        # Notify the other party
        other_party_id = investor_id if is_developer else project.developer_id
        other_party_notification = Notification(
            user_id=other_party_id,
            type='dispute_opened',
            title='⚠️ Dispute Opened',
            message=f'The {dispute_opener} has opened a dispute for project "{project.title}". Reason: {reason}.',
            link=f"/projects/{project_id}",
            details={
                'project_id': str(project.id),
                'dispute_id': str(dispute_record.id),
                'reason': reason,
                'opened_by': dispute_opener
            },
            read=False
        )
        db.add(other_party_notification)
        
        # Add system message to chat
        try:
            notes_part = f"Notes: {notes}\n\n" if notes else ""
            system_message_text = (
                f"⚠️ Dispute Opened\n\n"
                f"The {dispute_opener} has opened a dispute.\n\n"
                f"Reason: {reason}\n"
                f"{notes_part}"
                f"An admin will review this case and make a fair decision."
            )
            
            system_message = ChatMessage(
                room_id=room.id,
                sender_id=project.developer_id,
                message=system_message_text,
                message_type='system',
                created_at=datetime.utcnow()
            )
            db.add(system_message)
            logger.info(f"System message added to chat room {room.id}")
        except Exception as e:
            logger.error(f"Error adding system message: {e}")
        
        # Commit all changes
        await db.commit()

        logger.info(f"✅ Dispute created successfully for project {project_id}")

        # 📧 Send email notification to admin
        try:
            admin_email = os.getenv('ADMIN_EMAIL', 'bitebids@gmail.com')

            subject = f"🚨 New Dispute Opened - Project: {project.title}"

            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 600px;
                        margin: 0 auto;
                        padding: 20px;
                    }}
                    .header {{
                        background: linear-gradient(135deg, #ef4444, #dc2626);
                        color: white;
                        padding: 30px;
                        border-radius: 10px 10px 0 0;
                        text-align: center;
                    }}
                    .content {{
                        background: #f8fafc;
                        padding: 30px;
                        border-radius: 0 0 10px 10px;
                    }}
                    .detail-box {{
                        background: white;
                        padding: 20px;
                        margin: 15px 0;
                        border-radius: 8px;
                        border-left: 4px solid #ef4444;
                    }}
                    .label {{
                        font-weight: 600;
                        color: #64748b;
                        margin-bottom: 5px;
                    }}
                    .value {{
                        color: #1e293b;
                        margin-bottom: 15px;
                    }}
                    .reason-box {{
                        background: #fef2f2;
                        padding: 15px;
                        border-radius: 8px;
                        border-left: 4px solid #ef4444;
                        margin: 15px 0;
                    }}
                    .button {{
                        display: inline-block;
                        padding: 12px 30px;
                        background: #ef4444;
                        color: white;
                        text-decoration: none;
                        border-radius: 8px;
                        margin-top: 20px;
                        font-weight: 600;
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1 style="margin: 0; font-size: 24px;">🚨 New Dispute Opened</h1>
                    <p style="margin: 10px 0 0 0; opacity: 0.9;">Immediate admin attention required</p>
                </div>

                <div class="content">
                    <div class="detail-box">
                        <div class="label">Project:</div>
                        <div class="value"><strong>{project.title}</strong></div>

                        <div class="label">Project ID:</div>
                        <div class="value">{project_id}</div>

                        <div class="label">Disputed By:</div>
                        <div class="value"><strong>{dispute_opener.title()}</strong></div>

                        <div class="label">Developer:</div>
                        <div class="value">{developer.name} ({developer.email})</div>

                        <div class="label">Investor:</div>
                        <div class="value">{investor.name} ({investor.email})</div>
                    </div>

                    <div class="reason-box">
                        <div class="label">Dispute Reason:</div>
                        <div class="value">{reason}</div>

                        {f'<div class="label">Additional Notes:</div><div class="value">{notes}</div>' if notes else ''}
                    </div>

                    <p style="color: #64748b; margin-top: 20px;">
                        Please review this dispute and take appropriate action in the admin panel.
                    </p>

                    <a href="https://bite-bids.vercel.app/admin/disputes" class="button">
                        Review Dispute
                    </a>
                </div>
            </body>
            </html>
            """

            # Send email asynchronously (don't block the response)
            await send_email(admin_email, subject, html_content)
            logger.info(f"📧 Dispute notification email sent to admin")

        except Exception as email_error:
            logger.error(f"Failed to send dispute notification email: {email_error}")
            # Don't fail the request if email fails

        return {
            "success": True,
            "message": "Dispute created successfully. An admin will review your case.",
            "dispute_id": str(dispute_record.id),
            "project_id": str(project.id),
            "investor_id": str(investor_id),
            "developer_id": str(project.developer_id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error creating dispute: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create dispute: {str(e)}")


@app.get("/api/admin/disputes-simple")
async def get_simple_disputes(
    admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all active disputes
    UPDATED: Shows all unresolved disputes (can be multiple per project for fixed_price)
    UPDATED: Includes who opened each dispute (developer or investor)
    """
    try:
        # ✅ CHANGED: Get all ACTIVE (unresolved) disputes instead of disputed projects
        disputes_query = await db.execute(
            select(ProjectDisputeSimple)
            .where(ProjectDisputeSimple.resolved == False)
            .order_by(ProjectDisputeSimple.disputed_at.desc())
        )
        dispute_records = disputes_query.scalars().all()
        
        disputes = []
        for dispute_record in dispute_records:
            try:
                # Get project
                project_query = await db.execute(
                    select(Project).where(Project.id == dispute_record.project_id)
                )
                project = project_query.scalar_one_or_none()
                
                if not project:
                    logger.warning(f"Skipping dispute {dispute_record.id} - project not found")
                    continue
                
                # Get developer
                developer_query = await db.execute(
                    select(User).where(User.id == project.developer_id)
                )
                developer = developer_query.scalar_one_or_none()
                
                if not developer:
                    logger.warning(f"Skipping dispute {dispute_record.id} - developer not found")
                    continue
                
                # ✅ UPDATED: Get investor from dispute record
                investor_query = await db.execute(
                    select(User).where(User.id == dispute_record.investor_id)
                )
                investor = investor_query.scalar_one_or_none()
                
                if not investor:
                    logger.warning(f"Skipping dispute {dispute_record.id} - investor not found")
                    continue
                
                # Get the user who opened the dispute
                dispute_opener_query = await db.execute(
                    select(User).where(User.id == dispute_record.disputed_by)
                )
                dispute_opener = dispute_opener_query.scalar_one_or_none()
                
                if not dispute_opener:
                    logger.warning(f"Skipping dispute {dispute_record.id} - dispute opener not found")
                    continue
                
                # Determine if dispute was opened by developer or investor
                is_developer_dispute = dispute_record.disputed_by == project.developer_id
                opened_by_role = "developer" if is_developer_dispute else "investor"
                
                disputes.append({
                    "id": str(project.id),
                    "dispute_id": str(dispute_record.id),  # ✅ NEW: Include dispute ID for resolution
                    "project": {
                        "id": str(project.id),
                        "title": project.title,
                        "amount": float(project.budget),
                        "status": project.status,  # ✅ NEW: Show if it's fixed_price or disputed
                        "previous_status": dispute_record.previous_status  # ✅ NEW: Show original status
                    },
                    "developer": {
                        "id": str(developer.id),
                        "name": developer.name,
                        "email": developer.email
                    },
                    "investor": {
                        "id": str(investor.id),
                        "name": investor.name,
                        "email": investor.email
                    },
                    "dispute_reason": dispute_record.reason or 'Unknown',
                    "dispute_notes": dispute_record.notes or '',
                    "disputed_at": dispute_record.disputed_at.isoformat() if dispute_record.disputed_at else None,
                    "opened_by": opened_by_role,
                    "opened_by_name": dispute_opener.name,
                    "platform_commission": float(project.budget) * 0.06,
                    "developer_payout": float(project.budget) - (float(project.budget) * 0.06),
                    "delivery_url": None,
                    "delivery_notes": ""
                })
                
            except Exception as e:
                logger.error(f"Error processing dispute {dispute_record.id}: {e}", exc_info=True)
                continue
        
        logger.info(f"Successfully fetched {len(disputes)} active disputes")
        return {"disputes": disputes}
        
    except Exception as e:
        logger.error(f"Error fetching disputes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch disputes")


# ============================================
# STRIPE REFUND HELPER FUNCTION
# ============================================
async def process_stripe_refund(
    db: AsyncSession,
    project_id: uuid.UUID,
    investor_id: uuid.UUID,
    reason: str = "dispute_resolution"
) -> dict:
    """
    Process a full refund for a payment via Stripe.

    Args:
        db: Database session
        project_id: The project UUID
        investor_id: The investor (customer) UUID
        reason: Reason for the refund

    Returns:
        dict with success status, refund details, or error message
    """
    try:
        if not STRIPE_SECRET_KEY:
            return {
                "success": False,
                "error": "Stripe is not configured"
            }

        # Find the completed checkout session for this project and investor
        checkout_result = await db.execute(
            select(CheckoutSession).where(
                and_(
                    CheckoutSession.project_id == project_id,
                    CheckoutSession.customer_id == investor_id,
                    CheckoutSession.status == 'completed'
                )
            ).order_by(CheckoutSession.completed_at.desc())
        )
        checkout_session = checkout_result.scalar_one_or_none()

        if not checkout_session:
            return {
                "success": False,
                "error": "No completed payment found for this project and investor"
            }

        # Get the Stripe session ID
        stripe_session_id = checkout_session.session_id
        if not stripe_session_id:
            return {
                "success": False,
                "error": "No Stripe session ID found"
            }

        # Retrieve the Stripe session to get the payment intent
        try:
            stripe_session = stripe.checkout.Session.retrieve(stripe_session_id)
            payment_intent_id = stripe_session.payment_intent

            if not payment_intent_id:
                return {
                    "success": False,
                    "error": "No payment intent found for this session"
                }
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error retrieving session: {e}")
            return {
                "success": False,
                "error": f"Failed to retrieve payment details: {str(e)}"
            }

        # Create the refund
        try:
            refund = stripe.Refund.create(
                payment_intent=payment_intent_id,
                reason="requested_by_customer",  # Stripe valid reasons: duplicate, fraudulent, requested_by_customer
                metadata={
                    "project_id": str(project_id),
                    "investor_id": str(investor_id),
                    "reason": reason,
                    "refund_type": "full_refund",
                    "initiated_by": "admin_dispute_resolution"
                }
            )

            logger.info(f"Stripe refund created: {refund.id} for payment_intent: {payment_intent_id}")

            # Update checkout session status
            checkout_session.status = 'refunded'
            checkout_session.extra_data = checkout_session.extra_data or {}
            checkout_session.extra_data['refund_id'] = refund.id
            checkout_session.extra_data['refund_status'] = refund.status
            checkout_session.extra_data['refund_amount'] = refund.amount / 100  # Convert from cents
            checkout_session.extra_data['refunded_at'] = datetime.utcnow().isoformat()

            await db.commit()

            return {
                "success": True,
                "refund_id": refund.id,
                "refund_status": refund.status,
                "refund_amount": refund.amount / 100,
                "currency": refund.currency
            }

        except stripe.error.StripeError as e:
            logger.error(f"Stripe error creating refund: {e}")
            return {
                "success": False,
                "error": f"Stripe refund failed: {str(e)}"
            }

    except Exception as e:
        logger.error(f"Error processing refund: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Refund processing error: {str(e)}"
        }


@app.post("/api/admin/disputes-simple/{project_id}/resolve")
async def resolve_simple_dispute(
    project_id: str,
    data: dict,
    admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Resolve a dispute with new options:
    - refund_investor: Full refund to investor, project cancelled for this purchase
    - refund_developer: Full payment to developer, project completed for this purchase
    - continue_project: Return to previous status, give another chance
    
    For fixed_price projects: Only affects the specific investor's purchase
    For auction projects: Affects the entire project status
    """
    try:
        # ✅ NEW: Get dispute ID from request body
        dispute_id = data.get('dispute_id')
        
        # Get project
        project_query = await db.execute(
            select(Project).where(Project.id == uuid.UUID(project_id))
        )
        project = project_query.scalar_one_or_none()
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # ✅ UPDATED: Get active dispute record by dispute_id or project_id
        if dispute_id:
            # Resolve specific dispute by ID
            dispute_query = await db.execute(
                select(ProjectDisputeSimple).where(ProjectDisputeSimple.id == uuid.UUID(dispute_id))
            )
        else:
            # Get any active dispute for this project (legacy support)
            dispute_query = await db.execute(
                select(ProjectDisputeSimple)
                .where(
                    ProjectDisputeSimple.project_id == uuid.UUID(project_id),
                    ProjectDisputeSimple.resolved == False
                )
            )
        
        dispute = dispute_query.scalar_one_or_none()
        
        if not dispute:
            raise HTTPException(status_code=404, detail="Active dispute not found")
        
        resolution = data.get('resolution')
        admin_notes = data.get('admin_notes', '').strip()
        
        if not resolution or resolution not in ['refund_investor', 'refund_developer', 'continue_project']:
            raise HTTPException(
                status_code=400,
                detail="Invalid resolution. Must be 'refund_investor', 'refund_developer', or 'continue_project'"
            )
        
        logger.info(f"Resolving dispute {dispute.id} for project {project_id}: resolution={resolution}")

        # ✅ PROCESS STRIPE REFUND IF RESOLUTION IS refund_investor
        refund_result = None
        if resolution == 'refund_investor':
            logger.info(f"Processing Stripe refund for investor {dispute.investor_id}")
            refund_result = await process_stripe_refund(
                db=db,
                project_id=uuid.UUID(project_id),
                investor_id=dispute.investor_id,
                reason=f"Dispute resolution - Admin notes: {admin_notes[:100]}" if admin_notes else "Dispute resolution"
            )

            if refund_result["success"]:
                logger.info(f"✅ Stripe refund successful: {refund_result['refund_id']}, amount: ${refund_result['refund_amount']}")
            else:
                logger.warning(f"⚠️ Stripe refund failed: {refund_result.get('error')}. Continuing with dispute resolution.")

        # ✅ IMPORTANT: Mark dispute as resolved
        dispute.resolved = True
        dispute.resolution = resolution
        dispute.admin_notes = admin_notes
        dispute.resolved_by = uuid.UUID(admin['id'])
        dispute.resolved_at = datetime.utcnow()

        # Store refund result in dispute notes if available
        if refund_result:
            dispute.admin_notes = f"{admin_notes}\n\n--- Refund Details ---\n" + (
                f"Refund ID: {refund_result.get('refund_id')}\n"
                f"Amount: ${refund_result.get('refund_amount', 0):.2f}\n"
                f"Status: {refund_result.get('refund_status', 'N/A')}"
                if refund_result["success"]
                else f"Refund Failed: {refund_result.get('error')}"
            )

        # ✅ NEW LOGIC: Handle status changes differently for fixed_price vs auction
        is_fixed_price = dispute.previous_status == 'fixed_price'

        if is_fixed_price:
            # For fixed_price projects: Check if there are other active disputes
            other_disputes_query = await db.execute(
                select(ProjectDisputeSimple)
                .where(
                    ProjectDisputeSimple.project_id == uuid.UUID(project_id),
                    ProjectDisputeSimple.resolved == False,
                    ProjectDisputeSimple.id != dispute.id
                )
            )
            other_active_disputes = other_disputes_query.scalars().all()
            
            # Only change status back to fixed_price if no other active disputes
            if not other_active_disputes:
                project.status = 'fixed_price'
                logger.info(f"No other active disputes - returning project to fixed_price status")
            else:
                logger.info(f"Other active disputes exist - keeping project status as is")
            
            resolution_message = "Admin has resolved this dispute. "
            if resolution == 'refund_investor':
                if refund_result and refund_result["success"]:
                    resolution_message += f"Full refund of ${refund_result['refund_amount']:.2f} issued to investor."
                else:
                    resolution_message += "Full refund issued to investor."
            elif resolution == 'refund_developer':
                resolution_message += "Full payment released to developer."
            else:  # continue_project
                resolution_message += "Both parties can continue working together."
        else:
            # For auction projects: Change project status as before
            if resolution == 'refund_investor':
                project.status = 'cancelled'
                if refund_result and refund_result["success"]:
                    resolution_message = f"Admin ruled in favor of the investor. Full refund of ${refund_result['refund_amount']:.2f} issued."
                else:
                    resolution_message = "Admin ruled in favor of the investor. Full refund issued."
            elif resolution == 'refund_developer':
                project.status = 'completed'
                resolution_message = "Admin ruled in favor of the developer. Full payment released."
            else:  # continue_project
                project.status = dispute.previous_status or 'in_progress'
                resolution_message = "Admin decided to give both parties another chance. Project continues."
        
        # Get users
        developer_query = await db.execute(
            select(User).where(User.id == project.developer_id)
        )
        developer = developer_query.scalar_one()
        
        # ✅ UPDATED: Get the specific investor for this dispute
        investor_query = await db.execute(
            select(User).where(User.id == dispute.investor_id)
        )
        investor = investor_query.scalar_one()

        # ✅ CREATE PAYOUT RECORD IF RESOLUTION IS refund_developer
        if resolution == 'refund_developer':
            # Calculate payout amounts
            project_amount = float(project.budget)
            platform_fee_percentage = 6.0  # 6%
            platform_fee = project_amount * (platform_fee_percentage / 100)
            developer_payout_amount = project_amount - platform_fee

            # Find the checkout session for this project and investor
            checkout_result = await db.execute(
                select(CheckoutSession).where(
                    and_(
                        CheckoutSession.project_id == project.id,
                        CheckoutSession.customer_id == dispute.investor_id,
                        CheckoutSession.status == 'completed'
                    )
                ).order_by(CheckoutSession.completed_at.desc())
            )
            checkout_session = checkout_result.scalar_one_or_none()

            # Create payout record
            payout_record = DeveloperPayout(
                developer_id=project.developer_id,
                project_id=project.id,
                checkout_session_id=checkout_session.id if checkout_session else None,
                investor_id=dispute.investor_id,
                gross_amount=Decimal(str(project_amount)),
                platform_fee=Decimal(str(platform_fee)),
                net_amount=Decimal(str(developer_payout_amount)),
                currency='USD',
                payout_method=developer.payout_method,
                payout_email=developer.payout_email,
                payout_details=developer.payout_details,
                status='pending',
                description=f"Dispute resolution - Payment for project: {project.title}"
            )
            db.add(payout_record)

            # Update developer earnings
            developer.total_earnings = (developer.total_earnings or 0) + Decimal(str(developer_payout_amount))
            developer.projects_completed = (developer.projects_completed or 0) + 1

            logger.info(f"✅ Payout record created for developer {developer.email} via dispute resolution: ${developer_payout_amount:.2f}")

        # Create notifications for both parties
        notification_details = {
            'project_id': str(project.id),
            'project_title': project.title,
            'resolution': resolution,
            'admin_notes': admin_notes
        }

        # Add refund details to notifications if refund was processed
        if refund_result and refund_result["success"]:
            notification_details['refund_id'] = refund_result.get('refund_id')
            notification_details['refund_amount'] = refund_result.get('refund_amount')
            notification_details['refund_status'] = refund_result.get('refund_status')

        for user in [developer, investor]:
            notification = Notification(
                user_id=user.id,
                type='dispute_resolved',
                title='✅ Dispute Resolved',
                message=f'The dispute for project "{project.title}" has been resolved. {resolution_message}',
                link=f"/projects/{project_id}",
                details=notification_details,
                read=False
            )
            db.add(notification)
        
        # Send system message to chat
        try:
            # ✅ UPDATED: Filter by both project_id AND investor_id to get the correct chat room
            room_query = await db.execute(
                select(ChatRoom).where(
                    and_(
                        ChatRoom.project_id == project.id,
                        ChatRoom.investor_id == dispute.investor_id
                    )
                )
            )
            room = room_query.scalar_one_or_none()
            
            if room:
                admin_notes_part = f"\n\nAdmin Notes: {admin_notes}" if admin_notes else ""
                system_message_text = (
                    f"✅ Dispute Resolved\n\n"
                    f"{resolution_message}"
                    f"{admin_notes_part}"
                )
                
                system_message = ChatMessage(
                    room_id=room.id,
                    sender_id=project.developer_id,
                    message=system_message_text,
                    message_type='system',
                    created_at=datetime.utcnow()
                )
                db.add(system_message)
        except Exception as e:
            logger.error(f"Error adding system message to chat: {e}")
        
        # Commit all changes
        await db.commit()
        
        # Log activity
        try:
            activity = Activity(
                user_id=uuid.UUID(admin['id']),
                type='dispute_resolved',
                action=f'dispute_resolved_{resolution}',
                details={
                    'project_id': str(project.id),
                    'project_title': project.title,
                    'resolution': resolution,
                    'admin_notes': admin_notes
                }
            )
            db.add(activity)
            await db.commit()
        except Exception as e:
            logger.error(f"Failed to log activity: {e}")
        
        response = {
            "success": True,
            "message": "Dispute resolved successfully",
            "resolution": resolution,
            "new_status": project.status
        }

        # Add refund details to response if refund was processed
        if refund_result:
            response["refund"] = {
                "processed": refund_result["success"],
                "refund_id": refund_result.get("refund_id"),
                "refund_amount": refund_result.get("refund_amount"),
                "refund_status": refund_result.get("refund_status"),
                "error": refund_result.get("error") if not refund_result["success"] else None
            }

        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving dispute: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to resolve dispute: {str(e)}")


# Add this endpoint to server.py (after the dispute creation endpoint)
@app.get("/api/chat/rooms/{room_id}/has-active-dispute")
async def check_active_dispute(
    room_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Check if there's an active (unresolved) dispute for this chat room
    Returns: { "has_active_dispute": bool, "dispute_id": str | null }
    """
    try:
        # Get chat room
        room_query = await db.execute(
            select(ChatRoom).where(ChatRoom.id == uuid.UUID(room_id))
        )
        room = room_query.scalar_one_or_none()
        
        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")
        
        # Verify user is part of this chat
        user_uuid = uuid.UUID(current_user['id'])
        if user_uuid not in [room.developer_id, room.investor_id]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check for active dispute for this project-investor pair
        dispute_query = await db.execute(
            select(ProjectDisputeSimple).where(
                and_(
                    ProjectDisputeSimple.project_id == room.project_id,
                    ProjectDisputeSimple.investor_id == room.investor_id,
                    ProjectDisputeSimple.resolved == False
                )
            )
        )
        active_dispute = dispute_query.scalar_one_or_none()
        
        if active_dispute:
            return {
                "has_active_dispute": True,
                "dispute_id": str(active_dispute.id),
                "disputed_by": str(active_dispute.disputed_by),
                "reason": active_dispute.reason,
                "disputed_at": active_dispute.disputed_at.isoformat()
            }
        
        return {
            "has_active_dispute": False,
            "dispute_id": None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking active dispute: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat/rooms/{room_id}/pending-payout")
async def get_pending_payout_for_chat(
    room_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get pending payout info for a chat room (developer view)
    Returns payout details if there's a pending/processing payout for this project
    """
    try:
        # Get chat room
        room_query = await db.execute(
            select(ChatRoom).where(ChatRoom.id == uuid.UUID(room_id))
        )
        room = room_query.scalar_one_or_none()

        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")

        user_uuid = uuid.UUID(current_user['id'])

        # Verify user is part of this chat
        if user_uuid not in [room.developer_id, room.investor_id]:
            raise HTTPException(status_code=403, detail="Access denied")

        # Only show payout info to the developer
        is_developer = user_uuid == room.developer_id

        # Find pending/processing payout for this project and investor
        payout_query = await db.execute(
            select(DeveloperPayout).where(
                and_(
                    DeveloperPayout.project_id == room.project_id,
                    DeveloperPayout.developer_id == room.developer_id,
                    DeveloperPayout.investor_id == room.investor_id,
                    DeveloperPayout.status.in_(['pending', 'processing'])
                )
            ).order_by(DeveloperPayout.created_at.desc())
        )
        payout = payout_query.scalar_one_or_none()

        if not payout:
            return {
                "has_pending_payout": False,
                "payout": None,
                "is_developer": is_developer
            }

        # Get developer's current payout preferences
        developer_query = await db.execute(
            select(User).where(User.id == room.developer_id)
        )
        developer = developer_query.scalar_one_or_none()

        return {
            "has_pending_payout": True,
            "is_developer": is_developer,
            "payout": {
                "id": str(payout.id),
                "gross_amount": float(payout.gross_amount),
                "platform_fee": float(payout.platform_fee),
                "net_amount": float(payout.net_amount),
                "currency": payout.currency,
                "status": payout.status,
                "created_at": payout.created_at.isoformat() if payout.created_at else None,
                "payout_method": payout.payout_method,
                "payout_email": payout.payout_email
            },
            "developer_preferences": {
                "payout_method": developer.payout_method if developer else None,
                "payout_email": developer.payout_email if developer else None,
                "payout_currency": developer.payout_currency if developer else 'USD',
                "payout_verified": developer.payout_verified if developer else False,
                "has_payout_method": bool(developer and developer.payout_method)
            } if is_developer else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pending payout: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/rooms/{room_id}/confirm-payout-method")
async def confirm_payout_method_for_chat(
    room_id: str,
    data: dict,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Developer confirms/updates their payout method for a pending payout
    This updates both their profile preferences and the specific payout record
    """
    try:
        # Get chat room
        room_query = await db.execute(
            select(ChatRoom).where(ChatRoom.id == uuid.UUID(room_id))
        )
        room = room_query.scalar_one_or_none()

        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")

        user_uuid = uuid.UUID(current_user['id'])

        # Verify user is the developer
        if user_uuid != room.developer_id:
            raise HTTPException(status_code=403, detail="Only the developer can update payout method")

        # Get developer
        developer_query = await db.execute(
            select(User).where(User.id == room.developer_id)
        )
        developer = developer_query.scalar_one_or_none()

        if not developer:
            raise HTTPException(status_code=404, detail="Developer not found")

        # Extract payout method from request
        payout_method = data.get('payout_method')
        payout_email = data.get('payout_email')
        payout_details = data.get('payout_details', {})
        payout_currency = data.get('payout_currency', 'USD')

        if not payout_method:
            raise HTTPException(status_code=400, detail="Payout method is required")

        # Validate email for PayPal/Wise
        if payout_method in ['paypal', 'wise'] and not payout_email:
            raise HTTPException(status_code=400, detail="Email is required for PayPal/Wise")

        # Update developer's payout preferences in profile
        developer.payout_method = payout_method
        developer.payout_email = payout_email
        developer.payout_details = payout_details
        developer.payout_currency = payout_currency
        developer.payout_verified = False  # Reset verification when changed

        # Update the pending payout record with new method
        payout_query = await db.execute(
            select(DeveloperPayout).where(
                and_(
                    DeveloperPayout.project_id == room.project_id,
                    DeveloperPayout.developer_id == room.developer_id,
                    DeveloperPayout.investor_id == room.investor_id,
                    DeveloperPayout.status.in_(['pending', 'processing'])
                )
            ).order_by(DeveloperPayout.created_at.desc())
        )
        payout = payout_query.scalar_one_or_none()

        if payout:
            payout.payout_method = payout_method
            payout.payout_email = payout_email
            payout.payout_details = payout_details

        await db.commit()

        logger.info(f"Developer {developer.email} updated payout method to {payout_method}")

        # 📧 Send email notification to admin when developer confirms payout
        if payout:
            try:
                admin_email = os.getenv('ADMIN_EMAIL', 'bitebids@gmail.com')

                # Get project details
                project_query = await db.execute(
                    select(Project).where(Project.id == room.project_id)
                )
                project = project_query.scalar_one_or_none()

                # Get investor details
                investor_query = await db.execute(
                    select(User).where(User.id == room.investor_id)
                )
                investor = investor_query.scalar_one_or_none()

                subject = f"💰 Developer Confirmed Payout Method - {project.title if project else 'Unknown Project'}"

                payout_method_display = {
                    'paypal': 'PayPal',
                    'wise': 'Wise',
                    'bank_transfer': 'Bank Transfer',
                    'crypto': 'Cryptocurrency'
                }.get(payout_method, payout_method.title())

                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <style>
                        body {{
                            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                            line-height: 1.6;
                            color: #333;
                            max-width: 600px;
                            margin: 0 auto;
                            padding: 20px;
                        }}
                        .header {{
                            background: linear-gradient(135deg, #22c55e, #16a34a);
                            color: white;
                            padding: 30px;
                            border-radius: 10px 10px 0 0;
                            text-align: center;
                        }}
                        .content {{
                            background: #f8fafc;
                            padding: 30px;
                            border-radius: 0 0 10px 10px;
                        }}
                        .detail-box {{
                            background: white;
                            padding: 20px;
                            margin: 15px 0;
                            border-radius: 8px;
                            border-left: 4px solid #22c55e;
                        }}
                        .label {{
                            font-weight: 600;
                            color: #64748b;
                            margin-bottom: 5px;
                        }}
                        .value {{
                            color: #1e293b;
                            margin-bottom: 15px;
                        }}
                        .payout-box {{
                            background: #f0fdf4;
                            padding: 15px;
                            border-radius: 8px;
                            border-left: 4px solid #22c55e;
                            margin: 15px 0;
                        }}
                        .button {{
                            display: inline-block;
                            padding: 12px 30px;
                            background: #22c55e;
                            color: white;
                            text-decoration: none;
                            border-radius: 8px;
                            margin-top: 20px;
                            font-weight: 600;
                        }}
                        .amount {{
                            font-size: 24px;
                            font-weight: 700;
                            color: #22c55e;
                            margin: 10px 0;
                        }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1 style="margin: 0; font-size: 24px;">💰 Payout Method Confirmed</h1>
                        <p style="margin: 10px 0 0 0; opacity: 0.9;">Ready for admin processing</p>
                    </div>

                    <div class="content">
                        <div class="detail-box">
                            <div class="label">Project:</div>
                            <div class="value"><strong>{project.title if project else 'N/A'}</strong></div>

                            <div class="label">Developer:</div>
                            <div class="value">{developer.name} ({developer.email})</div>

                            <div class="label">Investor:</div>
                            <div class="value">{investor.name if investor else 'N/A'} ({investor.email if investor else 'N/A'})</div>

                            <div class="label">Payout Amount:</div>
                            <div class="amount">${payout.amount:,.2f}</div>

                            <div class="label">Payout Status:</div>
                            <div class="value"><strong>{payout.status.upper()}</strong></div>
                        </div>

                        <div class="payout-box">
                            <div class="label">Payout Method:</div>
                            <div class="value"><strong>{payout_method_display}</strong></div>

                            <div class="label">Payout Email/Account:</div>
                            <div class="value">{payout_email or 'Not provided'}</div>

                            {f'<div class="label">Currency:</div><div class="value">{payout_currency}</div>' if payout_currency else ''}
                        </div>

                        <p style="color: #64748b; margin-top: 20px;">
                            The developer has confirmed their payout method. Please process this payout in the admin panel.
                        </p>

                        <a href="https://bite-bids.vercel.app/admin/payouts" class="button">
                            Process Payout
                        </a>
                    </div>
                </body>
                </html>
                """

                # Send email asynchronously (don't block the response)
                await send_email(admin_email, subject, html_content)
                logger.info(f"📧 Payout confirmation email sent to admin")

            except Exception as email_error:
                logger.error(f"Failed to send payout confirmation email: {email_error}")
                # Don't fail the request if email fails

        return {
            "success": True,
            "message": "Payout method updated successfully",
            "payout_method": payout_method,
            "payout_email": payout_email
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating payout method: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# ADMIN DISPUTE ENDPOINTS
# ============================================

@app.get("/api/admin/disputes")
async def get_all_disputes(
    status: Optional[str] = None,
    admin: User = Depends(get_current_admin)
):
    """Get all disputes for admin review"""
    async with async_session_maker() as db:
        try:
            query = select(ProjectDelivery).where(ProjectDelivery.status == 'disputed')
            
            if status:
                query = query.where(ProjectDelivery.status == status)
            
            result = await db.execute(query.order_by(ProjectDelivery.disputed_at.desc()))
            deliveries = result.scalars().all()
            
            disputes = []
            for delivery in deliveries:
                # Get project details
                project_query = await db.execute(
                    select(Project).where(Project.id == delivery.project_id)
                )
                project = project_query.scalar_one()
                
                # Get developer and investor
                dev_query = await db.execute(
                    select(User).where(User.id == delivery.developer_id)
                )
                developer = dev_query.scalar_one()
                
                inv_query = await db.execute(
                    select(User).where(User.id == delivery.investor_id)
                )
                investor = inv_query.scalar_one()
                
                disputes.append({
                    "id": str(delivery.id),
                    "project": {
                        "id": str(project.id),
                        "title": project.title,
                        "amount": float(delivery.project_amount)
                    },
                    "developer": {
                        "id": str(developer.id),
                        "name": developer.name,
                        "email": developer.email
                    },
                    "investor": {
                        "id": str(investor.id),
                        "name": investor.name,
                        "email": investor.email
                    },
                    "dispute_reason": delivery.dispute_reason,
                    "dispute_notes": delivery.dispute_notes,
                    "disputed_at": delivery.disputed_at.isoformat(),
                    "delivery_url": delivery.delivery_url,
                    "delivery_notes": delivery.delivery_notes,
                    "platform_commission": float(delivery.platform_commission),
                    "developer_payout": float(delivery.developer_payout)
                })
            
            return {"disputes": disputes}
            
        except Exception as e:
            logger.error(f"Error getting disputes: {e}")
            raise HTTPException(status_code=500, detail="Failed to get disputes")


@app.get("/api/admin/content-filter-logs")
async def get_content_filter_logs(
    limit: int = 50,
    admin: User = Depends(get_current_admin)
):
    """Get content filter violation logs"""
    async with async_session_maker() as db:
        try:
            query = select(ContentFilterLog).order_by(ContentFilterLog.created_at.desc()).limit(limit)
            result = await db.execute(query)
            logs = result.scalars().all()
            
            log_data = []
            for log in logs:
                user_query = await db.execute(
                    select(User).where(User.id == log.user_id)
                )
                user = user_query.scalar_one()
                
                log_data.append({
                    "id": str(log.id),
                    "user": {
                        "id": str(user.id),
                        "name": user.name,
                        "email": user.email
                    },
                    "room_id": str(log.chat_room_id),
                    "original_message": log.original_message,
                    "filtered_content": log.filtered_content,
                    "action_taken": log.action_taken,
                    "created_at": log.created_at.isoformat()
                })
            
            return {"logs": log_data}
            
        except Exception as e:
            logger.error(f"Error getting filter logs: {e}")
            raise HTTPException(status_code=500, detail="Failed to get logs")


# ============================================
# PAYMENT ROUTES
# ============================================


@app.post("/api/payments/stripe/create-checkout-session")
async def create_stripe_checkout_session(
    request: StripeCheckoutRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Create a Stripe Checkout Session for payment processing
    This creates a REAL Stripe session and redirects user to Stripe's hosted checkout page
    """
    
    if not STRIPE_SECRET_KEY:
        raise HTTPException(
            status_code=500, 
            detail="Stripe is not configured. Please contact support."
        )
    
    try:
        # Calculate fees
        base_amount = request.amount
        fee_percentage = PLATFORM_FEE_PERCENTAGE / 100
        fees = round(base_amount * fee_percentage + PLATFORM_FIXED_FEE, 2)
        total_amount = base_amount + fees
        
        # Convert to cents for Stripe (Stripe uses smallest currency unit)
        amount_cents = int(total_amount * 100)
        
        # Build metadata to track payment details
        metadata = {
            # 'order_type': request.order_type,
            'item_id': request.item_id,
            'user_id': current_user.get('id'),
            'user_email': request.customer_email,
            'base_amount': str(base_amount),
            'fees': str(fees),
            'total_amount': str(total_amount),
        }
        
        # Add optional fields to metadata
        if request.project_id:
            metadata['project_id'] = request.project_id
        if request.winner_bid_id:
            metadata['winner_bid_id'] = request.winner_bid_id
        if request.notification_id:
            metadata['notification_id'] = request.notification_id
        
        # Determine item name/description
        item_name = f"BiteBids {request.order_type.title()} Payment"
        if request.order_type == 'auction':
            item_name = "Auction Winning Bid Payment"
        

        expiry = datetime.utcnow() + timedelta(minutes=30)
        expiry_ts = int(expiry.replace(tzinfo=timezone.utc).timestamp())

        # Create Stripe Checkout Session
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'unit_amount': amount_cents,
                    'product_data': {
                        'name': item_name,
                        'description': f'Payment for project: {request.item_id}',
                    },
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=f"{FRONTEND_URL}/payment/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{FRONTEND_URL}/payment/cancel",
            customer_email=request.customer_email,
            metadata=metadata,
            billing_address_collection='required',
            payment_intent_data={
                'metadata': metadata,
                'description': f'BiteBids payment for {item_name}',
            },
            expires_at=expiry_ts,

        )
        
        # ✅ NOW create database record with Stripe session ID
        customer_id_uuid = uuid.UUID(current_user.get('id'))
        project_id_uuid = uuid.UUID(request.project_id) if request.project_id else None
        
        checkout_record = CheckoutSession(
            id=uuid.uuid4(),
            session_id=checkout_session.id,  # ✅ Store Stripe session ID
            order_reference=f"stripe_{uuid.uuid4()}",
            external_reference=checkout_session.id,
            project_id=project_id_uuid,
            amount=base_amount,
            total_with_fees=total_amount,
            fees=fees,
            payment_method='stripe_card',
            customer_id=customer_id_uuid,
            status='pending',
            payment_url=checkout_session.url,
            payment_method_used='stripe',
            expires_at=datetime.utcnow() + timedelta(minutes=30),
            extra_data={
                'stripe_session_id': checkout_session.id,
                'metadata': metadata
            }
        )
        
        db.add(checkout_record)
        await db.commit()
        await db.refresh(checkout_record)
        
        logger.info(f"Stripe checkout session created: {checkout_session.id} for user {current_user.get('email')}")
        
        return {
            "success": True,
            "session_id": checkout_session.id,  # ✅ Return Stripe session ID
            "checkout_url": checkout_session.url,
            "order_reference": checkout_record.order_reference,
            "amount": base_amount,
            "fees": fees,
            "total": total_amount
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Stripe error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error creating Stripe checkout session: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create checkout session: {str(e)}"
        )



# ============================================
# COMPLETE UPDATED STRIPE WEBHOOK ENDPOINT
# Replace your existing webhook endpoint with this complete version
# ============================================

@app.post("/api/payments/stripe/webhook")
async def stripe_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Handle Stripe webhook events
    This is called by Stripe when payment status changes
    """
    
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET')
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        ) if webhook_secret else json.loads(payload)
    except ValueError as e:
        logger.error(f"Invalid payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid signature: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Handle the event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        
        # Stripe metadata we set when creating the session
        metadata = session.get('metadata', {}) or {}
        payment_type = metadata.get('payment_type')
        project_id_str = metadata.get('project_id')
        winner_bid_id_str = metadata.get('winner_bid_id')
        notification_id_str = metadata.get('notification_id')
        investor_id_str = metadata.get('user_id')
        investor_email = metadata.get('user_email')
        
        # Stripe amount is in cents
        amount_total = session.get('amount_total')
        amount_paid = (amount_total / 100.0) if amount_total is not None else float(metadata.get('amount', 0) or 0)

        # 1) Update checkout session record
        stmt = select(CheckoutSession).where(
            CheckoutSession.session_id == session['id']
        )
        result = await db.execute(stmt)
        checkout_record = result.scalar_one_or_none()
        
        if checkout_record:
            checkout_record.status = 'completed'
            checkout_record.completed_at = datetime.utcnow()
            await db.commit()
            
            logger.info(f"Payment completed for session: {session['id']}")
        
        # ============================================
        # NEW: Handle project posting payment ($0.99)
        # ============================================
      
        if payment_type == 'project_posting':
            user_id = metadata.get('user_id')
            user_email = metadata.get('user_email')
            
            logger.info(f"✅ Project posting payment verified for user {user_id}")
            
            # ✅ ADD POSTING CREDIT TO USER
            if user_id:
                try:
                    user_uuid = uuid.UUID(user_id)
                    user_result = await db.execute(
                        select(User).where(User.id == user_uuid)
                    )
                    user = user_result.scalar_one_or_none()
                    
                    if user:
                        # Add 1 posting credit
                        user.posting_credits = (user.posting_credits or 0) + 1
                        await db.commit()
                        await db.refresh(user)
                        
                        logger.info(f"✅ Added posting credit. User {user.email} now has {user.posting_credits} credits")
                        
                        # Send notification
                        # notif = Notification(
                        #     user_id=user_uuid,
                        #     type="payment_verified",
                        #     title="Posting Credit Added! ✓",
                        #     message=f"Payment successful! You now have {user.posting_credits} posting credit(s). You can post your project now!",
                        #     link="/dashboard",
                        #     details={
                        #         "payment_type": "project_posting",
                        #         "amount": float(amount_paid),
                        #         "credits_added": 1,
                        #         "total_credits": user.posting_credits,
                        #         "session_id": session['id']
                        #     },
                        #     read=False
                        # )
                        # db.add(notif)
                        # await db.commit()
                        # await db.refresh(notif)
                        
                        # Send live notification
                        await send_notification_to_user(
                            user_id,
                            {
                                "type": "payment_verified",
                                "title": "Posting Credit Added! ✓",
                                "message": f"You now have {user.posting_credits} posting credit(s). Ready to post your project!",
                                "link": "/dashboard",
                                "details": {
                                    "payment_type": "project_posting",
                                    "amount": float(amount_paid),
                                    "credits_added": 1,
                                    "total_credits": user.posting_credits
                                }
                            },
                            db
                        )
                        
                except Exception as e:
                    logger.error(f"Failed to add posting credit: {e}")
            
                return {"status": "success", "payment_type": "project_posting"}



        # ============================================
        # EXISTING: Handle project payment (auction/fixed price)
        # ============================================
        project = None
        investor_user = None
        developer_user = None

        # 2) Load related DB objects (project, investor, developer)
        try:
            if project_id_str:
                project = await db.scalar(
                    select(Project).where(Project.id == uuid.UUID(project_id_str))
                )
        except ValueError:
            logger.warning(f"Invalid project_id in Stripe metadata: {project_id_str}")

        try:
            if investor_id_str:
                investor_user = await db.scalar(
                    select(User).where(User.id == uuid.UUID(investor_id_str))
                )
        except ValueError:
            logger.warning(f"Invalid investor user_id in Stripe metadata: {investor_id_str}")

        if project and project.developer_id:
            developer_user = await db.scalar(
                select(User).where(User.id == project.developer_id)
            )

        # 3) Update project status -> in_progress (only for auction projects)
        if project:
            # Only change status to in_progress if it was an auction (status='winner_selected')
            # For fixed_price projects, keep the status as fixed_price
            if project.status == 'winner_selected':
                project.status = 'in_progress'
            # fixed_price projects keep their status
            project.updated_at = datetime.utcnow()
            await db.commit()
            logger.info(f"Project {project.id} status: {project.status} after payment")

        # 4) Mark the original "payment_required" notification as read
        if notification_id_str:
            try:
                notif_uuid = uuid.UUID(notification_id_str)
                notif_result = await db.execute(
                    select(Notification).where(Notification.id == notif_uuid)
                )
                existing_notif = notif_result.scalar_one_or_none()
                if existing_notif:
                    existing_notif.read = True
                    existing_notif.read_at = datetime.utcnow()
                    await db.commit()
                    logger.info(f"Notification {notification_id_str} marked as read after payment")
            except ValueError:
                logger.warning(f"Invalid notification_id in Stripe metadata: {notification_id_str}")

        # 5) Create a NEW notification for the developer: "Investor has paid"
        if project and developer_user:
            try:
                formatted_amount = f"${amount_paid:,.2f}"
                dev_notif = Notification(
                    user_id=project.developer_id,
                    type="payment_received",
                    title="💰 Investor payment received",
                    message=(
                        f"The investor {investor_user.name if investor_user else 'a client'} "
                        f"has completed a payment of {formatted_amount} for your project "
                        f"'{project.title}'. You can now start working on the project. "
                        f"Funds are held in BiteBids escrow."
                    ),
                    link=str(project.id),
                    details={
                        "project_id": str(project.id),
                        "project_title": project.title,
                        "amount": float(amount_paid),
                        "winner_bid_id": winner_bid_id_str,
                        "investor_id": investor_id_str,
                        "checkout_session_id": session['id'],
                        "payment_type": "project_payment",
                        "platform_fee_percentage": PLATFORM_FEE_PERCENTAGE,
                        "platform_fixed_fee": PLATFORM_FIXED_FEE,
                    },
                    read=False
                )
                db.add(dev_notif)
                await db.commit()
                logger.info(
                    f"Developer notification created for payment (project={project.id}, developer={project.developer_id})"
                )
            except Exception as e:
                logger.error(f"Failed to create developer payment notification: {e}")

        # ========================================
        # CREATE CHAT ROOM AFTER PAYMENT
        # ========================================
        if project and project.developer_id and investor_id_str:
            try:
                # ✅ UPDATED: Check if chat room already exists for THIS developer-investor pair
                # For fixed_price projects, multiple chat rooms can exist (one per investor)
                room_result = await db.execute(
                    select(ChatRoom).where(
                        and_(
                            ChatRoom.project_id == project.id,
                            ChatRoom.investor_id == uuid.UUID(investor_id_str)
                        )
                    )
                )
                existing_room = room_result.scalar_one_or_none()
                
                if not existing_room:
                    # Create new chat room for this developer-investor pair
                    chat_room = ChatRoom(
                        project_id=project.id,
                        developer_id=project.developer_id,
                        investor_id=uuid.UUID(investor_id_str),
                        status="active"
                    )
                    
                    db.add(chat_room)
                    await db.flush()
                    
                    # Create system message in chat
                    formatted_amount = f"${amount_paid:,.2f}"
                    system_message = ChatMessage(
                        room_id=chat_room.id,
                        sender_id=project.developer_id,
                        message=f"Payment of {formatted_amount} completed successfully! 🎉 Chat room is now active. You can start discussing the project.",
                        message_type="system"
                    )
                    
                    db.add(system_message)
                    await db.commit()
                    
                    logger.info(f"✅ Chat room created: {chat_room.id} for project {project.id} with investor {investor_id_str}")
                    
                    # Send live notification to developer
                    await send_notification_to_user(
                        str(project.developer_id),
                        {
                            "type": "chat_room_created",
                            "title": "Chat Room Created! 💬",
                            "message": f"Payment received! Chat room for '{project.title}' is now active. Start collaborating with your investor!",
                            "link": f"/chat/{chat_room.id}",
                            "details": {
                                "project_id": str(project.id),
                                "room_id": str(chat_room.id),
                                "amount": float(amount_paid),
                                "investor_name": investor_user.name if investor_user else "Investor"
                            }
                        },
                        db
                    )
                    
                    # Send live notification to investor
                    await send_notification_to_user(
                        investor_id_str,
                        {
                            "type": "chat_room_created",
                            "title": "Chat Room Created! 💬",
                            "message": f"Payment successful! Chat room for '{project.title}' is now active. Start discussing with the developer!",
                            "link": f"/chat/{chat_room.id}",
                            "details": {
                                "project_id": str(project.id),
                                "room_id": str(chat_room.id),
                                "amount": float(amount_paid),
                                "developer_name": developer_user.name if developer_user else "Developer"
                            }
                        },
                        db
                    )
                    
                    logger.info(f"✅ Chat room notifications sent for room {chat_room.id}")
                else:
                    logger.info(f"ℹ️ Chat room already exists for project {project.id} with investor {investor_id_str}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to create chat room: {e}")
                # Don't fail the webhook - payment already processed

        # 6) Send emails (investor + developer) using async tasks
        try:
            project_title = project.title if project else "your project"

            # Investor email
            if investor_email:
                asyncio.create_task(
                    send_payment_confirmation_email(
                        to_email=investor_email,
                        project_title=project_title,
                        amount=amount_paid,
                        role="investor"
                    )
                )
            elif investor_user and investor_user.email:
                asyncio.create_task(
                    send_payment_confirmation_email(
                        to_email=investor_user.email,
                        project_title=project_title,
                        amount=amount_paid,
                        role="investor"
                    )
                )

            # Developer email
            if developer_user and developer_user.email:
                asyncio.create_task(
                    send_payment_confirmation_email(
                        to_email=developer_user.email,
                        project_title=project_title,
                        amount=amount_paid,
                        role="developer"
                    )
                )

        except Exception as e:
            logger.error(f"Failed to schedule payment confirmation emails: {e}")
    
    elif event['type'] == 'checkout.session.expired':
        session = event['data']['object']
        
        # Mark session as expired
        stmt = select(CheckoutSession).where(
            CheckoutSession.session_id == session['id']
        )
        result = await db.execute(stmt)
        checkout_record = result.scalar_one_or_none()
        
        if checkout_record:
            checkout_record.status = 'expired'
            await db.commit()
            logger.info(f"Payment session expired: {session['id']}")
    
    return {"status": "success"}


@app.get("/api/payments/stripe/verify-session/{session_id}")
async def verify_stripe_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        query = select(CheckoutSession).where(CheckoutSession.session_id == session_id)
        result = await db.execute(query)
        checkout_record = result.scalar_one_or_none()
        
        if not checkout_record or str(checkout_record.customer_id) != current_user.get('id'):
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session.payment_status == 'paid':
            checkout_record.status = 'completed'
            checkout_record.paid_at = datetime.utcnow()
            await db.commit()
            return {"success": True, "payment_status": "completed"}
        
        return {"success": False, "payment_status": session.payment_status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/payments/session/{session_id}")
async def get_checkout_session(
    session_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get checkout session details"""
    result = await db.execute(
        select(CheckoutSession).where(CheckoutSession.session_id == session_id)
    )
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail="Checkout session not found")
    
    if str(session.customer_id) != current_user["_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return model_to_dict(session)



# ============================================
# OAUTH ROUTES - FIXED ASYNC VERSION
# ============================================

@app.post("/api/auth/oauth/validate")
async def validate_oauth_session(oauth_data: OAuthValidation, db: AsyncSession = Depends(get_db)):
    """Validate OAuth session"""
    try:
        # Call Emergent auth API
        response = requests.get(
            "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
            headers={"X-Session-ID": oauth_data.session_id},
            timeout=10
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid OAuth session")
        
        oauth_user = response.json()
        required_fields = ["id", "email", "name"]
        if not all(field in oauth_user for field in required_fields):
            raise HTTPException(status_code=400, detail="Incomplete OAuth user data")
        
        # Check if user exists
        result = await db.execute(select(User).where(User.email == oauth_user["email"]))
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            user_dict = model_to_dict(existing_user)
        else:
            # Create new user
            new_user = User(
                email=oauth_user["email"],
                password_hash=None,  # OAuth users don't have passwords
                role="developer",  # Default role
                status="active",
                name=oauth_user["name"],
                reputation_score=0,
                profile={
                    "cosmic_theme": "default",
                    "avatar": oauth_user.get("picture"),
                    "bio": "",
                    "oauth_provider": True
                },
                oauth_id=oauth_user["id"]
            )
            db.add(new_user)
            await db.commit()
            await db.refresh(new_user)
            user_dict = model_to_dict(new_user)
        
        # Generate JWT token
        token = create_jwt_token(user_dict)
        
        return {
            "token": token,
            "user": {
                "id": user_dict["_id"],
                "email": user_dict["email"],
                "role": user_dict["role"],
                "name": user_dict["name"],
                "avatar": user_dict.get("profile", {}).get("avatar") if isinstance(user_dict.get("profile"), dict) else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OAuth validation error: {e}")
        raise HTTPException(status_code=401, detail="OAuth validation failed")


@app.get("/api/auth/login/{provider}")
async def oauth_login(provider: str, request: Request):
    """Initiate OAuth login"""
    if provider not in ['github', 'google']:
        raise HTTPException(status_code=400, detail="Unsupported OAuth provider")
    
    if provider == 'github' and not (GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET):
        raise HTTPException(status_code=400, detail="GitHub OAuth not configured")
    
    if provider == 'google' and not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET):
        raise HTTPException(status_code=400, detail="Google OAuth not configured")
    
    # Build OAuth authorization URL
    if provider == 'github':
        auth_url = (f"https://github.com/login/oauth/authorize?"
                   f"client_id={GITHUB_CLIENT_ID}&"
                   f"scope=user:email&"
                   f"redirect_uri={BASE_URL}/api/auth/callback/github&"
                   f"state={uuid.uuid4()}")
    elif provider == 'google':
        auth_url = (f"https://accounts.google.com/o/oauth2/v2/auth?"
                   f"client_id={GOOGLE_CLIENT_ID}&"
                   f"response_type=code&"
                   f"scope=openid email profile&"
                   f"redirect_uri={BASE_URL}/api/auth/callback/google&"
                   f"state={uuid.uuid4()}")
    
    return RedirectResponse(url=auth_url)


@app.get("/api/auth/callback/github")
async def github_callback(code: str, request: Request, db: AsyncSession = Depends(get_db)):
    """Handle GitHub OAuth callback"""
    if not (GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET):
        raise HTTPException(status_code=400, detail="GitHub OAuth not configured")
    
    try:
        # Exchange code for access token
        token_url = "https://github.com/login/oauth/access_token"
        token_data = {
            "client_id": GITHUB_CLIENT_ID,
            "client_secret": GITHUB_CLIENT_SECRET,
            "code": code
        }
        headers = {"Accept": "application/json"}
        
        async with httpx.AsyncClient() as client:
            token_response = await client.post(token_url, data=token_data, headers=headers)
            token_json = token_response.json()
            
            if "access_token" not in token_json:
                error_description = token_json.get("error_description", "Unknown error")
                error_message = urllib.parse.quote(f"Failed to get access token: {error_description}")
                return RedirectResponse(url=f"{FRONTEND_URL}?auth=error&message={error_message}")
            
            access_token = token_json["access_token"]
            
            # Get user info
            user_response = await client.get(
                "https://api.github.com/user",
                headers={"Authorization": f"token {access_token}"}
            )
            user_data = user_response.json()
            
            # Get user email
            email_response = await client.get(
                "https://api.github.com/user/emails",
                headers={"Authorization": f"token {access_token}"}
            )
            emails = email_response.json()
            primary_email = next((email["email"] for email in emails if email["primary"]), None)
            
            if not primary_email:
                error_message = urllib.parse.quote("No primary email found")
                return RedirectResponse(url=f"{FRONTEND_URL}?auth=error&message={error_message}")
            
            # Check if user exists
            result = await db.execute(select(User).where(User.email == primary_email))
            existing_user = result.scalar_one_or_none()
            
            if existing_user:
                # Update last login and set verified for OAuth users
                existing_user.last_login = datetime.utcnow()
                existing_user.verified = True  # Set verified for users logging in via GitHub
                await db.commit()
                await db.refresh(existing_user)

                user_dict = model_to_dict(existing_user)
                token = create_jwt_token(user_dict)

                user_info = {
                    "id": str(existing_user.id),
                    "email": existing_user.email,
                    "name": existing_user.name,
                    "role": existing_user.role
                }

                # Role-based redirect
                if existing_user.role == "investor":
                    redirect_page = "home"
                elif existing_user.role == "developer":
                    redirect_page = "dashboard"
                elif existing_user.role == "admin":
                    redirect_page = "dashboard-admin"
                else:
                    redirect_page = "home"

                # Encode user info properly
                encoded_user = urllib.parse.quote(json.dumps(user_info))

                # Redirect to FRONTEND_URL
                return RedirectResponse(
                    url=f"{FRONTEND_URL}/{redirect_page}?token={token}&user={encoded_user}&auth=success"
                )
            else:
                # New user - redirect to registration
                github_data = {
                    "email": primary_email,
                    "name": user_data.get("name", user_data.get("login", primary_email.split("@")[0])),
                    "provider": "github",
                    "provider_id": str(user_data["id"]),
                    "avatar_url": user_data.get("avatar_url", "")
                }

                # Encode the data properly
                encoded_data = urllib.parse.quote(json.dumps(github_data))

                # Redirect to FRONTEND_URL
                return RedirectResponse(
                    url=f"{FRONTEND_URL}?oauth_data={encoded_data}&auth=register&provider=github"
                )
                
    except Exception as e:
        logger.error(f"GitHub OAuth callback error: {str(e)}", exc_info=True)
        # Encode error message
        error_message = urllib.parse.quote(str(e))
        # Redirect to FRONTEND_URL with error
        return RedirectResponse(url=f"{FRONTEND_URL}?auth=error&message={error_message}")


@app.get("/api/auth/callback/google")
async def google_callback(code: str, request: Request, db: AsyncSession = Depends(get_db)):
    """Handle Google OAuth callback"""
    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET):
        raise HTTPException(status_code=400, detail="Google OAuth not configured")
    
    try:
        # Exchange code for access token
        token_url = "https://oauth2.googleapis.com/token"
        token_data = {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": f"{BASE_URL}/api/auth/callback/google"
        }
        
        async with httpx.AsyncClient() as client:
            token_response = await client.post(token_url, data=token_data)
            token_json = token_response.json()
            
            if "access_token" not in token_json:
                error_description = token_json.get("error_description", "Unknown error")
                error_message = urllib.parse.quote(f"Failed to get access token: {error_description}")
                return RedirectResponse(url=f"{FRONTEND_URL}?auth=error&message={error_message}")
            
            access_token = token_json["access_token"]
            
            # Get user info
            user_response = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            user_data = user_response.json()
            
            email = user_data.get("email")
            if not email:
                error_message = urllib.parse.quote("No email found in Google account")
                return RedirectResponse(url=f"{FRONTEND_URL}?auth=error&message={error_message}")
            
            # Check if user exists
            result = await db.execute(select(User).where(User.email == email))
            existing_user = result.scalar_one_or_none()
            
            if existing_user:
                # Update last login and set verified for OAuth users
                existing_user.last_login = datetime.utcnow()
                existing_user.verified = True  # Set verified for users logging in via Google
                await db.commit()
                await db.refresh(existing_user)

                user_dict = model_to_dict(existing_user)
                token = create_jwt_token(user_dict)

                user_info = {
                    "id": str(existing_user.id),
                    "email": existing_user.email,
                    "name": existing_user.name,
                    "role": existing_user.role
                }

                # Role-based redirect
                if existing_user.role == "investor":
                    redirect_page = "home"
                elif existing_user.role == "developer":
                    redirect_page = "dashboard"
                elif existing_user.role == "admin":
                    redirect_page = "dashboard-admin"
                else:
                    redirect_page = "home"

                # Encode user info properly
                encoded_user = urllib.parse.quote(json.dumps(user_info))

                # Redirect to FRONTEND_URL
                return RedirectResponse(
                    url=f"{FRONTEND_URL}/{redirect_page}?token={token}&user={encoded_user}&auth=success"
                )
            else:
                # New user - redirect to registration with OAuth data
                google_data = {
                    "email": email,
                    "name": user_data.get("name", email.split("@")[0]),
                    "provider": "google",
                    "provider_id": user_data.get("id", ""),
                    "avatar_url": user_data.get("picture", "")
                }

                # Encode the data properly
                encoded_data = urllib.parse.quote(json.dumps(google_data))

                # Redirect to FRONTEND_URL
                return RedirectResponse(
                    url=f"{FRONTEND_URL}?oauth_data={encoded_data}&auth=register&provider=google"
                )
                
    except Exception as e:
        logger.error(f"Google OAuth callback error: {str(e)}", exc_info=True)
        # Encode error message
        error_message = urllib.parse.quote(str(e))
        # Redirect to FRONTEND_URL with error
        return RedirectResponse(url=f"{FRONTEND_URL}?auth=error&message={error_message}")


@app.post("/api/auth/oauth/complete")
async def complete_oauth_registration(request: dict, db: AsyncSession = Depends(get_db)):
    """Complete OAuth registration"""
    try:
        oauth_data = request.get("oauth_data")
        provider = request.get("provider")
        selected_role = request.get("role")
        company = request.get("company", "")
        password = request.get("password")  

        if not oauth_data or not provider or not selected_role:
            raise HTTPException(status_code=400, detail="OAuth data, provider, and role are required")
        
        email = oauth_data.get("email")
        if not email:
            raise HTTPException(status_code=400, detail="Email is required")
        
        # ✅ FIX: Ensure name is never NULL
        name = oauth_data.get("name")
        if not name or not name.strip():
            # Fallback 1: Use email username
            name = email.split("@")[0]
        
        # Additional validation
        if not company or not company.strip():
            raise HTTPException(status_code=400, detail="Company is required")
        
        # Check if user exists
        result = await db.execute(select(User).where(User.email == email))
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")
        
         # Hash Password
        password_hash = hash_password(password)

        # ✅ FIX: Create new user with guaranteed name value
        new_user = User(
            email=email,
            name=name,  # Now guaranteed to have a value
            password_hash=password_hash,
            role=selected_role,
            email_verified=True,
            verified=True,  # OAuth users are automatically verified
            status="active",
            company=company,
            oauth_provider=provider,
            oauth_id=str(oauth_data.get("provider_id", "")),
            reputation_score=0,
            profile={
                "avatar_url": oauth_data.get("avatar_url") or oauth_data.get("picture", "")
            }
        )
        
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        
        # Generate JWT token
        user_dict = model_to_dict(new_user)
        token = create_jwt_token(user_dict)
        
        return {
            "token": token,
            "user": {
                "id": str(new_user.id),
                "email": new_user.email,
                "name": new_user.name,
                "role": new_user.role
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OAuth registration error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to complete registration")


# ============================================
# ADMIN ROUTES
# ============================================

class AdminLogin(BaseModel):
    email: str
    password: str


@app.post("/api/admin/login")
async def admin_login(admin_data: AdminLogin):
    """Admin login endpoint"""
    if admin_data.email != ADMIN_EMAIL:
        raise HTTPException(status_code=401, detail="Invalid admin credentials")
    
    if admin_data.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid admin credentials")
    
    # Create admin JWT token
    admin_token = jwt.encode(
        {
            "admin": True,
            "email": admin_data.email,
            "exp": datetime.utcnow() + timedelta(hours=24)
        },
        JWT_SECRET,
        algorithm=JWT_ALGORITHM
    )
    
    return {
        "token": admin_token,
        "admin": {
            "email": admin_data.email,
            "role": "business_owner"
        }
    }


async def get_admin_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify admin authentication"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        
        if not payload.get("admin"):
            raise HTTPException(status_code=403, detail="Admin access required")
        
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Admin token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid admin token")


@app.get("/api/admin/dashboard")
async def admin_dashboard(admin = Depends(get_admin_user), db: AsyncSession = Depends(get_db)):
    """Admin dashboard statistics"""
    try:
        # User statistics
        total_users_result = await db.execute(select(func.count(User.id)))
        total_users = total_users_result.scalar()
        
        developers_result = await db.execute(select(func.count(User.id)).where(User.role == "developer"))
        developers = developers_result.scalar()
        
        investors_result = await db.execute(select(func.count(User.id)).where(User.role == "investor"))
        investors = investors_result.scalar()
        
        # Project statistics
        total_projects_result = await db.execute(select(func.count(Project.id)))
        total_projects = total_projects_result.scalar()

        # Active projects = any status except "closed"
        active_projects_result = await db.execute(select(func.count(Project.id)).where(Project.status != "closed"))
        active_projects = active_projects_result.scalar()

        # Disputes statistics (from project_disputes_simple table)
        total_disputes_result = await db.execute(select(func.count(ProjectDisputeSimple.id)))
        total_disputes = total_disputes_result.scalar()

        active_disputes_result = await db.execute(select(func.count(ProjectDisputeSimple.id)).where(ProjectDisputeSimple.resolved == False))
        active_disputes = active_disputes_result.scalar()

        # Total revenue from users total_earnings
        total_revenue_result = await db.execute(select(func.coalesce(func.sum(User.total_earnings), 0)))
        total_revenue = float(total_revenue_result.scalar() or 0)
        
        # Recent activity
        recent_users_result = await db.execute(
            select(User).order_by(User.created_at.desc()).limit(5)
        )
        recent_users = recent_users_result.scalars().all()
        
        recent_projects_result = await db.execute(
            select(Project).order_by(Project.created_at.desc()).limit(5)
        )
        recent_projects = recent_projects_result.scalars().all()
        
        return {
            "stats": {
                "users": {
                    "total": total_users,
                    "developers": developers,
                    "investors": investors
                },
                "projects": {
                    "total": total_projects,
                    "active": active_projects
                },
                "disputes": {
                    "total": total_disputes,
                    "active": active_disputes
                },
                "payments": {
                    "total_revenue": total_revenue,
                    "pending_payments": 3,
                    "completed_transactions": 47
                }
            },
            "recent_activity": {
                "users": models_to_list(recent_users),
                "projects": models_to_list(recent_projects)
            }
        }
    except Exception as e:
        logger.error(f"Admin dashboard error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load admin dashboard")


@app.get("/api/admin/transactions")
async def admin_get_transactions(
    skip: int = 0,
    limit: int = 50,
    admin = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all transactions"""
    try:
        total_result = await db.execute(select(func.count(CheckoutSession.id)))
        total = total_result.scalar()
        
        transactions_result = await db.execute(
            select(CheckoutSession).offset(skip).limit(limit)
        )
        transactions = transactions_result.scalars().all()
        
        return {
            "transactions": models_to_list(transactions),
            "total": total,
            "page": skip // limit + 1,
            "pages": (total + limit - 1) // limit
        }
    except Exception as e:
        logger.error(f"Admin get transactions error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch transactions")


@app.get("/api/admin/users")
async def admin_get_users(
    skip: int = 0,
    limit: int = 50,
    admin = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all users with pagination"""
    try:
        total_result = await db.execute(select(func.count(User.id)))
        total = total_result.scalar()
        
        users_result = await db.execute(
            select(User).offset(skip).limit(limit)
        )
        users = users_result.scalars().all()
        
        return {
            "users": models_to_list(users),
            "total": total,
            "page": skip // limit + 1,
            "pages": (total + limit - 1) // limit
        }
    except Exception as e:
        logger.error(f"Admin get users error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch users")



@app.post("/api/admin/create-user")
async def admin_create_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
    admin: dict = Depends(get_current_admin)
):
    """Admin creates a new user using the same UserCreate schema."""

    # Check if email exists
    result = await db.execute(select(User).where(User.email == user_data.email))
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Hash password
    hashed_password = hash_password(user_data.password)

    # Create user (NO BANK DETAILS)
    new_user = User(
        email=user_data.email,
        password_hash=hashed_password,
        role=user_data.role,
        name=user_data.name,
        company=user_data.company,
        status=user_data.status or "active",
        reputation_score=0,
        profile={
            "cosmic_theme": "default",
            "avatar": None,
            "bio": ""
        }
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    return {
        "message": "User created successfully",
        "user": model_to_dict(new_user)
    }


@app.delete("/api/admin/user/{user_id}")
async def admin_delete_user(
    user_id: str,
    admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Delete user account"""
    try:
        result = await db.execute(
            delete(User).where(User.id == uuid.UUID(user_id))
        )
        await db.commit()
        
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {"success": True, "message": "User deleted successfully"}
    except Exception as e:
        logger.error(f"Admin delete user error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete user")


@app.get("/api/auth/verify-email-change")
async def verify_email_change(token: str = Query(...), db: AsyncSession = Depends(get_db)):
    """Verify email change token and update user's email"""
    
    # Debug logging
    logger.info(f"🔍 Received email verification request with token: {token[:10]}...")
    
    result = await db.execute(select(User).where(User.email_change_token == token))
    user = result.scalar_one_or_none()
    
    # Debug: Check if we found a user
    if not user:
        logger.error(f"❌ No user found with token: {token}")
        
        # Debug: Let's check if there are ANY pending email changes
        all_pending = await db.execute(
            select(User).where(User.pending_email.isnot(None))
        )
        pending_users = all_pending.scalars().all()
        logger.info(f"📊 Total users with pending email changes: {len(pending_users)}")
        
        for u in pending_users:
            logger.info(f"   User: {u.email}, Pending: {u.pending_email}, Token: {u.email_change_token[:10] if u.email_change_token else 'None'}...")
        
        raise HTTPException(status_code=400, detail="Invalid verification token")

    if not user.pending_email:
        logger.error(f"❌ User {user.email} has no pending email")
        raise HTTPException(status_code=400, detail="No pending email change")

    # Check if token is expired (24 hours)
    if user.email_change_sent_at:
        token_age = datetime.now(timezone.utc).replace(tzinfo=None) - user.email_change_sent_at
        logger.info(f"⏰ Token age: {token_age}")
        if token_age > timedelta(hours=24):
            logger.error(f"❌ Token expired. Age: {token_age}")
            raise HTTPException(status_code=400, detail="Verification token has expired")

    # Update email
    old_email = user.email
    new_email = user.pending_email
    
    logger.info(f"📝 Updating email from {old_email} to {new_email}")
    
    user.email = user.pending_email
    user.pending_email = None
    user.email_change_token = None
    user.email_change_sent_at = None

    await db.commit()
    
    logger.info(f"✅ Email successfully changed from {old_email} to {new_email}")

    return {
        "success": True,
        "message": "Email updated successfully!",
        "new_email": new_email
    }

# ------------------------------
# BAN USER (Admin Only)
# ------------------------------
class BanRequest(BaseModel):
    reason: Optional[str] = None

@app.post("/api/admin/user/{identifier}/ban")
async def admin_ban_user(
    identifier: str,
    data: BanRequest,
    db: AsyncSession = Depends(get_db),
    admin: dict = Depends(get_current_admin)
):
    """Ban user by ID OR email."""

    user = None

    # 1️⃣ Try UUID first
    try:
        user_uuid = uuid.UUID(identifier)
        result = await db.execute(select(User).where(User.id == user_uuid))
        user = result.scalar_one_or_none()
    except ValueError:
        user = None

    # 2️⃣ If not UUID → treat as email
    if not user:
        result = await db.execute(select(User).where(User.email == identifier))
        user = result.scalar_one_or_none()

    # 3️⃣ Still not found?
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 4️⃣ Ban user
    user.status = "banned"
    user.updated_at = datetime.utcnow()
    db.add(user)

    # Log action
    log = ActivityLog(
        user_id=user.id,
        user_name=user.name,
        type="admin_action",
        action="user_banned",
        details={"reason": data.reason or "No reason provided"},
    )
    db.add(log)

    await db.commit()

    return {"message": f"User {user.email} banned successfully"}


@app.post("/api/admin/user/{user_id}/unban")
async def unban_user(
    user_id: str,
    current_admin: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Unban a user (set status back to active)"""
    try:
        user_uuid = uuid.UUID(user_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid user ID format")

    # Check if user exists
    result = await db.execute(select(User).where(User.id == user_uuid))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # If user is not banned
    if user.status != "banned":
        raise HTTPException(status_code=400, detail="User is not banned")

    # Update status
    await db.execute(
        update(User)
        .where(User.id == user_uuid)
        .values(status="active", updated_at=datetime.utcnow())
    )
    await db.commit()

    return {"message": "User has been unbanned successfully", "user_id": user_id}


# ============================================
# DEVELOPER PAYOUT SYSTEM
# ============================================

# --- Developer Payout Preferences ---
@app.get("/api/users/me/payout-preferences")
async def get_payout_preferences(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current user's payout preferences"""
    user_result = await db.execute(
        select(User).where(User.id == uuid.UUID(current_user['id']))
    )
    user = user_result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "payout_method": user.payout_method,
        "payout_email": user.payout_email,
        "payout_details": user.payout_details,
        "payout_currency": user.payout_currency or 'USD',
        "payout_verified": user.payout_verified or False,
        "total_earnings": float(user.total_earnings or 0)
    }


@app.put("/api/users/me/payout-preferences")
async def update_payout_preferences(
    data: PayoutPreferencesUpdate,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update current user's payout preferences"""
    user_result = await db.execute(
        select(User).where(User.id == uuid.UUID(current_user['id']))
    )
    user = user_result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Validate payout method
    valid_methods = ['paypal', 'wise', 'bank_transfer', 'crypto', 'other']
    if data.payout_method not in valid_methods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid payout method. Must be one of: {', '.join(valid_methods)}"
        )

    # Update payout preferences
    user.payout_method = data.payout_method
    user.payout_email = data.payout_email
    user.payout_details = data.payout_details
    user.payout_currency = data.payout_currency or 'USD'
    user.payout_verified = False  # Reset verification when details change
    user.updated_at = datetime.utcnow()

    await db.commit()

    logger.info(f"User {user.email} updated payout preferences: method={data.payout_method}")

    return {
        "message": "Payout preferences updated successfully",
        "payout_method": user.payout_method,
        "payout_email": user.payout_email,
        "payout_currency": user.payout_currency
    }


# --- Developer Payout History ---
@app.get("/api/users/me/payouts")
async def get_my_payouts(
    current_user: dict = Depends(get_current_user),
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """Get current user's payout history"""
    query = select(DeveloperPayout).where(
        DeveloperPayout.developer_id == uuid.UUID(current_user['id'])
    )

    if status:
        query = query.where(DeveloperPayout.status == status)

    query = query.order_by(DeveloperPayout.created_at.desc()).offset(skip).limit(limit)

    result = await db.execute(query)
    payouts = result.scalars().all()

    # Get total pending amount
    pending_result = await db.execute(
        select(sql_func.sum(DeveloperPayout.net_amount)).where(
            DeveloperPayout.developer_id == uuid.UUID(current_user['id']),
            DeveloperPayout.status == 'pending'
        )
    )
    pending_total = pending_result.scalar() or 0

    return {
        "payouts": [
            {
                "id": str(p.id),
                "project_id": str(p.project_id) if p.project_id else None,
                "gross_amount": float(p.gross_amount),
                "platform_fee": float(p.platform_fee),
                "net_amount": float(p.net_amount),
                "currency": p.currency,
                "status": p.status,
                "payout_method": p.payout_method,
                "transaction_id": p.transaction_id,
                "description": p.description,
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "completed_at": p.completed_at.isoformat() if p.completed_at else None
            }
            for p in payouts
        ],
        "pending_total": float(pending_total),
        "total": len(payouts)
    }


# --- Admin Payout Management ---
@app.get("/api/admin/payouts")
async def get_all_payouts(
    admin: dict = Depends(get_current_admin),
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """Admin: Get all developer payouts"""
    query = select(DeveloperPayout)

    if status:
        query = query.where(DeveloperPayout.status == status)

    query = query.order_by(DeveloperPayout.created_at.desc()).offset(skip).limit(limit)

    result = await db.execute(query)
    payouts = result.scalars().all()

    # Get developer info for each payout
    payouts_with_info = []
    for p in payouts:
        # Get developer info
        dev_result = await db.execute(
            select(User).where(User.id == p.developer_id)
        )
        developer = dev_result.scalar_one_or_none()

        # Get project info
        project_title = None
        if p.project_id:
            proj_result = await db.execute(
                select(Project).where(Project.id == p.project_id)
            )
            project = proj_result.scalar_one_or_none()
            project_title = project.title if project else None

        payouts_with_info.append({
            "id": str(p.id),
            "developer": {
                "id": str(p.developer_id),
                "name": developer.name if developer else "Unknown",
                "email": developer.email if developer else "Unknown"
            },
            "project_id": str(p.project_id) if p.project_id else None,
            "project_title": project_title,
            "gross_amount": float(p.gross_amount),
            "platform_fee": float(p.platform_fee),
            "net_amount": float(p.net_amount),
            "currency": p.currency,
            "status": p.status,
            "payout_method": p.payout_method,
            "payout_email": p.payout_email,
            "payout_details": p.payout_details,
            "transaction_id": p.transaction_id,
            "transaction_notes": p.transaction_notes,
            "failure_reason": p.failure_reason,
            "description": p.description,
            "created_at": p.created_at.isoformat() if p.created_at else None,
            "processed_at": p.processed_at.isoformat() if p.processed_at else None,
            "completed_at": p.completed_at.isoformat() if p.completed_at else None
        })

    # Get summary stats
    stats_result = await db.execute(
        select(
            DeveloperPayout.status,
            sql_func.count(DeveloperPayout.id).label('count'),
            sql_func.sum(DeveloperPayout.net_amount).label('total')
        ).group_by(DeveloperPayout.status)
    )
    stats = {row.status: {"count": row.count, "total": float(row.total or 0)} for row in stats_result}

    return {
        "payouts": payouts_with_info,
        "stats": stats,
        "total": len(payouts_with_info)
    }


@app.post("/api/admin/payouts/{payout_id}/process")
async def process_payout(
    payout_id: str,
    data: PayoutProcessRequest,
    admin: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Admin: Mark payout as processing (admin is working on it)"""
    try:
        payout_uuid = uuid.UUID(payout_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid payout ID")

    payout_result = await db.execute(
        select(DeveloperPayout).where(DeveloperPayout.id == payout_uuid)
    )
    payout = payout_result.scalar_one_or_none()

    if not payout:
        raise HTTPException(status_code=404, detail="Payout not found")

    if payout.status != 'pending':
        raise HTTPException(
            status_code=400,
            detail=f"Cannot process payout with status '{payout.status}'. Must be 'pending'."
        )

    # Update payout status
    payout.status = 'processing'
    payout.processed_by = uuid.UUID(admin['id'])
    payout.processed_at = datetime.utcnow()
    payout.transaction_id = data.transaction_id
    payout.transaction_notes = data.transaction_notes
    payout.updated_at = datetime.utcnow()

    await db.commit()

    logger.info(f"Admin {admin['email']} marked payout {payout_id} as processing")

    # Notify developer
    notification = Notification(
        user_id=payout.developer_id,
        type='payout_processing',
        title='💳 Payout Processing',
        message=f'Your payout of ${float(payout.net_amount):.2f} is being processed. You will receive the funds soon.',
        link='/dashboard',
        details={
            'payout_id': str(payout.id),
            'amount': float(payout.net_amount),
            'method': payout.payout_method
        },
        read=False
    )
    db.add(notification)
    await db.commit()

    return {
        "message": "Payout marked as processing",
        "payout_id": str(payout.id),
        "status": "processing"
    }


@app.post("/api/admin/payouts/{payout_id}/complete")
async def complete_payout(
    payout_id: str,
    data: PayoutCompleteRequest,
    admin: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Admin: Mark payout as completed with transaction proof"""
    try:
        payout_uuid = uuid.UUID(payout_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid payout ID")

    payout_result = await db.execute(
        select(DeveloperPayout).where(DeveloperPayout.id == payout_uuid)
    )
    payout = payout_result.scalar_one_or_none()

    if not payout:
        raise HTTPException(status_code=404, detail="Payout not found")

    if payout.status not in ['pending', 'processing']:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot complete payout with status '{payout.status}'."
        )

    # Update payout status
    payout.status = 'completed'
    payout.transaction_id = data.transaction_id
    payout.transaction_notes = data.transaction_notes
    payout.completed_at = datetime.utcnow()
    payout.processed_by = uuid.UUID(admin['id'])
    payout.updated_at = datetime.utcnow()

    await db.commit()

    logger.info(f"Admin {admin['email']} completed payout {payout_id} with transaction {data.transaction_id}")

    # Notify developer
    notification = Notification(
        user_id=payout.developer_id,
        type='payout_completed',
        title='✅ Payout Completed!',
        message=f'Your payout of ${float(payout.net_amount):.2f} has been sent! Transaction ID: {data.transaction_id}',
        link='/dashboard',
        details={
            'payout_id': str(payout.id),
            'amount': float(payout.net_amount),
            'transaction_id': data.transaction_id,
            'method': payout.payout_method
        },
        read=False
    )
    db.add(notification)
    await db.commit()

    return {
        "message": "Payout completed successfully",
        "payout_id": str(payout.id),
        "transaction_id": data.transaction_id,
        "status": "completed"
    }


@app.post("/api/admin/payouts/{payout_id}/fail")
async def fail_payout(
    payout_id: str,
    data: PayoutFailRequest,
    admin: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Admin: Mark payout as failed"""
    try:
        payout_uuid = uuid.UUID(payout_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid payout ID")

    payout_result = await db.execute(
        select(DeveloperPayout).where(DeveloperPayout.id == payout_uuid)
    )
    payout = payout_result.scalar_one_or_none()

    if not payout:
        raise HTTPException(status_code=404, detail="Payout not found")

    # Update payout status
    payout.status = 'failed'
    payout.failure_reason = data.failure_reason
    payout.retry_count = (payout.retry_count or 0) + 1
    payout.processed_by = uuid.UUID(admin['id'])
    payout.updated_at = datetime.utcnow()

    await db.commit()

    logger.warning(f"Admin {admin['email']} marked payout {payout_id} as failed: {data.failure_reason}")

    # Notify developer
    notification = Notification(
        user_id=payout.developer_id,
        type='payout_failed',
        title='⚠️ Payout Failed',
        message=f'Your payout of ${float(payout.net_amount):.2f} could not be processed. Reason: {data.failure_reason}. Please check your payout details.',
        link='/dashboard',
        details={
            'payout_id': str(payout.id),
            'amount': float(payout.net_amount),
            'failure_reason': data.failure_reason
        },
        read=False
    )
    db.add(notification)
    await db.commit()

    return {
        "message": "Payout marked as failed",
        "payout_id": str(payout.id),
        "failure_reason": data.failure_reason,
        "status": "failed"
    }


@app.post("/api/admin/payouts/{payout_id}/retry")
async def retry_payout(
    payout_id: str,
    admin: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Admin: Retry a failed payout (reset to pending)"""
    try:
        payout_uuid = uuid.UUID(payout_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid payout ID")

    payout_result = await db.execute(
        select(DeveloperPayout).where(DeveloperPayout.id == payout_uuid)
    )
    payout = payout_result.scalar_one_or_none()

    if not payout:
        raise HTTPException(status_code=404, detail="Payout not found")

    if payout.status != 'failed':
        raise HTTPException(
            status_code=400,
            detail=f"Can only retry failed payouts. Current status: '{payout.status}'"
        )

    # Reset to pending
    payout.status = 'pending'
    payout.failure_reason = None
    payout.transaction_id = None
    payout.transaction_notes = None
    payout.processed_at = None
    payout.updated_at = datetime.utcnow()

    await db.commit()

    logger.info(f"Admin {admin['email']} reset payout {payout_id} to pending for retry")

    return {
        "message": "Payout reset to pending for retry",
        "payout_id": str(payout.id),
        "status": "pending"
    }


@app.put("/api/admin/payouts/{payout_id}/verify-method")
async def verify_payout_method(
    payout_id: str,
    admin: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Admin: Verify developer's payout method/details"""
    try:
        payout_uuid = uuid.UUID(payout_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid payout ID")

    payout_result = await db.execute(
        select(DeveloperPayout).where(DeveloperPayout.id == payout_uuid)
    )
    payout = payout_result.scalar_one_or_none()

    if not payout:
        raise HTTPException(status_code=404, detail="Payout not found")

    # Update developer's payout_verified status
    await db.execute(
        update(User)
        .where(User.id == payout.developer_id)
        .values(payout_verified=True, updated_at=datetime.utcnow())
    )
    await db.commit()

    logger.info(f"Admin {admin['email']} verified payout method for developer {payout.developer_id}")

    return {
        "message": "Developer payout method verified",
        "developer_id": str(payout.developer_id)
    }


# ------------------------------
# BROADCAST MESSAGE (Admin Only)
# ------------------------------
class BroadcastMessage(BaseModel):
    title: str
    message: str
    audience: str = "all"  # all, developers, investors


@app.post("/api/admin/broadcasts")
async def admin_create_broadcast(
    data: BroadcastMessage,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """Create broadcast notification and send to all target users via WebSocket"""
    
    # Get target users
    query = select(User)

    if data.audience == "developers":
        query = query.where(User.role == "developer")
    elif data.audience == "investors":
        query = query.where(User.role == "investor")

    result = await db.execute(query)
    users = result.scalars().all()

    # Create notifications and send live
    notifications_sent = 0
    for user in users:
        # Create DB notification
        notif = Notification(
            user_id=user.id,
            type="broadcast",
            title=data.title,
            message=data.message,
        )
        db.add(notif)
        
    await db.commit()
    
    # ✅ NEW: Send live notifications via WebSocket
    for user in users:
        try:
            await send_notification_to_user(
                str(user.id),
                {
                    "type": "broadcast",
                    "title": data.title,
                    "message": data.message,
                    "details": {
                        "audience": data.audience,
                        "broadcast_time": datetime.utcnow().isoformat()
                    }
                },
                db
            )
            notifications_sent += 1
        except Exception as e:
            logger.error(f"Failed to send live notification to user {user.id}: {e}")
    
    logger.info(f"✅ Broadcast sent to {len(users)} users ({notifications_sent} live notifications)")

    return {
        "message": f"Broadcast sent to {len(users)} users",
        "live_notifications_sent": notifications_sent,
        "audience": data.audience
    }


# ============================================
# PAYMENT SUCCESS/FAILURE PAGES
# ============================================

@app.get("/demo-payment")
async def demo_payment_page(order: str, amount: float):
    """Demo payment page"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Demo Payment - BiteBids</title>
        <style>
            body {{
                font-family: 'Space Grotesk', Arial, sans-serif;
                background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
                color: white;
                margin: 0;
                padding: 20px;
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
            }}
            .payment-container {{
                background: rgba(0, 212, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 8px 32px 0 rgba(0, 212, 255, 0.37);
                border: 2px solid rgba(0, 212, 255, 0.3);
                max-width: 500px;
                width: 100%;
            }}
            .logo {{
                font-size: 32px;
                font-weight: bold;
                text-align: center;
                margin-bottom: 30px;
                background: linear-gradient(135deg, #00d4ff, #0099cc);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            .btn {{
                background: linear-gradient(135deg, #00d4ff, #0099cc);
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 8px;
                font-size: 18px;
                font-weight: bold;
                cursor: pointer;
                width: 100%;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="payment-container">
            <div class="logo">🚀 BiteBids</div>
            <h2>Demo Payment</h2>
            <p>Order: {order}</p>
            <p>Amount: ${amount:.2f}</p>
            <button class="btn" onclick="alert('Demo payment completed!'); window.close();">Complete Payment</button>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/payment/success")
async def payment_success_page():
    """Payment success page"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Payment Successful - BiteBids</title>
        <style>
            body {
                font-family: 'Space Grotesk', Arial, sans-serif;
                background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
                color: white;
                margin: 0;
                padding: 20px;
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .success-container {
                background: rgba(0, 212, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 60px 40px;
                box-shadow: 0 8px 32px 0 rgba(0, 212, 255, 0.37);
                border: 2px solid rgba(0, 212, 255, 0.3);
                max-width: 600px;
                width: 100%;
                text-align: center;
            }
            .success-icon {
                font-size: 80px;
                color: #00ff88;
                margin-bottom: 30px;
            }
            .logo {
                font-size: 32px;
                font-weight: bold;
                margin-bottom: 20px;
                background: linear-gradient(135deg, #00d4ff, #0099cc);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
        </style>
    </head>
    <body>
        <div class="success-container">
            <div class="success-icon">✓</div>
            <div class="logo">🚀 BiteBids</div>
            <h1>Payment Successful!</h1>
            <p>Thank you for your payment. Your transaction has been completed.</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/payment/retry")
async def payment_retry_page():
    """Payment retry page"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Payment Failed - BiteBids</title>
        <style>
            body {
                font-family: 'Space Grotesk', Arial, sans-serif;
                background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
                color: white;
                margin: 0;
                padding: 20px;
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .retry-container {
                background: rgba(255, 193, 7, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 60px 40px;
                box-shadow: 0 8px 32px 0 rgba(255, 193, 7, 0.37);
                border: 2px solid rgba(255, 193, 7, 0.3);
                max-width: 600px;
                width: 100%;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="retry-container">
            <h1>Payment was cancelled or failed</h1>
            <p>Don't worry - you can try again!</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# @app.on_event("startup")
# async def startup_event():
#     """Create database tables on startup"""
#     try:
#         async with engine.begin() as conn:
#             await conn.run_sync(Base.metadata.create_all)
#         logger.info("✅ Database tables initialized successfully")
#     except Exception as e:
#         logger.error(f"❌ Database initialization failed: {e}")
#         raise


if __name__ == "__main__":
    import uvicorn
    import os

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8001)),
        reload=False
    )
