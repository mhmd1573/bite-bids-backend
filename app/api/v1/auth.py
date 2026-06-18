# app/api/v1/auth.py
from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks, Query
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from datetime import datetime, timedelta, timezone
import uuid
import json
import urllib.parse

from app.database import get_db
from app.models.user import User
from app.schemas.auth import UserCreate, UserLogin, OAuthValidation
from app.core.security import hash_password, verify_password, create_jwt_token, generate_verification_token
from app.core.dependencies import get_current_user
from app.core.exceptions import UnauthorizedException, ForbiddenException, ValidationException
from app.services.email_service import EmailService
from app.config import settings
from app.utils.converters import model_to_dict

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register")
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user with email verification"""
    
    # Check if user exists
    result = await db.execute(select(User).where(User.email == user_data.email))
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash password
    hashed_password = hash_password(user_data.password)
    
    # Generate verification token
    verification_token = generate_verification_token()
    
    # Create user
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
    await EmailService.send_verification_email(new_user.email, verification_token)
    
    return {
        "message": "Registration successful! Please check your email to verify your account."
    }


@router.post("/login")
async def login(
    login_data: UserLogin,
    db: AsyncSession = Depends(get_db)
):
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
    
    # Update last login
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    user.last_login = now_utc
    
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


@router.get("/verify/{token}")
async def verify_email(
    token: str,
    db: AsyncSession = Depends(get_db)
):
    """Verify email with token"""
    
    result = await db.execute(select(User).where(User.verification_token == token))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=400, detail="Invalid verification token")
    
    user.email_verified = True
    user.verification_token = None
    user.status = "active"
    user.verified = True
    
    await db.commit()
    
    return {"message": "Email verified successfully! You may now login."}


@router.get("/me")
async def get_me(
    current_user = Depends(get_current_user)
):
    """Get current user profile"""
    return current_user


@router.put("/role")
async def update_role(
    role: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update user role"""
    
    if role not in ["developer", "investor"]:
        raise HTTPException(status_code=400, detail="Invalid role")
    
    user_id = uuid.UUID(current_user["_id"])
    await db.execute(
        update(User).where(User.id == user_id).values(role=role)
    )
    await db.commit()
    
    return {"message": "Role updated successfully", "role": role}


@router.get("/verify-email-change")
async def verify_email_change(
    token: str = Query(...),
    db: AsyncSession = Depends(get_db)
):
    """Verify email change token and update user's email"""
    
    result = await db.execute(select(User).where(User.email_change_token == token))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=400, detail="Invalid verification token")
    
    if not user.pending_email:
        raise HTTPException(status_code=400, detail="No pending email change")
    
    # Check if token is expired (24 hours)
    if user.email_change_sent_at:
        token_age = datetime.now(timezone.utc).replace(tzinfo=None) - user.email_change_sent_at
        if token_age > timedelta(hours=24):
            raise HTTPException(status_code=400, detail="Verification token has expired")
    
    # Update email
    user.email = user.pending_email
    user.pending_email = None
    user.email_change_token = None
    user.email_change_sent_at = None
    
    await db.commit()
    
    return {
        "success": True,
        "message": "Email updated successfully!",
        "new_email": user.email
    }