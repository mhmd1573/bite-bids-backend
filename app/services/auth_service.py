# app/services/auth_service.py
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.user import User
from app.core.security import hash_password, verify_password, create_jwt_token, generate_verification_token
from app.core.exceptions import UnauthorizedException, ValidationException, ConflictException
from app.services.email_service import EmailService
from app.utils.converters import model_to_dict


class AuthService:
    """Service for authentication operations"""
    
    @staticmethod
    async def register_user(
        email: str,
        password: str,
        role: str,
        name: str,
        company: Optional[str] = None,
        address: Optional[str] = None,
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """Register a new user"""
        
        # Check if user exists
        result = await db.execute(select(User).where(User.email == email))
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            raise ConflictException("Email already registered")
        
        # Hash password
        hashed_password = hash_password(password)
        
        # Generate verification token
        verification_token = generate_verification_token()
        
        # Create user
        new_user = User(
            email=email,
            password_hash=hashed_password,
            role=role,
            status="pending",
            name=name,
            company=company,
            address=address,
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
        await EmailService.send_verification_email(email, verification_token)
        
        return {
            "message": "Registration successful! Please check your email to verify your account.",
            "user_id": str(new_user.id)
        }
    
    @staticmethod
    async def login_user(
        email: str,
        password: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Login a user"""
        
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        
        if not user:
            raise UnauthorizedException("Invalid credentials")
        
        # Check email verified
        if not user.email_verified:
            raise ValidationException("Please verify your email before logging in.")
        
        # Check password
        if not user.password_hash or not verify_password(password, user.password_hash):
            raise UnauthorizedException("Invalid credentials")
        
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
        
        return {
            "token": token,
            "user": {
                "id": str(user.id),
                "email": user.email,
                "role": user.role,
                "name": user.name,
                "reputation_score": user.reputation_score,
            }
        }
    
    @staticmethod
    async def verify_email(
        token: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Verify user email with token"""
        
        result = await db.execute(select(User).where(User.verification_token == token))
        user = result.scalar_one_or_none()
        
        if not user:
            raise ValidationException("Invalid verification token")
        
        user.email_verified = True
        user.verification_token = None
        user.status = "active"
        user.verified = True
        
        await db.commit()
        
        return {"message": "Email verified successfully! You may now login."}
    
    @staticmethod
    async def get_current_user(
        user_id: str,
        db: AsyncSession
    ) -> Optional[Dict[str, Any]]:
        """Get current user by ID"""
        
        try:
            user_uuid = uuid.UUID(user_id)
        except ValueError:
            return None
        
        result = await db.execute(select(User).where(User.id == user_uuid))
        user = result.scalar_one_or_none()
        
        if not user:
            return None
        
        return model_to_dict(user)