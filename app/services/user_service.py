# app/services/user_service.py
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.models.user import User
from app.core.security import generate_verification_token
from app.core.exceptions import NotFoundException, ConflictException
from app.services.email_service import EmailService
from app.utils.converters import model_to_dict


class UserService:
    """Service for user management operations"""
    
    @staticmethod
    async def get_user_by_id(
        user_id: str,
        db: AsyncSession
    ) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        
        try:
            user_uuid = uuid.UUID(user_id)
        except ValueError:
            return None
        
        result = await db.execute(select(User).where(User.id == user_uuid))
        user = result.scalar_one_or_none()
        
        if not user:
            return None
        
        return model_to_dict(user)
    
    @staticmethod
    async def update_user_profile(
        user_id: str,
        update_data: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Update user profile"""
        
        try:
            user_uuid = uuid.UUID(user_id)
        except ValueError:
            raise NotFoundException("Invalid user ID")
        
        result = await db.execute(select(User).where(User.id == user_uuid))
        user = result.scalar_one_or_none()
        
        if not user:
            raise NotFoundException("User not found")
        
        # Check if email is being updated
        if update_data.get("email") and update_data["email"] != user.email:
            # Check if new email is already taken
            email_check = await db.execute(
                select(User).where(
                    User.email == update_data["email"],
                    User.id != user_uuid
                )
            )
            existing_user = email_check.scalar_one_or_none()
            if existing_user:
                raise ConflictException("Email already registered")
            
            # Generate email change verification token
            email_change_token = generate_verification_token()
            
            # Store pending email
            user.pending_email = update_data["email"]
            user.email_change_token = email_change_token
            user.email_change_sent_at = datetime.now(timezone.utc).replace(tzinfo=None)
            
            # Remove email from update_data
            del update_data["email"]
            
            # Send verification email (async)
            await EmailService.send_verification_email(
                user.pending_email,
                email_change_token,
                is_email_change=True
            )
        
        # Apply updates
        if update_data:
            update_data['updated_at'] = datetime.now(timezone.utc).replace(tzinfo=None)
            
            for key, value in update_data.items():
                if value is not None:
                    setattr(user, key, value)
        
        await db.commit()
        await db.refresh(user)
        
        return model_to_dict(user)
    
    @staticmethod
    async def get_posting_credits(
        user_id: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Get user's posting credits"""
        
        try:
            user_uuid = uuid.UUID(user_id)
        except ValueError:
            raise NotFoundException("Invalid user ID")
        
        result = await db.execute(
            select(User.posting_credits).where(User.id == user_uuid)
        )
        credits = result.scalar_one_or_none() or 0
        
        return {
            "credits": credits,
            "has_credit": credits > 0
        }