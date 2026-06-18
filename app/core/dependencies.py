# app/core/dependencies.py
import uuid
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.core.security import decode_jwt_token
from app.models.user import User
from app.core.exceptions import UnauthorizedException, ForbiddenException, NotFoundException

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> dict:
    """Get current user from JWT token"""
    try:
        token = credentials.credentials
        payload = decode_jwt_token(token)
        user_id = payload.get("user_id")
        
        if not user_id:
            raise UnauthorizedException("Invalid token")
        
        # Convert user_id to UUID if it's a string
        if isinstance(user_id, str):
            user_id = uuid.UUID(user_id)
        
        # Query user from database
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        
        if not user:
            raise UnauthorizedException("User not found")
        
        # Return user as dictionary
        from app.utils.converters import model_to_dict
        user_dict = model_to_dict(user)
        
        # Ensure user_id is a string in the returned dict
        if 'id' in user_dict and isinstance(user_dict['id'], uuid.UUID):
            user_dict['id'] = str(user_dict['id'])
        if '_id' in user_dict and isinstance(user_dict['_id'], uuid.UUID):
            user_dict['_id'] = str(user_dict['_id'])
            
        return user_dict
        
    except ValueError as e:
        raise UnauthorizedException(f"Invalid user ID format: {str(e)}")


async def get_current_admin(
    current_user: dict = Depends(get_current_user)
) -> dict:
    """Verify current user is admin"""
    if current_user.get("role") != "admin":
        raise ForbiddenException("Admin access required")
    return current_user


async def get_user_or_self(
    user_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Get user if admin or if user is requesting their own data"""
    current_user_id = current_user.get("_id") or current_user.get("id")
    
    # Allow access if user is requesting their own data or is admin
    if user_id == current_user_id or current_user.get("role") == "admin":
        result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
        user = result.scalar_one_or_none()
        
        if not user:
            raise NotFoundException("User not found")
        
        return user
    
    raise ForbiddenException("Access denied")