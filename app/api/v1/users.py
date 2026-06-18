# app/api/v1/users.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
import uuid
from datetime import datetime, timezone

from app.database import get_db
from app.models.user import User
from app.schemas.user import UserUpdate, UserResponse
from app.core.dependencies import get_current_user, get_current_admin, get_user_or_self
from app.utils.converters import model_to_dict

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current user's full profile"""
    
    user_id = current_user.get("_id") or current_user.get("id")
    
    if isinstance(user_id, str):
        user_id = uuid.UUID(user_id)
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return model_to_dict(user)


@router.put("/me", response_model=UserResponse)
async def update_current_user_profile(
    user_update: UserUpdate,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
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
        email_change_token = generate_verification_token()
        
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
        
        # TODO: Send verification email (will implement in email service)
        
        # Remove email from update_data as it's pending verification
        user_update.email = None
        
        # Return message about verification
        result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
        user = result.scalar_one_or_none()
        response = model_to_dict(user)
        return {**response, "message": "Verification email sent to your new email address"}
    
    # Prepare update data
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


@router.get("/{user_id}", response_model=UserResponse)
async def get_user_by_id(
    user: User = Depends(get_user_or_self)
):
    """Get user by ID (admin or self)"""
    return model_to_dict(user)


@router.put("/{user_id}", response_model=UserResponse)
async def update_user_by_id(
    user_update: UserUpdate,
    user: User = Depends(get_user_or_self),
    db: AsyncSession = Depends(get_db)
):
    """Update user by ID (admin or self)"""
    
    user_id = str(user.id)
    
    # Check if email is being updated
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
    
    # Prepare update data
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


@router.get("/me/posting-credits")
async def get_posting_credits(
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current user's posting credits"""
    
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