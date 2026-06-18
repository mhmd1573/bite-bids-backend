# app/api/v1/stripe.py
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
import uuid
from datetime import datetime

from app.database import get_db
from app.models.user import User
from app.models.payment import DeveloperPayout
from app.models.notification import Notification
from app.core.dependencies import get_current_user, get_current_admin
from app.core.exceptions import NotFoundException, ForbiddenException
from app.utils.converters import model_to_dict
from app.services.notification_service import NotificationService
from app.config import settings

router = APIRouter(prefix="/stripe-connect", tags=["Stripe Connect"])


@router.get("/account-status")
async def get_stripe_connect_status(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current user's Stripe Connect account status"""
    import stripe
    stripe.api_key = settings.STRIPE_SECRET_KEY
    
    user_result = await db.execute(
        select(User).where(User.id == uuid.UUID(current_user['id']))
    )
    user = user_result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # If user has a Stripe account, fetch latest status from Stripe
    if user.stripe_account_id:
        try:
            account = stripe.Account.retrieve(user.stripe_account_id)
            user.stripe_payouts_enabled = account.payouts_enabled
            user.stripe_account_status = 'enabled' if account.payouts_enabled else 'pending'
            user.stripe_onboarding_completed = account.details_submitted
            await db.commit()
        except:
            pass
    
    return {
        "stripe_account_id": user.stripe_account_id,
        "stripe_account_status": user.stripe_account_status,
        "stripe_payouts_enabled": user.stripe_payouts_enabled or False,
        "stripe_onboarding_completed": user.stripe_onboarding_completed or False,
        "total_earnings": float(user.total_earnings or 0)
    }


@router.post("/create-account")
async def create_stripe_connect_account(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a Stripe Connect Express account for a developer"""
    import stripe
    stripe.api_key = settings.STRIPE_SECRET_KEY
    
    user_result = await db.execute(
        select(User).where(User.id == uuid.UUID(current_user['id']))
    )
    user = user_result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if user already has a Stripe account
    if user.stripe_account_id:
        try:
            account_link = stripe.AccountLink.create(
                account=user.stripe_account_id,
                refresh_url=f"{settings.FRONTEND_URL}/payout-settings?stripe_refresh=true",
                return_url=f"{settings.FRONTEND_URL}/payout-settings?stripe_success=true",
                type="account_onboarding",
            )
            return {"onboarding_url": account_link.url, "account_id": user.stripe_account_id}
        except:
            pass
    
    # Create new Stripe Express account
    try:
        account = stripe.Account.create(
            type="express",
            country="US",
            email=user.email,
            capabilities={
                "transfers": {"requested": True},
            },
            business_type="individual",
            metadata={
                "bitebids_user_id": str(user.id),
                "bitebids_email": user.email
            }
        )
        
        # Save account ID to user
        user.stripe_account_id = account.id
        user.stripe_account_status = "pending"
        user.stripe_payouts_enabled = False
        user.stripe_onboarding_completed = False
        await db.commit()
        
        # Create account link for onboarding
        account_link = stripe.AccountLink.create(
            account=account.id,
            refresh_url=f"{settings.FRONTEND_URL}/payout-settings?stripe_refresh=true",
            return_url=f"{settings.FRONTEND_URL}/payout-settings?stripe_success=true",
            type="account_onboarding",
        )
        
        return {
            "onboarding_url": account_link.url,
            "account_id": account.id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create Stripe account: {str(e)}")