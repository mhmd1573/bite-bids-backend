# app/api/v1/bank_account.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, Field
from typing import Optional
import uuid

from app.database import get_db
from app.models.user import User
from app.core.dependencies import get_current_user
from app.core.exceptions import NotFoundException

router = APIRouter(prefix="/bank-account", tags=["Bank Account"])


class BankAccountSaveRequest(BaseModel):
    bank_account_holder: str = Field(..., min_length=1, max_length=255)
    bank_name: str = Field(..., min_length=1, max_length=255)
    bank_account_number: str = Field(..., min_length=1, max_length=255)
    bank_routing_number: str = Field(..., min_length=1, max_length=255)
    bank_iban: Optional[str] = Field(None, max_length=255)
    bank_swift_code: Optional[str] = Field(None, max_length=255)
    bank_currency: str = Field(default='USD', max_length=10)


@router.get("/status")
async def get_bank_account_status(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current user's bank account status"""
    user_result = await db.execute(
        select(User).where(User.id == uuid.UUID(current_user['id']))
    )
    user = user_result.scalar_one_or_none()
    
    if not user:
        raise NotFoundException("User not found")
    
    # Mask account number for security
    account_number = user.bank_account_number or ""
    masked_account = ""
    if len(account_number) > 4:
        masked_account = "****" + account_number[-4:]
    elif account_number:
        masked_account = "****" + account_number
    
    return {
        "has_bank_account": bool(user.bank_account_holder and user.bank_name and user.bank_account_number),
        "bank_account_holder": user.bank_account_holder,
        "bank_name": user.bank_name,
        "bank_account_number": masked_account,
        "bank_routing_number": user.bank_routing_number,
        "bank_iban": user.bank_iban,
        "bank_swift_code": user.bank_swift_code,
        "bank_currency": user.bank_currency or "USD",
        "total_earnings": float(user.total_earnings or 0)
    }


@router.post("/save")
async def save_bank_account(
    bank_data: BankAccountSaveRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Save or update bank account details"""
    user_result = await db.execute(
        select(User).where(User.id == uuid.UUID(current_user['id']))
    )
    user = user_result.scalar_one_or_none()
    
    if not user:
        raise NotFoundException("User not found")
    
    # Update bank fields
    user.bank_account_holder = bank_data.bank_account_holder
    user.bank_name = bank_data.bank_name
    user.bank_account_number = bank_data.bank_account_number
    user.bank_routing_number = bank_data.bank_routing_number
    user.bank_iban = bank_data.bank_iban
    user.bank_swift_code = bank_data.bank_swift_code
    user.bank_currency = bank_data.bank_currency
    
    await db.commit()
    await db.refresh(user)
    
    return {
        "success": True,
        "message": "Bank account details saved successfully",
        "bank_name": user.bank_name,
        "bank_currency": user.bank_currency
    }


@router.post("/verify")
async def verify_bank_account(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Verify bank account (placeholder - verifies required fields are filled)"""
    user_result = await db.execute(
        select(User).where(User.id == uuid.UUID(current_user['id']))
    )
    user = user_result.scalar_one_or_none()
    
    if not user:
        raise NotFoundException("User not found")
    
    # Check required fields
    missing_fields = []
    if not user.bank_account_holder:
        missing_fields.append("Account holder name")
    if not user.bank_name:
        missing_fields.append("Bank name")
    if not user.bank_account_number:
        missing_fields.append("Account number")
    if not user.bank_routing_number:
        missing_fields.append("Routing number")
    
    if missing_fields:
        return {
            "verified": False,
            "missing_fields": missing_fields,
            "message": f"Missing required fields: {', '.join(missing_fields)}"
        }
    
    return {
        "verified": True,
        "message": "Bank account details are complete and ready for payouts"
    }