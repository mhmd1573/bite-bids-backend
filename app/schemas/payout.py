# app/schemas/payout.py
from pydantic import BaseModel
from typing import Optional, Dict

class PayoutPreferencesUpdate(BaseModel):
    """Used when updating payout preferences"""
    payout_method: str  # paypal, wise, bank_transfer, crypto, other
    payout_email: Optional[str] = None
    payout_details: Optional[dict] = None
    payout_currency: Optional[str] = 'USD'

class PayoutProcessRequest(BaseModel):
    """Used when admin processes a payout"""
    transaction_id: Optional[str] = None
    transaction_notes: Optional[str] = None

class PayoutCompleteRequest(BaseModel):
    """Used when admin marks payout as complete"""
    transaction_id: str
    transaction_notes: Optional[str] = None

class PayoutFailRequest(BaseModel):
    """Used when admin marks payout as failed"""
    failure_reason: str