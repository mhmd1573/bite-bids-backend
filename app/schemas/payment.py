# app/schemas/payment.py
from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime

class PaymentRequest(BaseModel):
    """Used when initiating payment"""
    order_type: str  # 'auction' or 'fixed'
    item_id: str
    customer_email: str
    customer_name: str
    billing_address: Dict[str, str]
    payment_method: str
    amount: float
    auction_id: Optional[str] = None

class CheckoutSessionResponse(BaseModel):
    """Response for checkout session creation"""
    session_id: str
    order_reference: str
    payment_url: str
    expires_at: datetime

class PaymentResponse(BaseModel):
    """Generic payment response"""
    success: bool
    session_id: Optional[str] = None
    payment_url: Optional[str] = None
    order_reference: Optional[str] = None
    message: str

class StripeCheckoutRequest(BaseModel):
    """Used when creating Stripe checkout session"""
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