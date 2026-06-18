# app/services/stripe_service.py
import stripe
import logging
from typing import Dict, Any, Optional
from decimal import Decimal
from datetime import datetime, timedelta, timezone

from app.config import settings
from app.core.constants import PLATFORM_FEE_PERCENTAGE, PLATFORM_FIXED_FEE

logger = logging.getLogger(__name__)

# Initialize Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY


class StripeService:
    """Service for Stripe payment processing"""
    
    @staticmethod
    def calculate_fees(amount: float) -> Dict[str, float]:
        """Calculate platform fees for a transaction"""
        fee_percentage = PLATFORM_FEE_PERCENTAGE / 100
        fees = round(amount * fee_percentage + PLATFORM_FIXED_FEE, 2)
        total = amount + fees
        return {
            "base_amount": amount,
            "fees": fees,
            "total": total,
            "platform_fee_percentage": PLATFORM_FEE_PERCENTAGE,
            "platform_fixed_fee": PLATFORM_FIXED_FEE
        }

    @staticmethod
    async def create_checkout_session(
        amount: float,
        customer_email: str,
        item_name: str,
        item_description: str,
        metadata: Dict[str, Any],
        success_url: str,
        cancel_url: str,
        expires_in_minutes: int = 30
    ) -> Dict[str, Any]:
        """Create a Stripe Checkout Session"""
        try:
            fee_calculation = StripeService.calculate_fees(amount)
            amount_cents = int(fee_calculation["total"] * 100)
            
            expiry = datetime.utcnow() + timedelta(minutes=expires_in_minutes)
            expiry_ts = int(expiry.replace(tzinfo=timezone.utc).timestamp())
            
            session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price_data': {
                        'currency': 'usd',
                        'unit_amount': amount_cents,
                        'product_data': {
                            'name': item_name,
                            'description': item_description,
                        },
                    },
                    'quantity': 1,
                }],
                mode='payment',
                success_url=success_url,
                cancel_url=cancel_url,
                customer_email=customer_email,
                metadata={
                    **metadata,
                    "base_amount": str(fee_calculation["base_amount"]),
                    "fees": str(fee_calculation["fees"]),
                    "total_amount": str(fee_calculation["total"])
                },
                payment_intent_data={
                    'metadata': metadata,
                    'description': f'BiteBids payment for {item_name}',
                },
                expires_at=expiry_ts,
            )
            
            return {
                "success": True,
                "session_id": session.id,
                "checkout_url": session.url,
                "amount": fee_calculation["base_amount"],
                "fees": fee_calculation["fees"],
                "total": fee_calculation["total"]
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    @staticmethod
    async def create_connect_account(email: str, user_id: str) -> Dict[str, Any]:
        """Create a Stripe Connect Express account"""
        try:
            account = stripe.Account.create(
                type="express",
                country="US",
                email=email,
                capabilities={
                    "transfers": {"requested": True},
                },
                business_type="individual",
                metadata={
                    "bitebids_user_id": user_id,
                    "bitebids_email": email
                }
            )
            
            return {
                "success": True,
                "account_id": account.id
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe Connect error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    @staticmethod
    async def create_account_link(account_id: str, return_url: str, refresh_url: str) -> Dict[str, Any]:
        """Create a Stripe Connect account onboarding link"""
        try:
            account_link = stripe.AccountLink.create(
                account=account_id,
                refresh_url=refresh_url,
                return_url=return_url,
                type="account_onboarding",
            )
            
            return {
                "success": True,
                "url": account_link.url
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe Account Link error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    @staticmethod
    async def create_transfer(
        amount: float,
        destination_account_id: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a Stripe transfer to a connected account"""
        try:
            transfer = stripe.Transfer.create(
                amount=int(amount * 100),
                currency="usd",
                destination=destination_account_id,
                metadata=metadata
            )
            
            return {
                "success": True,
                "transfer_id": transfer.id,
                "status": transfer.status
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe Transfer error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    @staticmethod
    async def create_refund(
        payment_intent_id: str,
        metadata: Dict[str, Any],
        reason: str = "requested_by_customer"
    ) -> Dict[str, Any]:
        """Create a Stripe refund"""
        try:
            refund = stripe.Refund.create(
                payment_intent=payment_intent_id,
                reason=reason,
                metadata=metadata
            )
            
            return {
                "success": True,
                "refund_id": refund.id,
                "status": refund.status,
                "amount": refund.amount / 100
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe Refund error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    @staticmethod
    async def verify_webhook_signature(payload: bytes, signature: str, webhook_secret: str) -> Dict[str, Any]:
        """Verify Stripe webhook signature"""
        try:
            event = stripe.Webhook.construct_event(
                payload, signature, webhook_secret
            )
            return {
                "success": True,
                "event": event
            }
        except stripe.error.SignatureVerificationError as e:
            logger.error(f"Webhook signature verification error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }