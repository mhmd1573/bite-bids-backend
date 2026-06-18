# app/services/payoneer_service.py
import httpx
import os
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from cryptography.fernet import Fernet

from app.config import settings

logger = logging.getLogger(__name__)


class PayoneerService:
    """Service for Payoneer Mass Payout API integration"""
    
    def __init__(self):
        self.client_id = settings.PAYONEER_CLIENT_ID
        self.client_secret = settings.PAYONEER_CLIENT_SECRET
        self.partner_id = settings.PAYONEER_PARTNER_ID
        self.program_id = settings.PAYONEER_PROGRAM_ID
        self.base_url = settings.PAYONEER_API_URL or "https://api.sandbox.payoneer.com/v1/"
        self.access_token = None
        self.token_expires_at = None
        
        # Validate required config
        if not all([self.client_id, self.client_secret, self.partner_id, self.program_id]):
            logger.warning("Payoneer credentials not fully configured")
    
    async def _authenticate(self) -> str:
        """Get OAuth 2.0 access token for Payoneer API"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}oauth/token",
                    data={
                        "grant_type": "client_credentials",
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                    },
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded"
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self.access_token = data.get("access_token")
                    self.token_expires_at = datetime.now().timestamp() + data.get("expires_in", 3600)
                    logger.info("✅ Payoneer authentication successful")
                    return self.access_token
                else:
                    logger.error(f"Payoneer authentication failed: {response.text}")
                    raise Exception(f"Authentication failed: {response.text}")
                    
        except Exception as e:
            logger.error(f"Payoneer authentication error: {str(e)}")
            raise
    
    async def _ensure_authenticated(self) -> str:
        """Ensure we have a valid token, refresh if needed"""
        if not self.access_token or datetime.now().timestamp() > self.token_expires_at - 300:
            return await self._authenticate()
        return self.access_token
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict:
        """Make authenticated request to Payoneer API"""
        token = await self._ensure_authenticated()
        
        url = f"{self.base_url}{endpoint.lstrip('/')}"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data,
                    params=params,
                    timeout=60.0
                )
                
                if response.status_code in [200, 201, 202]:
                    return response.json()
                else:
                    logger.error(f"Payoneer API error: {response.status_code} - {response.text}")
                    return {"error": response.text, "status_code": response.status_code}
                    
        except Exception as e:
            logger.error(f"Payoneer request error: {str(e)}")
            return {"error": str(e)}
    
    async def create_payee_signup_link(
        self,
        payee_id: str,
        email: str,
        name: str,
        redirect_url: Optional[str] = None,
        country_code: str = "US"
    ) -> Dict:
        """
        Create a payee registration link for a developer.
        
        This generates a unique URL that the developer can use to create
        their Payoneer account and complete KYC.
        """
        try:
            data = {
                "payeeId": payee_id,
                "email": email,
                "name": name,
                "country": country_code,
                "programId": self.program_id,
            }
            
            if redirect_url:
                data["redirectUrl"] = redirect_url
            
            response = await self._make_request(
                method="POST",
                endpoint="/payees/registration-link",
                data=data
            )
            
            if "error" not in response:
                logger.info(f"✅ Payoneer signup link created for {email}")
                return {
                    "success": True,
                    "link": response.get("registrationLink"),
                    "payee_id": payee_id
                }
            else:
                return {
                    "success": False,
                    "error": response.get("error", "Unknown error")
                }
                
        except Exception as e:
            logger.error(f"Error creating Payoneer signup link: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_payee_status(self, payee_id: str) -> Dict:
        """Check the onboarding status of a payee"""
        try:
            response = await self._make_request(
                method="GET",
                endpoint=f"/payees/{payee_id}/status"
            )
            
            if "error" not in response:
                return {
                    "success": True,
                    "status": response.get("status", "unknown"),
                    "is_active": response.get("status") == "active",
                    "details": response
                }
            else:
                return {
                    "success": False,
                    "error": response.get("error", "Unknown error")
                }
                
        except Exception as e:
            logger.error(f"Error getting payee status: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def initiate_payout(
        self,
        payee_id: str,
        amount: float,
        currency: str = "USD",
        description: str = "Project payment",
        reference_id: Optional[str] = None
    ) -> Dict:
        """
        Initiate a payout to a payee.
        
        The payee must have an active account with completed KYC.
        """
        try:
            data = {
                "payeeId": payee_id,
                "amount": str(amount),
                "currency": currency,
                "description": description,
                "programId": self.program_id,
            }
            
            if reference_id:
                data["referenceId"] = reference_id
            
            response = await self._make_request(
                method="POST",
                endpoint="/payouts",
                data=data
            )
            
            if "error" not in response:
                logger.info(f"✅ Payoneer payout initiated: {response.get('transferId')}")
                return {
                    "success": True,
                    "transfer_id": response.get("transferId"),
                    "quote_id": response.get("quoteId"),
                    "batch_id": response.get("batchId"),
                    "status": response.get("status", "pending"),
                    "estimated_arrival": response.get("estimatedArrival")
                }
            else:
                logger.error(f"Payoneer payout failed: {response.get('error')}")
                return {
                    "success": False,
                    "error": response.get("error", "Unknown error"),
                    "code": response.get("code")
                }
                
        except Exception as e:
            logger.error(f"Error initiating payout: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_payout_status(self, transfer_id: str) -> Dict:
        """Get the status of a payout"""
        try:
            response = await self._make_request(
                method="GET",
                endpoint=f"/payouts/{transfer_id}"
            )
            
            if "error" not in response:
                return {
                    "success": True,
                    "status": response.get("status"),
                    "amount": response.get("amount"),
                    "currency": response.get("currency"),
                    "payee_id": response.get("payeeId"),
                    "completed_at": response.get("completedAt"),
                    "details": response
                }
            else:
                return {
                    "success": False,
                    "error": response.get("error", "Unknown error")
                }
                
        except Exception as e:
            logger.error(f"Error getting payout status: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_payee_balance(self, payee_id: str) -> Dict:
        """Get the balance for a payee (if supported)"""
        try:
            response = await self._make_request(
                method="GET",
                endpoint=f"/payees/{payee_id}/balance"
            )
            
            if "error" not in response:
                return {
                    "success": True,
                    "balances": response.get("balances", []),
                    "currency": response.get("currency"),
                    "amount": response.get("amount")
                }
            else:
                return {
                    "success": False,
                    "error": response.get("error", "Unknown error")
                }
                
        except Exception as e:
            logger.error(f"Error getting payee balance: {str(e)}")
            return {"success": False, "error": str(e)}