# app/services/payoneer.py
"""
Payoneer Checkout integration (replaces Stripe Checkout for collecting payments).

Implements the Payoneer "Checkout Server Payment API" (formerly Optile).
Only the "collecting" side (LIST sessions / CHARGE) is used here.
Developer payouts are NOT handled by this module - they use manual bank transfer.

Key API facts (from docs):
  - Base URL: https://api.sandbox.oscato.com/api (sandbox) / https://api.oscato.com/api (live)
  - Create a payment session:  POST /lists
  - Get session state:          GET  /lists/{listId}
  - Auth:                       Authorization header (Bearer / payment_auth token)
"""
import logging
from typing import Dict, Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# Statuses at which we consider the payment successfully charged.
# `charged` is the Payoneer LIST/CHARGE status after a successful card payment.
PAYONEER_SUCCESS_STATUSES = {
    "charged",
    "charge_successful",
    "paid",
    "captured",
    "success",
}


def payoneer_base_url() -> str:
    return (settings.PAYONEER_API_BASE_URL or "https://api.sandbox.oscato.com/api").rstrip("/")


def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.PAYONEER_AUTH_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/vnd.optile.payment.enterprise-v1-extensible+json",
    }


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.status_code >= 400:
        logger.error(
            "Payoneer API error: %s %s -> %s",
            resp.request.method,
            resp.request.url,
            resp.text[:1000],
        )
        resp.raise_for_status()


async def create_payment_session(payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST /lists - initialize a payment (LIST) session."""
    url = f"{payoneer_base_url()}/lists"
    async with httpx.AsyncClient(timeout=40) as client:
        resp = await client.post(url, json=payload, headers=_headers())
        _raise_for_status(resp)
        return resp.json()


async def get_session(list_id: str) -> Dict[str, Any]:
    """GET /lists/{listId} - retrieve current state of a LIST session."""
    url = f"{payoneer_base_url()}/lists/{list_id}"
    async with httpx.AsyncClient(timeout=40) as client:
        resp = await client.get(url, headers=_headers())
        _raise_for_status(resp)
        return resp.json()


def is_payment_successful(session: Dict[str, Any]) -> bool:
    """Determine whether a Payoneer session reached a successful/charged state."""
    status = (session or {}).get("status") or {}
    code = (status.get("code") or "").lower()
    if code in PAYONEER_SUCCESS_STATUSES:
        return True
    # Some responses expose the resultCode / interaction after charge
    result_code = (session or {}).get("resultCode") or ""
    if "charged" in result_code.lower() or "capture" in result_code.lower():
        return True
    return False


def extract_list_id(session: Dict[str, Any]) -> str:
    """Get the Payoneer longId from a LIST response."""
    identification = (session or {}).get("identification") or {}
    return identification.get("longId") or ""


def extract_redirect_url(session: Dict[str, Any]) -> str:
    """Get the hosted payment page URL from a LIST/CHARGE response."""
    redirect = (session or {}).get("redirect") or {}
    return redirect.get("url") or ""
