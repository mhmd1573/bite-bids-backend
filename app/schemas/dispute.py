# app/schemas/dispute.py
from pydantic import BaseModel
from typing import Optional, List

class DeliverySubmitRequest(BaseModel):
    """Used when submitting project delivery"""
    delivery_url: Optional[str] = None
    delivery_notes: str

class DeliveryApprovalRequest(BaseModel):
    """Used when approving/rejecting delivery"""
    approved: bool
    feedback: Optional[str] = None

class DisputeCreateRequest(BaseModel):
    """Used when creating a dispute"""
    reason: str
    notes: Optional[str] = None
    evidence: Optional[List[str]] = None

class DisputeResolveRequest(BaseModel):
    """Used when resolving a dispute (admin)"""
    resolution: str
    admin_notes: str
    developer_payout: Optional[float] = None
    investor_refund: Optional[float] = None