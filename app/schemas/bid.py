# app/schemas/bid.py
from pydantic import BaseModel
from typing import Optional, List

class BidCreate(BaseModel):
    """Used when placing a bid on a project"""
    amount: float

class PricingRequest(BaseModel):
    """Used to calculate pricing"""
    title: str
    description: str
    tech_stack: List[str]
    requirements: str
    complexity: Optional[str] = "medium"