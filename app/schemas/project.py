# app/schemas/project.py
from pydantic import BaseModel
from typing import Optional, List

class ProjectCreate(BaseModel):
    """Used when creating a new project"""
    title: str
    status: str
    description: str
    tech_stack: List[str]
    requirements: str = ""
    budget: float
    lowest_bid: float
    budget_range: Optional[str] = None
    deadline: Optional[str] = None
    location: Optional[str] = "Remote"
    category: str = "Machine Learning"
    images: Optional[List[str]] = []

class ProjectUpdate(BaseModel):
    """Used when updating a project"""
    title: Optional[str] = None
    description: Optional[str] = None
    tech_stack: Optional[List[str]] = None
    requirements: Optional[str] = None
    budget: Optional[float] = None
    lowest_bid: Optional[float] = None
    budget_range: Optional[str] = None
    deadline: Optional[str] = None
    location: Optional[str] = None
    category: Optional[str] = None
    status: Optional[str] = None
    images: Optional[List[str]] = None