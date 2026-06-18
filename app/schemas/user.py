# app/schemas/user.py
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime

class RoleUpdate(BaseModel):
    """Used when updating user role"""
    role: str

class UserUpdate(BaseModel):
    """Used when updating user profile"""
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    company: Optional[str] = None
    address: Optional[str] = None
    avatar: Optional[str] = None
    bio: Optional[str] = None
    skills: Optional[List[str]] = None
    profile: Optional[Dict[str, Any]] = None

class UserResponse(BaseModel):
    """Response shape when returning user data"""
    id: str
    email: str
    name: str
    role: str
    company: Optional[str] = None
    address: Optional[str] = None
    avatar: Optional[str] = None
    bio: Optional[str] = None
    skills: Optional[List[str]] = None
    verified: bool
    projects_completed: int
    total_earnings: float
    total_spent: float
    avg_rating: float
    total_reviews: int
    response_rate: int
    on_time_delivery: int
    reputation_score: int
    profile: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    pending_email: Optional[str] = None
    
    class Config:
        from_attributes = True

class UserListResponse(BaseModel):
    """Response for paginated user list"""
    users: List[UserResponse]
    total: int
    page: int
    page_size: int