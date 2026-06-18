# app/schemas/contact.py
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional

class ContactFormSubmission(BaseModel):
    """Used when submitting contact form"""
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    subject: str = Field(..., min_length=3, max_length=200)
    category: str  # general, technical, billing, partnership, feedback
    message: str = Field(..., min_length=10, max_length=5000)
    
    @validator('category')
    def validate_category(cls, v):
        allowed_categories = ['general', 'technical', 'billing', 'partnership', 'feedback']
        if v not in allowed_categories:
            raise ValueError(f'Category must be one of: {", ".join(allowed_categories)}')
        return v
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError('Name must be at least 2 characters')
        return v.strip()
    
    @validator('subject')
    def validate_subject(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError('Subject must be at least 3 characters')
        return v.strip()
    
    @validator('message')
    def validate_message(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError('Message must be at least 10 characters')
        return v.strip()