# app/core/exceptions.py
from fastapi import HTTPException, status


class BiteBidsException(HTTPException):
    """Base exception for BiteBids application"""
    
    def __init__(self, status_code: int, detail: str, headers: dict = None):
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class NotFoundException(BiteBidsException):
    """Exception for resource not found"""
    
    def __init__(self, detail: str = "Resource not found"):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


class UnauthorizedException(BiteBidsException):
    """Exception for unauthorized access"""
    
    def __init__(self, detail: str = "Unauthorized"):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


class ForbiddenException(BiteBidsException):
    """Exception for forbidden access"""
    
    def __init__(self, detail: str = "Forbidden"):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


class ValidationException(BiteBidsException):
    """Exception for validation errors"""
    
    def __init__(self, detail: str = "Validation error"):
        super().__init__(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)


class ConflictException(BiteBidsException):
    """Exception for conflict (duplicate, etc.)"""
    
    def __init__(self, detail: str = "Conflict"):
        super().__init__(status_code=status.HTTP_409_CONFLICT, detail=detail)


class PaymentRequiredException(BiteBidsException):
    """Exception for payment required"""
    
    def __init__(self, detail: str = "Payment required"):
        super().__init__(status_code=status.HTTP_402_PAYMENT_REQUIRED, detail=detail)