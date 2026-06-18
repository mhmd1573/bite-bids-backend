# app/api/__init__.py
from app.api.v1 import router
from app.api.v1.websocket import router as websocket_router

__all__ = [
    "router",
    "websocket_router",
]