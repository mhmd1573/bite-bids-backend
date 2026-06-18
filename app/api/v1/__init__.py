# app/api/v1/__init__.py
from fastapi import APIRouter
from app.api.v1 import (
    auth,
    users,
    projects,
    bids,
    chat,
    payments,
    admin,
    notifications,
    disputes,
    contact,
    github,
    uploads,
    stripe,
    oauth,
)

# Create main router
router = APIRouter()

# Include all sub-routers
router.include_router(auth.router)
router.include_router(users.router)
router.include_router(projects.router)
router.include_router(bids.router)
router.include_router(chat.router)
router.include_router(payments.router)
router.include_router(admin.router)
router.include_router(notifications.router)
router.include_router(disputes.router)
router.include_router(contact.router)
router.include_router(github.router)
router.include_router(uploads.router)
router.include_router(stripe.router)
router.include_router(oauth.router)

__all__ = ["router"]