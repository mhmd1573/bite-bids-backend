# app/core/constants.py

# Platform Fees
PLATFORM_FEE_PERCENTAGE = 6  # 6%
PLATFORM_FIXED_FEE = 30      # $30 fixed fee
PROJECT_POSTING_FEE = 0.99   # $0.99 fee for posting a project

# File Upload
MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']

# User Roles
ROLES = {
    'DEVELOPER': 'developer',
    'INVESTOR': 'investor',
    'ADMIN': 'admin',
}

# Project Status
PROJECT_STATUS = [
    'open',
    'in_progress',
    'completed',
    'closed',
    'cancelled',
    'fixed_price',
    'disputed',
]

# Bid Status
BID_STATUS = [
    'pending',
    'accepted',
    'rejected',
    'withdrawn',
]

# Chat Room Status
CHAT_ROOM_STATUS = [
    'active',
    'closed',
    'archived',
]

# Notification Types
NOTIFICATION_TYPES = [
    'bid_received',
    'bid_accepted',
    'bid_rejected',
    'payment_required',
    'payment_received',
    'payment_verified',
    'project_approved',
    'project_edited_by_admin',
    'dispute_opened',
    'dispute_resolved',
    'payout_processing',
    'payout_completed',
    'payout_failed',
    'broadcast',
    'chat_room_created',
    'credits_added',
]

# Dispute Resolutions
DISPUTE_RESOLUTIONS = [
    'refund_investor',
    'refund_developer',
    'continue_project',
]

# Payment Status
PAYMENT_STATUS = [
    'pending',
    'completed',
    'cancelled',
    'expired',
    'refunded',
]

# Payout Status
PAYOUT_STATUS = [
    'pending',
    'processing',
    'completed',
    'failed',
    'cancelled',
]