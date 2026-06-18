🍔 BiteBids Backend
A modern, modular backend for the BiteBids project bidding platform built with FastAPI and PostgreSQL.

🚀 Overview
BiteBids is a project bidding platform where developers can post projects and investors can bid on them. This backend provides:

User authentication (JWT-based, OAuth support)
Project management (CRUD operations)
Bidding system (place, accept, reject bids)
Real-time chat (WebSocket support)
Payment processing (Stripe integration)
Dispute resolution (admin-managed)
File uploads (Cloudflare R2 storage)
Email notifications
Content moderation (AI-powered)



bite-bids-backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Application entry point
│   ├── config.py               # Configuration & environment
│   ├── database.py             # Database connection
│   │
│   ├── api/                    # API layer
│   │   ├── __init__.py
│   │   ├── v1/                 # API version 1
│   │   │   ├── __init__.py
│   │   │   ├── auth.py         # Authentication endpoints
│   │   │   ├── users.py        # User management
│   │   │   ├── projects.py     # Project CRUD
│   │   │   ├── bids.py         # Bidding endpoints
│   │   │   ├── chat.py         # Chat endpoints
│   │   │   ├── payments.py     # Payment endpoints
│   │   │   ├── admin.py        # Admin endpoints
│   │   │   ├── notifications.py # Notification endpoints
│   │   │   ├── disputes.py     # Dispute endpoints
│   │   │   ├── contact.py      # Contact form
│   │   │   ├── github.py       # GitHub integration
│   │   │   ├── uploads.py      # File uploads
│   │   │   ├── stripe.py       # Stripe Connect
│   │   │   └── oauth.py        # OAuth endpoints
│   │   └── websocket.py        # WebSocket handlers
│   │
│   ├── models/                 # SQLAlchemy models
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── user.py
│   │   ├── project.py
│   │   ├── bid.py
│   │   ├── chat.py
│   │   ├── payment.py
│   │   ├── dispute.py
│   │   ├── notification.py
│   │   ├── github.py
│   │   ├── upload.py
│   │   └── contact.py
│   │
│   ├── schemas/                # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── user.py
│   │   ├── project.py
│   │   ├── bid.py
│   │   ├── chat.py
│   │   ├── payment.py
│   │   ├── dispute.py
│   │   ├── admin.py
│   │   ├── contact.py
│   │   └── payout.py
│   │
│   ├── core/                   # Core utilities
│   │   ├── __init__.py
│   │   ├── security.py         # JWT, password hashing, encryption
│   │   ├── dependencies.py     # FastAPI dependencies
│   │   ├── rate_limiter.py     # Rate limiting
│   │   ├── websocket_manager.py # WebSocket connection manager
│   │   ├── constants.py        # Application constants
│   │   ├── exceptions.py       # Custom exceptions
│   │   └── logging.py          # Logging configuration
│   │
│   ├── services/               # Business logic
│   │   ├── __init__.py
│   │   ├── auth_service.py
│   │   ├── user_service.py
│   │   ├── project_service.py
│   │   ├── email_service.py
│   │   ├── notification_service.py
│   │   ├── moderation_service.py
│   │   ├── stripe_service.py
│   │   └── chat_service.py
│   │
│   └── utils/                  # Helper functions
│       └── converters.py       # Model to dict conversion
│
├── uploads/                    # Local file storage (ignored by Git)
│   ├── chat_files/
│   ├── project_images/
│   └── temp/
│
│
├── .env                        # Environment variables (ignored)
├── .env.example                # Environment template
├── .gitignore                  # Git ignore file
├── requirements.txt            # Python dependencies
├── run.py                      # Application runner
├── init_postgres.py            # Database initialization
├── README.md                   # This file
└── LICENSE                     # License file




# Clone the repo
git clone https://github.com/mhmd1573/bite-bids-backend.git
cd bite-bids-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your database and other credentials

# Initialize database
python init_postgres.py

# Run the server
python run.py

# Visit http://localhost:8001/docs for API documentation