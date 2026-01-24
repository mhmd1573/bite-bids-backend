"""
BiteBids PostgreSQL Initialization with SQLAlchemy
Complete setup with models, migrations, and sample data
"""

import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, Integer, Boolean, DECIMAL, TIMESTAMP, Text, ARRAY, ForeignKey, CheckConstraint, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB, INET
from sqlalchemy.sql import func
import uuid
from datetime import datetime, timedelta
import bcrypt
import random

# Database URL
DATABASE_URL = "postgresql+asyncpg://postgres:root@localhost:5432/bitebids"

# SQLAlchemy setup
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# ============================================
# SQLALCHEMY MODELS
# ============================================

class User(Base):
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=True)  # Nullable for OAuth users
    name = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, index=True)
    company = Column(String(255))

    status = Column(String(50), default="active", index=True)

    
    # Developer fields
    account_name = Column(String(255))
    iban = Column(String(50))
    swift_code = Column(String(20))
    bank_name = Column(String(255))
    address = Column(Text)
    
    # Profile
    avatar = Column(String(500))
    bio = Column(Text)
    skills = Column(ARRAY(Text))
    verified = Column(Boolean, default=False)
    verification_date = Column(TIMESTAMP)
    
    # Statistics
    projects_completed = Column(Integer, default=0)
    total_earnings = Column(DECIMAL(12,2), default=0)
    total_spent = Column(DECIMAL(12,2), default=0)
    avg_rating = Column(DECIMAL(3,2), default=0)
    total_reviews = Column(Integer, default=0)
    response_rate = Column(Integer, default=100)
    on_time_delivery = Column(Integer, default=100)
    reputation_score = Column(Integer, default=0)
    
    # OAuth
    oauth_provider = Column(String(50))
    oauth_id = Column(String(255))
    
    # Profile data (stored as JSONB)
    profile = Column(JSONB)
    bank_details = Column(JSONB)
    
    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    last_login = Column(TIMESTAMP)
    
    __table_args__ = (
        CheckConstraint("role IN ('developer', 'investor', 'admin')"),
        CheckConstraint("status IN ('active', 'banned', 'suspended', 'pending', 'deleted')")
        Index('idx_users_oauth', 'oauth_provider', 'oauth_id', unique=True, postgresql_where=(oauth_provider != None)),
    )


class Project(Base):
    __tablename__ = 'projects'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    tech_stack = Column(ARRAY(Text), nullable=False)
    requirements = Column(Text, nullable=False)
    budget_range = Column(String(100))
    budget = Column(DECIMAL(12,2), nullable=False, index=True)
    deadline = Column(TIMESTAMP, index=True)
    
    # OWNER = developer (developers create projects, investors bid on them)
    developer_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Status
    status = Column(String(50), default='open', index=True)
    featured = Column(Boolean, default=False)
    priority = Column(String(20), default='medium')
    
    # Assignment
    assigned_to = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'))
    
    # Bids
    bids_count = Column(Integer, default=0)
    highest_bid = Column(DECIMAL(12,2))
    lowest_bid = Column(DECIMAL(12,2))
    
    # Location
    location = Column(String(255))
    remote = Column(Boolean, default=True)
    
    # Categories
    category = Column(String(50), nullable=False, index=True)
    tags = Column(ARRAY(Text))
    
    # Ratings
    rating = Column(DECIMAL(3,2), default=0)
    reviews_count = Column(Integer, default=0)
    
    # Progress
    progress = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now(), index=True)
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    completed_at = Column(TIMESTAMP)
    
    __table_args__ = (
        CheckConstraint("status IN ('open', 'in_progress', 'completed', 'closed', 'cancelled', 'fixed_price', 'disputed')"),
        CheckConstraint("priority IN ('low', 'medium', 'high')"),
        CheckConstraint("progress >= 0 AND progress <= 100"),
        Index('idx_projects_featured', 'featured', postgresql_where=(featured == True)),
    )


class Bid(Base):
    __tablename__ = 'bids'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id', ondelete='CASCADE'), nullable=False, index=True)
    investor_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Bid details
    amount = Column(DECIMAL(12,2), nullable=False)
    proposal = Column(Text, nullable=False)
    timeline = Column(String(100), nullable=False)
    estimated_hours = Column(Integer)
    
    # Status
    status = Column(String(50), default='pending', index=True)
    
    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now(), index=True)
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    accepted_at = Column(TIMESTAMP)
    
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'accepted', 'rejected', 'withdrawn')"),
        UniqueConstraint('project_id', 'investor_id'),
    )


class Auction(Base):
    __tablename__ = 'auctions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    seller_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Auction details
    starting_bid = Column(DECIMAL(12,2), nullable=False)
    current_bid = Column(DECIMAL(12,2))
    minimum_increment = Column(DECIMAL(12,2), default=100)
    reserve_price = Column(DECIMAL(12,2))
    
    # Timing
    start_time = Column(TIMESTAMP, nullable=False, server_default=func.now())
    end_time = Column(TIMESTAMP, nullable=False, index=True)
    duration_days = Column(Integer, nullable=False)
    
    # Status
    status = Column(String(50), default='active', index=True)
    is_hot = Column(Boolean, default=False)
    
    # Engagement
    bids_count = Column(Integer, default=0)
    highest_bidder_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'))
    watchers_count = Column(Integer, default=0)
    views = Column(Integer, default=0)
    
    # Category
    category = Column(String(100), index=True)
    tags = Column(ARRAY(Text))
    
    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    sold_at = Column(TIMESTAMP)
    
    __table_args__ = (
        CheckConstraint("status IN ('active', 'ended', 'cancelled', 'sold')"),
        Index('idx_auctions_is_hot', 'is_hot', postgresql_where=(is_hot == True)),
    )


class AuctionBid(Base):
    __tablename__ = 'auction_bids'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    auction_id = Column(UUID(as_uuid=True), ForeignKey('auctions.id', ondelete='CASCADE'), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    amount = Column(DECIMAL(12,2), nullable=False)
    timestamp = Column(TIMESTAMP, server_default=func.now(), index=True)
    
    __table_args__ = (
        CheckConstraint("amount > 0"),
    )


class Order(Base):
    __tablename__ = 'orders'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    order_reference = Column(String(100), unique=True, nullable=False, index=True)
    order_type = Column(String(50), nullable=False)
    
    # Parties
    buyer_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='RESTRICT'), nullable=False, index=True)
    buyer_name = Column(String(255), nullable=False)
    buyer_email = Column(String(255), nullable=False)
    seller_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='RESTRICT'), nullable=False, index=True)
    seller_name = Column(String(255), nullable=False)
    
    # Item
    item_id = Column(UUID(as_uuid=True), nullable=False)
    item_title = Column(String(500), nullable=False)
    item_description = Column(Text)
    
    # Pricing
    base_price = Column(DECIMAL(12,2), nullable=False)
    platform_fee = Column(DECIMAL(12,2), nullable=False)
    fixed_fee = Column(DECIMAL(12,2), nullable=False)
    total_amount = Column(DECIMAL(12,2), nullable=False)
    seller_payout = Column(DECIMAL(12,2), nullable=False)
    
    # Payment
    payment_method = Column(String(50), nullable=False)
    payment_status = Column(String(50), default='pending', index=True)
    payment_provider = Column(String(50), default='2checkout')
    payment_session_id = Column(String(255))
    transaction_id = Column(String(255))
    
    # Billing
    billing_address_line1 = Column(String(255), nullable=False)
    billing_city = Column(String(100), nullable=False)
    billing_state = Column(String(100))
    billing_zip = Column(String(20), nullable=False)
    billing_country = Column(String(2), nullable=False)
    
    # Status
    status = Column(String(50), default='pending', index=True)
    delivery_status = Column(String(50), default='pending')
    
    # Files
    files_delivered = Column(Boolean, default=False)
    delivery_date = Column(TIMESTAMP)
    
    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now(), index=True)
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    completed_at = Column(TIMESTAMP)
    cancelled_at = Column(TIMESTAMP)
    
    __table_args__ = (
        CheckConstraint("order_type IN ('auction', 'fixed', 'project')"),
        CheckConstraint("payment_status IN ('pending', 'completed', 'failed', 'refunded')"),
        CheckConstraint("status IN ('pending', 'processing', 'completed', 'cancelled', 'disputed', 'refunded')"),
        CheckConstraint("delivery_status IN ('pending', 'in_progress', 'delivered')"),
    )


class Notification(Base):
    __tablename__ = 'notifications'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Notification details
    type = Column(String(50), nullable=False)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    link = Column(String(500))
    
    # Status
    read = Column(Boolean, default=False)
    read_at = Column(TIMESTAMP)
    
    # Timestamp
    created_at = Column(TIMESTAMP, server_default=func.now(), index=True)
    
    __table_args__ = (
        Index('idx_notifications_read', 'read', postgresql_where=(read == False)),
    )


class ActivityLog(Base):
    __tablename__ = 'activity_log'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'), index=True)
    user_name = Column(String(255))
    
    # Activity
    type = Column(String(50), nullable=False, index=True)
    action = Column(String(255), nullable=False)
    details = Column(JSONB)
    
    # Metadata
    ip_address = Column(INET)
    user_agent = Column(Text)
    
    # Timestamp
    created_at = Column(TIMESTAMP, server_default=func.now(), index=True)


# ============================================
# DATABASE INITIALIZATION
# ============================================

async def init_database():
    """Initialize database with tables"""
    print("🚀 Initializing PostgreSQL database...")
    
    async with engine.begin() as conn:
        # Drop all tables (be careful in production!)
        # await conn.run_sync(Base.metadata.drop_all)
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        
    print("✅ Database tables created successfully!")


async def create_sample_data():
    """Create sample data for testing"""
    print("\n🌱 Creating sample data...")
    
    async with async_session() as session:
        try:
            # Check if data already exists
            from sqlalchemy import select
            result = await session.execute(select(User))
            if result.scalars().first():
                print("⏭️  Sample data already exists. Skipping...")
                return
            
            # Create users
            users = []
            
            # Admin
            admin_password = bcrypt.hashpw("admin123".encode('utf-8'), bcrypt.gensalt())
            admin = User(
                email="admin@bitebids.com",
                password_hash=admin_password.decode('utf-8'),
                name="Admin User",
                role="admin",
                verified=True,
                created_at=datetime.utcnow()
            )
            users.append(admin)
            
            # Developers
            developer_data = [
                ("Sarah Martinez", "sarah.martinez@gmail.com"),
                ("Michael Chen", "michael.chen@gmail.com"),
                ("Emily Rodriguez", "emily.rodriguez@gmail.com"),
                ("Developer", "developer@gmail.com"),
            ]
            
            dev_password = bcrypt.hashpw("developer123".encode('utf-8'), bcrypt.gensalt())
            
            for name, email in developer_data:
                user = User(
                    email=email,
                    password_hash=dev_password.decode('utf-8'),
                    name=name,
                    role="developer",
                    company=f"{name.split()[0]} Tech Solutions",
                    verified=True,
                    skills=["Python", "TensorFlow", "React", "Node.js"],
                    projects_completed=random.randint(5, 30),
                    total_earnings=random.uniform(10000, 100000),
                    avg_rating=round(random.uniform(4.5, 5.0), 2),
                    total_reviews=random.randint(10, 50),
                    created_at=datetime.utcnow() - timedelta(days=random.randint(30, 365))
                )
                users.append(user)
            
            # Investors
            investor_data = [
                ("John Smith", "john.smith@techcorp.com", "TechCorp Inc."),
                ("Lisa Wang", "lisa.wang@datainsights.com", "DataInsights LLC"),
            ]
            
            inv_password = bcrypt.hashpw("investor123".encode('utf-8'), bcrypt.gensalt())
            
            for name, email, company in investor_data:
                user = User(
                    email=email,
                    password_hash=inv_password.decode('utf-8'),
                    name=name,
                    role="investor",
                    company=company,
                    verified=True,
                    total_spent=random.uniform(20000, 200000),
                    avg_rating=round(random.uniform(4.5, 5.0), 2),
                    total_reviews=random.randint(5, 25),
                    created_at=datetime.utcnow() - timedelta(days=random.randint(60, 500))
                )
                users.append(user)
            
            session.add_all(users)
            await session.commit()
            print(f"✅ Created {len(users)} users")
            
            # Refresh to get IDs
            for user in users:
                await session.refresh(user)
            
            # Create projects
            developers = [u for u in users if u.role == "developer"]
            investors = [u for u in users if u.role == "investor"]
            
            project_templates = [
                {
                    "title": "AI Chatbot Development",
                    "description": "Build an intelligent customer service chatbot",
                    "category": "nlp",
                    "budget": 5000.00,
                    "tech_stack": ["Python", "TensorFlow", "NLP"]
                },
                {
                    "title": "Predictive Analytics Model",
                    "description": "ML model for sales forecasting",
                    "category": "ml",
                    "budget": 8500.00,
                    "tech_stack": ["Python", "Scikit-learn", "Pandas"]
                },
            ]
            
            projects = []
            for i, template in enumerate(project_templates):
                developer = developers[i % len(developers)]
                project = Project(
                    title=template["title"],
                    description=template["description"],
                    tech_stack=template["tech_stack"],
                    requirements=f"Requirements for {template['title']}",
                    budget=template["budget"],
                    deadline=datetime.utcnow() + timedelta(days=30),
                    developer_id=developer.id,
                    category=template["category"],
                    status="open",
                    location="San Francisco, CA",
                    created_at=datetime.utcnow() - timedelta(days=random.randint(1, 10))
                )
                projects.append(project)
            
            session.add_all(projects)
            await session.commit()
            print(f"✅ Created {len(projects)} projects")
            
            # Refresh projects
            for project in projects:
                await session.refresh(project)
            
            # Create auctions
            auction_templates = [
                {
                    "title": "Enterprise AI Assistant",
                    "description": "Sophisticated AI assistant for enterprise",
                    "starting_bid": 10000.00,
                    "category": "AI Development"
                },
            ]
            
            auctions = []
            for i, template in enumerate(auction_templates):
                seller = developers[i % len(developers)]
                start_time = datetime.utcnow() - timedelta(hours=12)
                end_time = start_time + timedelta(days=3)
                
                auction = Auction(
                    title=template["title"],
                    description=template["description"],
                    seller_id=seller.id,
                    starting_bid=template["starting_bid"],
                    current_bid=template["starting_bid"] + 1000,
                    start_time=start_time,
                    end_time=end_time,
                    duration_days=3,
                    category=template["category"],
                    status="active",
                    bids_count=5
                )
                auctions.append(auction)
            
            session.add_all(auctions)
            await session.commit()
            print(f"✅ Created {len(auctions)} auctions")
            
            # Create activity logs
            activities = []
            for i, user in enumerate(users[1:4]):  # Skip admin
                activity = ActivityLog(
                    user_id=user.id,
                    user_name=user.name,
                    type="user_registered",
                    action="New user registered",
                    details={"entity_type": "user"},
                    created_at=user.created_at
                )
                activities.append(activity)
            
            session.add_all(activities)
            await session.commit()
            print(f"✅ Created {len(activities)} activity logs")
            
            print("\n✨ Sample data creation complete!")
            print("\n📋 Login Credentials:")
            print("   Admin:     admin@bitebids.com / admin123")
            print("   Developer: sarah.martinez@example.com / developer123")
            print("   Investor:  john.smith@techcorp.com / investor123")
            
        except Exception as e:
            await session.rollback()
            print(f"❌ Error creating sample data: {e}")
            raise


async def main():
    """Main function"""
    try:
        # Initialize database
        await init_database()
        
        # Ask if user wants sample data
        print("\n❓ Do you want to create sample data for testing? (yes/no): ", end='')
        response = input().strip().lower()
        
        if response in ['yes', 'y']:
            await create_sample_data()
        
        print("\n🎉 Setup complete! Your BiteBids PostgreSQL database is ready.")
        print("\n🚀 Update your DATABASE_URL in .env:")
        print("   DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/bitebids")
        print("\n📝 Next steps:")
        print("   1. Update server.py to use PostgreSQL")
        print("   2. Install: pip install sqlalchemy asyncpg psycopg2-binary")
        print("   3. Start your server: python server.py")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())