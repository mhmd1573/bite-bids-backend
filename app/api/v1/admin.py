# app/api/v1/admin.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_
import uuid
from datetime import datetime, timedelta

from app.database import get_db
from app.models.user import User
from app.models.project import Project
from app.models.bid import Bid
from app.models.chat import ChatRoom, ChatMessage
from app.models.payment import CheckoutSession, DeveloperPayout
from app.models.dispute import ProjectDisputeSimple
from app.models.notification import Notification
from app.schemas.admin import AdminLogin, BroadcastMessage, BanRequest
from app.schemas.auth import UserCreate
from app.schemas.user import UserUpdate, UserResponse
from app.core.dependencies import get_current_admin, get_current_user
from app.core.security import hash_password, create_jwt_token
from app.core.constants import PROJECT_POSTING_FEE
from app.core.exceptions import NotFoundException, ForbiddenException
from app.utils.converters import model_to_dict
from app.services.notification_service import NotificationService
from app.services.email_service import EmailService
from app.config import settings

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/dashboard")
async def admin_dashboard(
    admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Admin dashboard statistics"""
    try:
        # User statistics
        total_users_result = await db.execute(select(func.count(User.id)))
        total_users = total_users_result.scalar()
        
        developers_result = await db.execute(select(func.count(User.id)).where(User.role == "developer"))
        developers = developers_result.scalar()
        
        investors_result = await db.execute(select(func.count(User.id)).where(User.role == "investor"))
        investors = investors_result.scalar()
        
        # Project statistics
        total_projects_result = await db.execute(select(func.count(Project.id)))
        total_projects = total_projects_result.scalar()
        
        active_projects_result = await db.execute(select(func.count(Project.id)).where(Project.status != "closed"))
        active_projects = active_projects_result.scalar()
        
        # Disputes statistics
        total_disputes_result = await db.execute(select(func.count(ProjectDisputeSimple.id)))
        total_disputes = total_disputes_result.scalar()
        
        active_disputes_result = await db.execute(
            select(func.count(ProjectDisputeSimple.id)).where(ProjectDisputeSimple.resolved == False)
        )
        active_disputes = active_disputes_result.scalar()
        
        # Revenue
        total_revenue_result = await db.execute(select(func.coalesce(func.sum(User.total_earnings), 0)))
        total_revenue = float(total_revenue_result.scalar() or 0)
        
        # Recent activity
        recent_users_result = await db.execute(
            select(User).order_by(User.created_at.desc()).limit(5)
        )
        recent_users = recent_users_result.scalars().all()
        
        recent_projects_result = await db.execute(
            select(Project).order_by(Project.created_at.desc()).limit(5)
        )
        recent_projects = recent_projects_result.scalars().all()
        
        return {
            "stats": {
                "users": {
                    "total": total_users,
                    "developers": developers,
                    "investors": investors
                },
                "projects": {
                    "total": total_projects,
                    "active": active_projects
                },
                "disputes": {
                    "total": total_disputes,
                    "active": active_disputes
                },
                "payments": {
                    "total_revenue": total_revenue,
                    "pending_payments": 0,
                    "completed_transactions": 0
                }
            },
            "recent_activity": {
                "users": model_to_dict(recent_users),
                "projects": model_to_dict(recent_projects)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load admin dashboard: {str(e)}")


@router.get("/users")
async def admin_get_users(
    skip: int = 0,
    limit: int = 50,
    admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get all users with pagination"""
    try:
        total_result = await db.execute(select(func.count(User.id)))
        total = total_result.scalar()
        
        users_result = await db.execute(
            select(User).offset(skip).limit(limit)
        )
        users = users_result.scalars().all()
        
        return {
            "users": model_to_dict(users),
            "total": total,
            "page": skip // limit + 1,
            "pages": (total + limit - 1) // limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch users: {str(e)}")


@router.post("/create-user")
async def admin_create_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
    admin: dict = Depends(get_current_admin)
):
    """Admin creates a new user"""
    
    # Check if email exists
    result = await db.execute(select(User).where(User.email == user_data.email))
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash password
    hashed_password = hash_password(user_data.password)
    
    # Create user
    new_user = User(
        email=user_data.email,
        password_hash=hashed_password,
        role=user_data.role,
        name=user_data.name,
        company=user_data.company,
        status=user_data.status or "active",
        reputation_score=0,
        profile={
            "cosmic_theme": "default",
            "avatar": None,
            "bio": ""
        }
    )
    
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    return {
        "message": "User created successfully",
        "user": model_to_dict(new_user)
    }


@router.post("/user/{identifier}/ban")
async def admin_ban_user(
    identifier: str,
    data: BanRequest,
    db: AsyncSession = Depends(get_db),
    admin: dict = Depends(get_current_admin)
):
    """Ban user by ID or email"""
    
    user = None
    
    # Try UUID first
    try:
        user_uuid = uuid.UUID(identifier)
        result = await db.execute(select(User).where(User.id == user_uuid))
        user = result.scalar_one_or_none()
    except ValueError:
        user = None
    
    # If not UUID, treat as email
    if not user:
        result = await db.execute(select(User).where(User.email == identifier))
        user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Ban user
    user.status = "banned"
    user.updated_at = datetime.utcnow()
    
    await db.commit()
    
    return {"message": f"User {user.email} banned successfully"}


@router.post("/user/{user_id}/unban")
async def admin_unban_user(
    user_id: str,
    admin: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Unban a user"""
    try:
        user_uuid = uuid.UUID(user_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    
    result = await db.execute(select(User).where(User.id == user_uuid))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.status != "banned":
        raise HTTPException(status_code=400, detail="User is not banned")
    
    user.status = "active"
    user.updated_at = datetime.utcnow()
    await db.commit()
    
    return {"message": "User has been unbanned successfully", "user_id": user_id}


@router.post("/broadcasts")
async def admin_create_broadcast(
    data: BroadcastMessage,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """Create broadcast notification and send to all target users"""
    
    # Get target users
    query = select(User)
    
    if data.audience == "developers":
        query = query.where(User.role == "developer")
    elif data.audience == "investors":
        query = query.where(User.role == "investor")
    
    result = await db.execute(query)
    users = result.scalars().all()
    
    # Create notifications
    for user in users:
        notif = Notification(
            user_id=user.id,
            type="broadcast",
            title=data.title,
            message=data.message,
            details={"audience": data.audience}
        )
        db.add(notif)
    
    await db.commit()
    
    # Send live notifications via WebSocket
    notifications_sent = 0
    for user in users:
        try:
            await send_notification_to_user(
                str(user.id),
                {
                    "type": "broadcast",
                    "title": data.title,
                    "message": data.message,
                    "details": {
                        "audience": data.audience,
                        "broadcast_time": datetime.utcnow().isoformat()
                    }
                },
                db
            )
            notifications_sent += 1
        except:
            pass
    
    return {
        "message": f"Broadcast sent to {len(users)} users",
        "live_notifications_sent": notifications_sent,
        "audience": data.audience
    }


@router.post("/users/{user_id}/add-credits")
async def admin_add_posting_credits(
    user_id: str,
    credits: int = 1,
    admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Admin manually adds posting credits to a user"""
    try:
        user_uuid = uuid.UUID(user_id)
        
        result = await db.execute(select(User).where(User.id == user_uuid))
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        user.posting_credits = (user.posting_credits or 0) + credits
        await db.commit()
        await db.refresh(user)
        
        # Send notification
        await send_notification_to_user(
            user_id,
            {
                "type": "credits_added",
                "title": "Posting Credits Added! 🎉",
                "message": f"Admin has added {credits} posting credit(s) to your account.",
                "link": "/dashboard",
                "details": {
                    "credits_added": credits,
                    "total_credits": user.posting_credits
                }
            },
            db
        )
        
        return {
            "success": True,
            "message": f"Added {credits} credits to user",
            "new_total": user.posting_credits
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add credits: {str(e)}")


@router.get("/transactions")
async def admin_get_transactions(
    skip: int = 0,
    limit: int = 50,
    admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get all transactions"""
    try:
        total_result = await db.execute(select(func.count(CheckoutSession.id)))
        total = total_result.scalar()
        
        transactions_result = await db.execute(
            select(CheckoutSession).offset(skip).limit(limit)
        )
        transactions = transactions_result.scalars().all()
        
        return {
            "transactions": model_to_dict(transactions),
            "total": total,
            "page": skip // limit + 1,
            "pages": (total + limit - 1) // limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch transactions: {str(e)}")


@router.get("/payouts")
async def admin_get_payouts(
    status: str = None,
    skip: int = 0,
    limit: int = 50,
    admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get all developer payouts"""
    try:
        query = select(DeveloperPayout)
        
        if status:
            query = query.where(DeveloperPayout.status == status)
        
        query = query.order_by(DeveloperPayout.created_at.desc()).offset(skip).limit(limit)
        
        result = await db.execute(query)
        payouts = result.scalars().all()
        
        # Get developer info for each payout
        payouts_with_info = []
        for p in payouts:
            dev_result = await db.execute(
                select(User).where(User.id == p.developer_id)
            )
            developer = dev_result.scalar_one_or_none()
            
            project_title = None
            if p.project_id:
                proj_result = await db.execute(
                    select(Project).where(Project.id == p.project_id)
                )
                project = proj_result.scalar_one_or_none()
                project_title = project.title if project else None
            
            payouts_with_info.append({
                "id": str(p.id),
                "developer": {
                    "id": str(p.developer_id),
                    "name": developer.name if developer else "Unknown",
                    "email": developer.email if developer else "Unknown"
                },
                "project_id": str(p.project_id) if p.project_id else None,
                "project_title": project_title,
                "gross_amount": float(p.gross_amount),
                "platform_fee": float(p.platform_fee),
                "net_amount": float(p.net_amount),
                "currency": p.currency,
                "status": p.status,
                "failure_reason": p.failure_reason,
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "completed_at": p.completed_at.isoformat() if p.completed_at else None
            })
        
        return {
            "payouts": payouts_with_info,
            "total": len(payouts_with_info)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch payouts: {str(e)}")