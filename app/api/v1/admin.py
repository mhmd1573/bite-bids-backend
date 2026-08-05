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
from app.core.websocket_manager import manager
from app.config import settings
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/admin", tags=["Admin"])



# @router.get("/dashboard")
# async def admin_dashboard(
#     admin = Depends(get_current_admin),
#     db: AsyncSession = Depends(get_db)
# ):
#     """Admin dashboard statistics"""
#     try:
#         # User statistics
#         total_users_result = await db.execute(select(func.count(User.id)))
#         total_users = total_users_result.scalar()
        
#         developers_result = await db.execute(select(func.count(User.id)).where(User.role == "developer"))
#         developers = developers_result.scalar()
        
#         investors_result = await db.execute(select(func.count(User.id)).where(User.role == "investor"))
#         investors = investors_result.scalar()
        
#         # Project statistics
#         total_projects_result = await db.execute(select(func.count(Project.id)))
#         total_projects = total_projects_result.scalar()
        
#         active_projects_result = await db.execute(select(func.count(Project.id)).where(Project.status != "closed"))
#         active_projects = active_projects_result.scalar()
        
#         # Disputes statistics
#         total_disputes_result = await db.execute(select(func.count(ProjectDisputeSimple.id)))
#         total_disputes = total_disputes_result.scalar()
        
#         active_disputes_result = await db.execute(
#             select(func.count(ProjectDisputeSimple.id)).where(ProjectDisputeSimple.resolved == False)
#         )
#         active_disputes = active_disputes_result.scalar()
        
#         # Revenue - fix this query
#         # If User.total_earnings doesn't exist, you might need to sum from payments table
#         total_revenue_result = await db.execute(select(func.sum(User.total_earnings)))
#         total_revenue = float(total_revenue_result.scalar() or 0)
        
#         # Recent activity - manually convert to dict
#         recent_users_result = await db.execute(
#             select(User).order_by(User.created_at.desc()).limit(5)
#         )
#         recent_users = recent_users_result.scalars().all()
        
#         recent_projects_result = await db.execute(
#             select(Project).order_by(Project.created_at.desc()).limit(5)
#         )
#         recent_projects = recent_projects_result.scalars().all()
        
#         # Convert users manually
#         users_list = []
#         for user in recent_users:
#             users_list.append({
#                 "id": str(user.id),
#                 "email": user.email,
#                 "name": user.name,
#                 "role": user.role,
#                 "created_at": user.created_at.isoformat() if user.created_at else None
#             })
        
#         # Convert projects manually
#         projects_list = []
#         for project in recent_projects:
#             projects_list.append({
#                 "id": str(project.id),
#                 "title": project.title,
#                 "status": project.status,
#                 "created_at": project.created_at.isoformat() if project.created_at else None
#             })
        
#         return {
#             "stats": {
#                 "users": {
#                     "total": total_users or 0,
#                     "developers": developers or 0,
#                     "investors": investors or 0
#                 },
#                 "projects": {
#                     "total": total_projects or 0,
#                     "active": active_projects or 0
#                 },
#                 "disputes": {
#                     "total": total_disputes or 0,
#                     "active": active_disputes or 0
#                 },
#                 "payments": {
#                     "total_revenue": total_revenue,
#                     "pending_payments": 0,
#                     "completed_transactions": 0
#                 }
#             },
#             "recent_activity": {
#                 "users": users_list,
#                 "projects": projects_list
#             }
#         }
#     except Exception as e:
#         print(f"Error in admin_dashboard: {str(e)}")  # Add logging
#         print(f"Error type: {type(e)}")  # Add logging
#         import traceback
#         traceback.print_exc()  # Print full traceback
#         raise HTTPException(status_code=500, detail=f"Failed to load admin dashboard: {str(e)}")



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
        
        # ✅ CORRECT: Sum platform fees from DeveloperPayout table
        # This gives us the actual platform revenue (6% of each project)
        total_revenue_result = await db.execute(
            select(func.sum(DeveloperPayout.platform_fee))
            .where(DeveloperPayout.status.in_(['completed', 'processing']))
        )
        total_revenue = float(total_revenue_result.scalar() or 0)
        
        # ✅ Pending revenue (approved projects waiting for payout)
        pending_revenue_result = await db.execute(
            select(func.sum(DeveloperPayout.platform_fee))
            .where(DeveloperPayout.status == 'pending')
        )
        pending_revenue = float(pending_revenue_result.scalar() or 0)
        
        # ✅ Completed transactions count
        completed_transactions_result = await db.execute(
            select(func.count(DeveloperPayout.id))
            .where(DeveloperPayout.status == 'completed')
        )
        completed_transactions = completed_transactions_result.scalar() or 0
        
        # ✅ Total transaction volume (all money flowing through platform)
        total_volume_result = await db.execute(
            select(func.sum(DeveloperPayout.gross_amount))
            .where(DeveloperPayout.status.in_(['pending', 'processing', 'completed']))
        )
        total_volume = float(total_volume_result.scalar() or 0)
        
        # Recent activity - manually convert to dict
        recent_users_result = await db.execute(
            select(User).order_by(User.created_at.desc()).limit(5)
        )
        recent_users = recent_users_result.scalars().all()
        
        recent_projects_result = await db.execute(
            select(Project).order_by(Project.created_at.desc()).limit(5)
        )
        recent_projects = recent_projects_result.scalars().all()
        
        # Convert users manually
        users_list = []
        for user in recent_users:
            users_list.append({
                "id": str(user.id),
                "email": user.email,
                "name": user.name,
                "role": user.role,
                "created_at": user.created_at.isoformat() if user.created_at else None
            })
        
        # Convert projects manually
        projects_list = []
        for project in recent_projects:
            projects_list.append({
                "id": str(project.id),
                "title": project.title,
                "status": project.status,
                "created_at": project.created_at.isoformat() if project.created_at else None
            })
        
        return {
            "stats": {
                "users": {
                    "total": total_users or 0,
                    "developers": developers or 0,
                    "investors": investors or 0
                },
                "projects": {
                    "total": total_projects or 0,
                    "active": active_projects or 0
                },
                "disputes": {
                    "total": total_disputes or 0,
                    "active": active_disputes or 0
                },
                "payments": {
                    "total_revenue": total_revenue,  # ✅ Platform's actual revenue (6% fees)
                    "pending_revenue": pending_revenue,  # ✅ Pending platform fees
                    "total_volume": total_volume,  # ✅ Total transaction volume
                    "completed_transactions": completed_transactions
                }
            },
            "recent_activity": {
                "users": users_list,
                "projects": projects_list
            }
        }
    except Exception as e:
        print(f"Error in admin_dashboard: {str(e)}")
        import traceback
        traceback.print_exc()
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
        
        # Convert each user manually to avoid model_to_dict issues
        users_list = []
        for user in users:
            users_list.append({
                "id": str(user.id),
                "email": user.email,
                "name": user.name,
                "role": user.role,
                "status": user.status,
                "company": user.company,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "reputation_score": user.reputation_score,
                "posting_credits": user.posting_credits,
                "total_earnings": float(user.total_earnings) if user.total_earnings else 0
            })
        
        return {
            "users": users_list,
            "total": total,
            "page": skip // limit + 1,
            "pages": (total + limit - 1) // limit
        }
    except Exception as e:
        print(f"Error in admin_get_users: {str(e)}")  # Add logging
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
    
    # 📡 Notify admin dashboards that a user was created
    await manager.broadcast_data_change(
        event_type="admin_update",
        data={"entity": "user", "action": "created", "user_id": str(new_user.id)},
    )
    
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
    
    # 📡 Notify admin dashboards that a user was banned
    await manager.broadcast_data_change(
        event_type="admin_update",
        data={"entity": "user", "action": "banned", "user_id": str(user.id)},
    )
    
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
    
    # 📡 Notify admin dashboards that a user was unbanned
    await manager.broadcast_data_change(
        event_type="admin_update",
        data={"entity": "user", "action": "unbanned", "user_id": user_id},
    )
    
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
        
        # 📡 Notify admin dashboards that credits were added
        await manager.broadcast_data_change(
            event_type="admin_update",
            data={"entity": "credits", "action": "added", "user_id": user_id, "credits": credits},
        )
        
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
                    "email": developer.email if developer else "Unknown",
                    "bank_account_holder": developer.bank_account_holder if developer else None,
                    "bank_name": developer.bank_name if developer else None,
                    "bank_account_number": developer.bank_account_number if developer else None,
                    "bank_routing_number": developer.bank_routing_number if developer else None,
                    "bank_iban": developer.bank_iban if developer else None,
                    "bank_swift_code": developer.bank_swift_code if developer else None,
                    "bank_currency": developer.bank_currency if developer else None
                },
                "project_id": str(p.project_id) if p.project_id else None,
                "project_title": project_title,
                "gross_amount": float(p.gross_amount),
                "platform_fee": float(p.platform_fee),
                "net_amount": float(p.net_amount),
                "currency": p.currency,
                "status": p.status,
                "payout_method": p.payout_method,
                "bank_transfer_reference": p.bank_transfer_reference,
                "admin_notes": p.admin_notes,
                "failure_reason": p.failure_reason,
                "retry_count": p.retry_count,
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "completed_at": p.completed_at.isoformat() if p.completed_at else None
            })
        
        # Calculate stats
        total_query = select(DeveloperPayout)
        all_payouts_result = await db.execute(total_query)
        all_payouts = all_payouts_result.scalars().all()
        
        stats = {
            "pending": {"count": 0, "total": 0.0},
            "processing": {"count": 0, "total": 0.0},
            "completed": {"count": 0, "total": 0.0},
            "failed": {"count": 0, "total": 0.0}
        }
        for p in all_payouts:
            s = p.status
            if s in stats:
                stats[s]["count"] += 1
                stats[s]["total"] += float(p.net_amount)
        
        return {
            "payouts": payouts_with_info,
            "total": len(payouts_with_info),
            "stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch payouts: {str(e)}")


@router.get("/payouts/{payout_id}")
async def admin_get_payout_detail(
    payout_id: str,
    admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed payout information including developer bank details"""
    try:
        payout_uuid = uuid.UUID(payout_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid payout ID")
    
    result = await db.execute(
        select(DeveloperPayout).where(DeveloperPayout.id == payout_uuid)
    )
    payout = result.scalar_one_or_none()
    
    if not payout:
        raise NotFoundException("Payout not found")
    
    # Get developer info
    dev_result = await db.execute(
        select(User).where(User.id == payout.developer_id)
    )
    developer = dev_result.scalar_one_or_none()
    
    # Get project title
    project_title = None
    if payout.project_id:
        proj_result = await db.execute(
            select(Project).where(Project.id == payout.project_id)
        )
        project = proj_result.scalar_one_or_none()
        project_title = project.title if project else None
    
    # Mask bank account number
    account_number = developer.bank_account_number or ""
    masked_account = ""
    if len(account_number) > 4:
        masked_account = "****" + account_number[-4:]
    elif account_number:
        masked_account = account_number
    
    return {
        "id": str(payout.id),
        "developer": {
            "id": str(payout.developer_id),
            "name": developer.name if developer else "Unknown",
            "email": developer.email if developer else "Unknown",
            "bank_details": {
                "account_holder": developer.bank_account_holder,
                "bank_name": developer.bank_name,
                "account_number": masked_account,
                "routing_number": developer.bank_routing_number,
                "iban": developer.bank_iban,
                "swift_code": developer.bank_swift_code,
                "currency": developer.bank_currency or "USD"
            }
        },
        "project_id": str(payout.project_id) if payout.project_id else None,
        "project_title": project_title,
        "gross_amount": float(payout.gross_amount),
        "platform_fee": float(payout.platform_fee),
        "net_amount": float(payout.net_amount),
        "currency": payout.currency,
        "status": payout.status,
        "payout_method": payout.payout_method,
        "bank_transfer_reference": payout.bank_transfer_reference,
        "admin_notes": payout.admin_notes,
        "failure_reason": payout.failure_reason,
        "retry_count": payout.retry_count,
        "transaction_id": payout.transaction_id,
        "created_at": payout.created_at.isoformat() if payout.created_at else None,
        "processed_at": payout.processed_at.isoformat() if payout.processed_at else None,
        "completed_at": payout.completed_at.isoformat() if payout.completed_at else None
    }


class MarkPaidRequest(BaseModel):
    bank_transfer_reference: str
    admin_notes: Optional[str] = None


@router.post("/payouts/{payout_id}/mark-paid")
async def admin_mark_payout_paid(
    payout_id: str,
    request: MarkPaidRequest,
    admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Manually mark a payout as paid with transfer reference"""
    try:
        payout_uuid = uuid.UUID(payout_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid payout ID")
    
    result = await db.execute(
        select(DeveloperPayout).where(DeveloperPayout.id == payout_uuid)
    )
    payout = result.scalar_one_or_none()
    
    if not payout:
        raise NotFoundException("Payout not found")
    
    if payout.status not in ['pending', 'failed']:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot mark payout with status '{payout.status}' as paid. Must be 'pending' or 'failed'."
        )
    
    # Update payout
    payout.status = 'completed'
    payout.bank_transfer_reference = request.bank_transfer_reference
    payout.admin_notes = request.admin_notes
    payout.payout_method = 'bank_transfer'
    payout.processed_by = uuid.UUID(admin['id'])
    payout.processed_at = datetime.utcnow()
    payout.completed_at = datetime.utcnow()
    payout.transaction_id = request.bank_transfer_reference
    
    await db.commit()
    
    # Notify developer
    dev_notification = Notification(
        user_id=payout.developer_id,
        type='payout_completed',
        title=f'✅ Payout Sent: ${float(payout.net_amount):.2f}',
        message=f'Payment of ${float(payout.net_amount):.2f} sent to your bank. Ref: {request.bank_transfer_reference[:20]}',
        link='/dashboard',
        details={
            'amount': float(payout.net_amount),
            'reference': request.bank_transfer_reference,
            'method': 'bank_transfer'
        },
        read=False
    )
    db.add(dev_notification)
    await db.commit()
    
    # 📡 Notify admin dashboards that a payout was marked paid
    await manager.broadcast_data_change(
        event_type="admin_update",
        data={"entity": "payout", "action": "paid", "payout_id": str(payout.id), "project_id": str(payout.project_id)},
    )
    
    return {
        "success": True,
        "message": "Payout marked as paid successfully",
        "payout_id": str(payout.id),
        "bank_transfer_reference": request.bank_transfer_reference
    }


class MarkFailedRequest(BaseModel):
    failure_reason: str
    admin_notes: Optional[str] = None


@router.post("/payouts/{payout_id}/mark-failed")
async def admin_mark_payout_failed(
    payout_id: str,
    request: MarkFailedRequest,
    admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Manually mark a payout as failed"""
    try:
        payout_uuid = uuid.UUID(payout_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid payout ID")
    
    result = await db.execute(
        select(DeveloperPayout).where(DeveloperPayout.id == payout_uuid)
    )
    payout = result.scalar_one_or_none()
    
    if not payout:
        raise NotFoundException("Payout not found")
    
    if payout.status == 'completed':
        raise HTTPException(
            status_code=400,
            detail="Cannot mark a completed payout as failed."
        )
    
    # Update payout
    payout.status = 'failed'
    payout.failure_reason = request.failure_reason
    payout.admin_notes = request.admin_notes
    payout.processed_by = uuid.UUID(admin['id'])
    
    await db.commit()
    
    # 📡 Notify admin dashboards that a payout was marked failed
    await manager.broadcast_data_change(
        event_type="admin_update",
        data={"entity": "payout", "action": "failed", "payout_id": str(payout.id), "project_id": str(payout.project_id)},
    )
    
    return {
        "success": True,
        "message": "Payout marked as failed",
        "payout_id": str(payout.id),
        "failure_reason": request.failure_reason
    }


# Pydantic model for admin chat filters
class AdminChatFilter(BaseModel):
    """Filter options for admin chat viewing"""
    search: Optional[str] = None
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    has_unread: Optional[bool] = None


@router.get("/chat/rooms")
async def admin_get_all_chat_rooms(
    search: Optional[str] = None,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """
    Admin endpoint to get all chat rooms with filters
    Compatible with any ChatRoom schema
    """
    try:
        # Base query
        query = select(ChatRoom)
        
        # Apply filters
        if project_id:
            query = query.where(ChatRoom.project_id == uuid.UUID(project_id))
        
        if user_id:
            query = query.where(
                or_(
                    ChatRoom.developer_id == uuid.UUID(user_id),
                    ChatRoom.investor_id == uuid.UUID(user_id)
                )
            )
        
        # Try to order by updated_at if it exists, otherwise by created_at
        try:
            query = query.order_by(ChatRoom.updated_at.desc())
        except AttributeError:
            try:
                query = query.order_by(ChatRoom.created_at.desc())
            except AttributeError:
                # If neither exists, order by id
                query = query.order_by(ChatRoom.id.desc())
        
        # Pagination
        query = query.offset(skip).limit(limit)
        
        result = await db.execute(query)
        rooms = result.scalars().all()
        
        # Get additional details for each room
        rooms_data = []
        for room in rooms:
            # Fetch project details separately
            project_query = select(Project).where(Project.id == room.project_id)
            project_result = await db.execute(project_query)
            project = project_result.scalar_one_or_none()
            
            # Apply search filter if provided
            if search:
                search_pattern = search.lower()
                project_title = (project.title if project else "").lower()
                if search_pattern not in project_title:
                    continue  # Skip this room if search doesn't match
            
            # Get message count
            msg_count_query = select(func.count(ChatMessage.id)).where(
                ChatMessage.room_id == room.id
            )
            msg_count_result = await db.execute(msg_count_query)
            message_count = msg_count_result.scalar()
            
            # Get unread counts for both users
            dev_unread_query = select(func.count(ChatMessage.id)).where(
                and_(
                    ChatMessage.room_id == room.id,
                    ChatMessage.sender_id != room.developer_id,
                    ChatMessage.read == False
                )
            )
            dev_unread_result = await db.execute(dev_unread_query)
            developer_unread = dev_unread_result.scalar()
            
            inv_unread_query = select(func.count(ChatMessage.id)).where(
                and_(
                    ChatMessage.room_id == room.id,
                    ChatMessage.sender_id != room.investor_id,
                    ChatMessage.read == False
                )
            )
            inv_unread_result = await db.execute(inv_unread_query)
            investor_unread = inv_unread_result.scalar()
            
            # Get developer details
            dev_query = select(User).where(User.id == room.developer_id)
            dev_result = await db.execute(dev_query)
            developer = dev_result.scalar_one_or_none()
            
            # Get investor details
            inv_query = select(User).where(User.id == room.investor_id)
            inv_result = await db.execute(inv_query)
            investor = inv_result.scalar_one_or_none()
            
            # Get last message to determine "updated_at" time
            last_msg_query = select(ChatMessage).where(
                ChatMessage.room_id == room.id
            ).order_by(ChatMessage.created_at.desc()).limit(1)
            last_msg_result = await db.execute(last_msg_query)
            last_message = last_msg_result.scalar_one_or_none()
            
            # Determine updated_at - use last message time or room created_at
            updated_at = None
            if last_message:
                updated_at = last_message.created_at
            elif hasattr(room, 'updated_at'):
                updated_at = room.updated_at
            elif hasattr(room, 'created_at'):
                updated_at = room.created_at
            else:
                updated_at = datetime.utcnow()
            
            # Determine created_at
            created_at = room.created_at if hasattr(room, 'created_at') else datetime.utcnow()
            
            rooms_data.append({
                "id": str(room.id),
                "project_id": str(room.project_id),
                "project_title": project.title if project else "Unknown Project",
                "developer_id": str(room.developer_id),
                "developer_name": developer.name if developer else "Unknown",
                "developer_email": developer.email if developer else "Unknown",
                "investor_id": str(room.investor_id),
                "investor_name": investor.name if investor else "Unknown",
                "investor_email": investor.email if investor else "Unknown",
                "created_at": created_at.isoformat(),
                "updated_at": updated_at.isoformat(),
                "message_count": message_count,
                "developer_unread_count": developer_unread,
                "investor_unread_count": investor_unread,
                "last_message": {
                    "message": last_message.message if last_message else None,
                    "sender_id": str(last_message.sender_id) if last_message else None,
                    "created_at": last_message.created_at.isoformat() if last_message else None
                } if last_message else None
            })
        
        # Get total count for pagination
        count_query = select(func.count(ChatRoom.id))
        if project_id:
            count_query = count_query.where(ChatRoom.project_id == uuid.UUID(project_id))
        if user_id:
            count_query = count_query.where(
                or_(
                    ChatRoom.developer_id == uuid.UUID(user_id),
                    ChatRoom.investor_id == uuid.UUID(user_id)
                )
            )
        
        count_result = await db.execute(count_query)
        total = count_result.scalar()
        
        # If search is applied, use filtered results count
        if search:
            total = len(rooms_data)
        
        return {
            "rooms": rooms_data,
            "total": total,
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        import traceback
        print(f"Error fetching admin chat rooms: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/rooms/{room_id}/messages")
async def admin_get_room_messages(
    room_id: str,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """
    Admin endpoint to get all messages in a specific chat room
    """
    try:
        # Verify room exists
        room_query = select(ChatRoom).where(ChatRoom.id == uuid.UUID(room_id))
        room_result = await db.execute(room_query)
        room = room_result.scalar_one_or_none()
        
        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")
        
        # Get messages with sender information
        query = select(ChatMessage).where(
            ChatMessage.room_id == uuid.UUID(room_id)
        ).order_by(ChatMessage.created_at.asc()).offset(skip).limit(limit)
        
        result = await db.execute(query)
        messages = result.scalars().all()
        
        # Get sender details for each message
        messages_data = []
        for msg in messages:
            sender_query = select(User).where(User.id == msg.sender_id)
            sender_result = await db.execute(sender_query)
            sender = sender_result.scalar_one_or_none()
            
            # Handle optional updated_at field
            updated_at = msg.updated_at if hasattr(msg, 'updated_at') else msg.created_at
            
            messages_data.append({
                "id": str(msg.id),
                "room_id": str(msg.room_id),
                "sender_id": str(msg.sender_id),
                "sender_name": sender.name if sender else "Unknown",
                "sender_email": sender.email if sender else "Unknown",
                "sender_role": sender.role if sender else "Unknown",
                "message": msg.message,
                "file_url": msg.file_url if hasattr(msg, 'file_url') else None,
                "file_name": msg.file_name if hasattr(msg, 'file_name') else None,
                "read": msg.read if hasattr(msg, 'read') else False,
                "created_at": msg.created_at.isoformat(),
                "updated_at": updated_at.isoformat(),
                # Moderation fields
                "flagged": getattr(msg, 'flagged', False),
                "moderation_status": getattr(msg, 'moderation_status', 'approved'),
                "moderation_reason": getattr(msg, 'moderation_reason', None)
            })
        
        # Get total message count
        count_query = select(func.count(ChatMessage.id)).where(
            ChatMessage.room_id == uuid.UUID(room_id)
        )
        count_result = await db.execute(count_query)
        total = count_result.scalar()
        
        return {
            "messages": messages_data,
            "total": total,
            "skip": skip,
            "limit": limit
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid room ID format")
    except Exception as e:
        import traceback
        print(f"Error fetching admin room messages: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/statistics")
async def admin_get_chat_statistics(
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """
    Admin endpoint to get overall chat statistics
    """
    try:
        # Total chat rooms
        total_rooms_query = select(func.count(ChatRoom.id))
        total_rooms_result = await db.execute(total_rooms_query)
        total_rooms = total_rooms_result.scalar()
        
        # Total messages
        total_messages_query = select(func.count(ChatMessage.id))
        total_messages_result = await db.execute(total_messages_query)
        total_messages = total_messages_result.scalar()
        
        # Active rooms (rooms with messages in last 7 days)
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        active_rooms_query = select(func.count(func.distinct(ChatMessage.room_id))).where(
            ChatMessage.created_at >= seven_days_ago
        )
        active_rooms_result = await db.execute(active_rooms_query)
        active_rooms = active_rooms_result.scalar()
        
        # Total unread messages (if read field exists)
        try:
            unread_messages_query = select(func.count(ChatMessage.id)).where(
                ChatMessage.read == False
            )
            unread_messages_result = await db.execute(unread_messages_query)
            unread_messages = unread_messages_result.scalar()
        except AttributeError:
            unread_messages = 0
        
        # Messages sent today
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_messages_query = select(func.count(ChatMessage.id)).where(
            ChatMessage.created_at >= today_start
        )
        today_messages_result = await db.execute(today_messages_query)
        today_messages = today_messages_result.scalar()
        
        # Average messages per room
        avg_messages_per_room = total_messages / total_rooms if total_rooms > 0 else 0
        
        return {
            "total_rooms": total_rooms,
            "total_messages": total_messages,
            "active_rooms_last_7_days": active_rooms,
            "unread_messages": unread_messages,
            "messages_today": today_messages,
            "avg_messages_per_room": round(avg_messages_per_room, 2)
        }
        
    except Exception as e:
        import traceback
        print(f"Error fetching chat statistics: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/search-messages")
async def admin_search_messages(
    query: str,
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """
    Admin endpoint to search all messages across all chat rooms
    """
    try:
        search_pattern = f"%{query.lower()}%"
        
        # Search messages
        search_query = select(ChatMessage).where(
            func.lower(ChatMessage.message).like(search_pattern)
        ).order_by(ChatMessage.created_at.desc()).offset(skip).limit(limit)
        
        result = await db.execute(search_query)
        messages = result.scalars().all()
        
        # Get details for each message
        messages_data = []
        for msg in messages:
            # Get room details
            room_query = select(ChatRoom).where(ChatRoom.id == msg.room_id)
            room_result = await db.execute(room_query)
            room = room_result.scalar_one_or_none()
            
            # Get project details separately
            project = None
            if room:
                project_query = select(Project).where(Project.id == room.project_id)
                project_result = await db.execute(project_query)
                project = project_result.scalar_one_or_none()
            
            # Get sender details
            sender_query = select(User).where(User.id == msg.sender_id)
            sender_result = await db.execute(sender_query)
            sender = sender_result.scalar_one_or_none()
            
            messages_data.append({
                "id": str(msg.id),
                "room_id": str(msg.room_id),
                "project_title": project.title if project else "Unknown",
                "sender_id": str(msg.sender_id),
                "sender_name": sender.name if sender else "Unknown",
                "sender_role": sender.role if sender else "Unknown",
                "message": msg.message,
                "created_at": msg.created_at.isoformat()
            })
        
        # Get total count
        count_query = select(func.count(ChatMessage.id)).where(
            func.lower(ChatMessage.message).like(search_pattern)
        )
        count_result = await db.execute(count_query)
        total = count_result.scalar()
        
        return {
            "messages": messages_data,
            "total": total,
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        import traceback
        print(f"Error searching messages: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))