# app/api/v1/projects.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.orm import joinedload
import uuid
from datetime import datetime, timedelta

from app.database import get_db
from app.models.user import User
from app.models.project import Project
from app.schemas.project import ProjectCreate, ProjectUpdate
from app.core.dependencies import get_current_user, get_current_admin
from app.core.constants import PROJECT_POSTING_FEE
from app.core.exceptions import PaymentRequiredException, NotFoundException, ForbiddenException
from app.utils.converters import model_to_dict
from app.services.email_service import EmailService

from decimal import Decimal
import logging
import asyncio

from app.models.payment import CheckoutSession, DeveloperPayout
from app.models.notification import Notification
from app.models.chat import ChatRoom, ChatMessage
from app.core.constants import PLATFORM_FEE_PERCENTAGE, PLATFORM_FIXED_FEE



from app.services.payoneer_service import PayoneerService  # ✅ ADD THIS
from app.services.notification_service import NotificationService  # ✅ ADD THIS


router = APIRouter(prefix="/projects", tags=["Projects"])

payoneer_service = PayoneerService()



# Setup logger
logger = logging.getLogger(__name__)




@router.get("")
async def get_projects(
    skip: int = 0,
    limit: int = 20,
    status: str = None,
    db: AsyncSession = Depends(get_db)
):
    """Get projects with developer info"""
    
    query = select(Project).options(joinedload(Project.developer))
    
    if status:
        query = query.where(Project.status == status)
    
    query = query.order_by(Project.created_at.desc()).offset(skip).limit(limit)
    result = await db.execute(query)
    projects = result.scalars().all()
    
    project_list = []
    for project in projects:
        p = model_to_dict(project)
        
        if project.developer:
            p["developer"] = {
                "id": str(project.developer.id),
                "name": project.developer.name,
                "company": project.developer.company,
                "avatar": project.developer.avatar,
            }
        
        project_list.append(p)
    
    return {
        "projects": project_list,
        "total": len(project_list),
        "skip": skip,
        "limit": limit
    }


@router.post("")
async def create_project(
    project_data: ProjectCreate,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new project"""
    
    if current_user["role"] != "developer":
        raise ForbiddenException("Only developers can create projects")
    
    user_id = uuid.UUID(current_user["id"])
    
    # Check posting credits
    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalar_one_or_none()
    
    if not user:
        raise NotFoundException("User not found")
    
    # Check if user has credits
    if (user.posting_credits or 0) <= 0:
        raise PaymentRequiredException(
            detail={
                "message": "You need to purchase posting credit before creating a project",
                "code": "NO_POSTING_CREDITS",
                "posting_fee": PROJECT_POSTING_FEE
            }
        )
    
    # Deduct 1 credit
    user.posting_credits = (user.posting_credits or 0) - 1
    
    # Parse deadline
    deadline = None
    if project_data.deadline:
        try:
            deadline = datetime.fromisoformat(project_data.deadline.replace("Z", "+00:00"))
        except:
            deadline = datetime.utcnow() + timedelta(days=30)
    
    # Create project
    new_project = Project(
        title=project_data.title,
        status=project_data.status,
        description=project_data.description,
        tech_stack=project_data.tech_stack,
        requirements=project_data.requirements,
        budget=project_data.budget,
        lowest_bid=project_data.lowest_bid,
        deadline=deadline,
        location=project_data.location or "Remote",
        developer_id=current_user["id"],
        category=project_data.category or "Machine Learning",
        images=project_data.images if hasattr(project_data, 'images') else []
    )
    
    db.add(new_project)
    await db.commit()
    await db.refresh(new_project)
    
    # Send admin notification
    background_tasks.add_task(
        EmailService.send_admin_project_notification,
        action="created",
        project_id=str(new_project.id),
        project_title=new_project.title,
        developer_name=current_user.get('name', 'Unknown'),
        developer_email=current_user.get('email', 'Unknown'),
        project_data={
            'category': new_project.category,
            'budget': float(new_project.budget),
            'status': new_project.status,
            'location': new_project.location,
            'tech_stack': new_project.tech_stack
        }
    )
    
    return {
        **model_to_dict(new_project),
        "remaining_credits": user.posting_credits
    }


@router.get("/check-posting-status")
async def check_posting_status(
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Check if user has available posting credits"""
    
    user_id = uuid.UUID(current_user["id"])
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise NotFoundException("User not found")
    
    if user.role != "developer":
        raise ForbiddenException("Only developers can post projects")
    
    credits = user.posting_credits or 0
    has_credit = credits > 0
    
    return {
        "success": True,
        "has_credit": has_credit,
        "credits": credits,
        "needs_payment": not has_credit,
        "posting_fee": PROJECT_POSTING_FEE
    }


@router.get("/{project_id}")
async def get_project(
    project_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific project"""
    
    project = await db.scalar(
        select(Project)
        .options(joinedload(Project.developer))
        .where(Project.id == uuid.UUID(project_id))
    )
    
    if not project:
        raise NotFoundException("Project not found")
    
    data = model_to_dict(project)
    data["developer"] = {
        "id": str(project.developer.id),
        "name": project.developer.name,
        "company": project.developer.company,
        "avatar": project.developer.avatar,
    }
    return data


@router.put("/{project_id}")
async def update_project(
    project_id: str,
    project_data: ProjectUpdate,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update a project"""
    
    project = await db.scalar(select(Project).where(Project.id == uuid.UUID(project_id)))
    if not project:
        raise NotFoundException("Project not found")
    
    # Check permissions
    is_owner = str(project.developer_id) == str(current_user["id"])
    is_admin = current_user.get("role") == "admin"
    
    if not is_owner and not is_admin:
        raise ForbiddenException("Only the project owner or admin can edit this project")
    
    # Track changes if admin is editing
    changes = {}
    admin_editing_others_project = is_admin and not is_owner
    
    if admin_editing_others_project:
        for field, value in project_data.dict(exclude_unset=True).items():
            if field == "deadline" and value:
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            
            old_value = getattr(project, field, None)
            
            if old_value != value:
                if field == 'images':
                    old_images = old_value or []
                    new_images = value or []
                    if len(old_images) != len(new_images):
                        changes['images'] = {
                            'old': old_images,
                            'new': new_images,
                            'reason': 'Image(s) removed for policy compliance'
                        }
                elif field == 'tech_stack':
                    changes['tech_stack'] = {'old': old_value or [], 'new': value or []}
                elif field == 'budget':
                    changes['budget'] = {'old': float(old_value) if old_value else 0, 'new': float(value) if value else 0}
                elif field == 'description':
                    changes['description'] = {'old': old_value or '', 'new': value or '', 'reason': 'Content updated for policy compliance'}
                elif field == 'title':
                    changes['title'] = {'old': old_value or '', 'new': value or ''}
                elif field == 'status':
                    changes['status'] = {'old': old_value or '', 'new': value or ''}
                elif field == 'category':
                    changes['category'] = {'old': old_value or '', 'new': value or ''}
                elif field == 'location':
                    changes['location'] = {'old': old_value or '', 'new': value or ''}
    
    # Apply updates
    for field, value in project_data.dict(exclude_unset=True).items():
        if field == "deadline" and value:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        setattr(project, field, value)
    
    await db.commit()
    await db.refresh(project)
    
    # Send admin notification
    background_tasks.add_task(
        send_admin_project_notification,
        action="updated",
        project_id=str(project.id),
        project_title=project.title,
        developer_name=current_user.get('name', 'Unknown'),
        developer_email=current_user.get('email', 'Unknown'),
        project_data={
            'category': project.category,
            'budget': float(project.budget),
            'status': project.status,
            'location': project.location,
            'tech_stack': project.tech_stack
        }
    )
    
    # Send email to developer if admin edited
    if admin_editing_others_project and changes:
        developer = await db.scalar(select(User).where(User.id == project.developer_id))
        
        if developer and developer.email:
            background_tasks.add_task(
                EmailService.send_developer_edit_notification,
                developer_email=developer.email,
                developer_name=developer.name,
                project_title=project.title,
                project_id=str(project.id),
                admin_name=current_user.get('name', 'Admin'),
                changes=changes
            )
    
    return model_to_dict(project)


@router.delete("/{project_id}")
async def delete_project(
    project_id: str,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a project"""
    
    project = await db.scalar(select(Project).where(Project.id == uuid.UUID(project_id)))
    
    if not project:
        raise NotFoundException("Project not found")
    
    if str(project.developer_id) != str(current_user["id"]) and current_user["role"] != "admin":
        raise ForbiddenException("Unauthorized")
    
    # Capture data before delete
    project_title = project.title
    project_category = project.category
    project_budget = project.budget
    
    await db.delete(project)
    await db.commit()
    
    # Send admin notification
    background_tasks.add_task(
        send_admin_project_notification,
        action="deleted",
        project_id=project_id,
        project_title=project_title,
        developer_name=current_user.get('name', 'Unknown'),
        developer_email=current_user.get('email', 'Unknown'),
        project_data={
            'category': project_category,
            'budget': project_budget,
            'status': 'deleted',
            'location': 'N/A',
            'tech_stack': []
        }
    )
    
    return {"message": "Project deleted", "project_id": project_id}


@router.get("/developer/{developer_id}")
async def get_developer_projects(
    developer_id: str,
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """Get all projects for a developer"""
    
    result = await db.execute(
        select(Project)
        .where(Project.developer_id == uuid.UUID(developer_id))
        .order_by(Project.created_at.desc())
        .offset(skip).limit(limit)
    )
    projects = result.scalars().all()
    
    # Check for chat rooms for each project
    projects_with_chat_info = []
    for project in projects:
        project_dict = model_to_dict(project)
        
        # Check if project has any chat rooms
        from app.models.chat import ChatRoom
        chat_room_result = await db.execute(
            select(ChatRoom).where(ChatRoom.project_id == project.id).limit(1)
        )
        has_chat_room = chat_room_result.scalar() is not None
        project_dict["has_chat_room"] = has_chat_room
        
        projects_with_chat_info.append(project_dict)
    
    return {
        "projects": projects_with_chat_info,
        "total": len(projects),
        "skip": skip,
        "limit": limit
    }



@router.post("/projects/{project_id}/simple-approve")
async def simple_approve_project(
    project_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Simplified project approval - investor confirms completion
    ✅ UPDATED: Uses Payoneer for payouts instead of Stripe Connect
    """
    try:
        # Get project
        project_query = await db.execute(
            select(Project).where(Project.id == uuid.UUID(project_id))
        )
        project = project_query.scalar_one_or_none()
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Verify project is active
        if project.status not in ['fixed_price', 'in_progress']:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot confirm project with status '{project.status}'. Project must be in progress."
            )
        
        # Calculate payout
        project_amount = float(project.budget)
        platform_fee_percentage = 6.0  # 6%
        platform_fee = project_amount * (platform_fee_percentage / 100)
        developer_payout = project_amount - platform_fee
        
        logger.info(f"Approving project {project_id}: amount=${project_amount}, fee=${platform_fee}, payout=${developer_payout}")
        
        # Store original status
        original_status = project.status
        
        # Update project status
        if original_status == 'in_progress':
            project.status = 'completed'
            logger.info(f"Auction project {project_id} marked as completed")
        else:
            logger.info(f"Fixed-price project {project_id} keeps status '{project.status}'")
        
        project.completed_at = datetime.utcnow()
        
        # Get developer user
        developer_query = await db.execute(
            select(User).where(User.id == project.developer_id)
        )
        developer = developer_query.scalar_one()
        
        # Update developer stats
        developer.projects_completed = (developer.projects_completed or 0) + 1
        developer.total_earnings = (developer.total_earnings or 0) + Decimal(str(developer_payout))

        # Find the checkout session
        user_uuid = uuid.UUID(current_user['id'])
        checkout_result = await db.execute(
            select(CheckoutSession).where(
                and_(
                    CheckoutSession.project_id == project.id,
                    CheckoutSession.customer_id == user_uuid,
                    CheckoutSession.status == 'completed'
                )
            ).order_by(CheckoutSession.completed_at.desc())
        )
        checkout_session = checkout_result.scalar_one_or_none()

        # ✅ CHECK: Developer has Payoneer set up
        if not developer.payoneer_payee_id or not developer.payoneer_onboarding_completed:
            # Create payout record as 'pending' but notify developer to set up Payoneer
            payout_record = DeveloperPayout(
                developer_id=project.developer_id,
                project_id=project.id,
                checkout_session_id=checkout_session.id if checkout_session else None,
                investor_id=user_uuid,
                gross_amount=Decimal(str(project_amount)),
                platform_fee=Decimal(str(platform_fee)),
                net_amount=Decimal(str(developer_payout)),
                currency='USD',
                status='pending',
                description=f"Payment for project: {project.title}",
                failure_reason="Payoneer account not set up"
            )
            db.add(payout_record)
            await db.commit()
            
            # Notify developer to set up Payoneer
            notification = Notification(
                user_id=project.developer_id,
                type='payoneer_setup_required',
                title='⚠️ Payoneer Setup Required',
                message=f'Your payout of ${developer_payout:.2f} is ready, but you need to connect your Payoneer account to receive it.',
                link='/payout-settings',
                details={
                    'project_id': str(project.id),
                    'project_title': project.title,
                    'amount': float(developer_payout),
                    'action_required': 'payoneer_setup'
                },
                read=False
            )
            db.add(notification)
            await db.commit()
            
            return {
                "success": True,
                "message": "Project approved! Developer needs to set up Payoneer to receive payment.",
                "project_id": str(project.id),
                "project_status": project.status,
                "developer_payout": float(developer_payout),
                "platform_commission": float(platform_fee),
                "gross_amount": float(project_amount),
                "action_required": "payoneer_setup",
                "notification_sent": True
            }

        # ✅ CREATE PAYOUT RECORD (with Payoneer transfer)
        payout_record = DeveloperPayout(
            developer_id=project.developer_id,
            project_id=project.id,
            checkout_session_id=checkout_session.id if checkout_session else None,
            investor_id=user_uuid,
            gross_amount=Decimal(str(project_amount)),
            platform_fee=Decimal(str(platform_fee)),
            net_amount=Decimal(str(developer_payout)),
            currency='USD',
            status='pending',
            description=f"Payment for project: {project.title}"
        )
        db.add(payout_record)
        await db.flush()

        # ✅ INITIATE PAYONEER TRANSFER
        try:
            payoneer_result = await payoneer_service.initiate_payout(
                payee_id=developer.payoneer_payee_id,
                amount=developer_payout,
                currency='USD',
                description=f"Payment for project: {project.title}",
                reference_id=str(payout_record.id)
            )

            if payoneer_result.get("success"):
                # Update payout with Payoneer transfer details
                payout_record.payoneer_transfer_id = payoneer_result.get("transfer_id")
                payout_record.payoneer_transfer_status = payoneer_result.get("status", "pending")
                payout_record.payoneer_quote_id = payoneer_result.get("quote_id")
                payout_record.payoneer_batch_id = payoneer_result.get("batch_id")
                payout_record.status = 'processing'
                payout_record.transaction_id = payoneer_result.get("transfer_id")
                
                logger.info(f"✅ Payoneer transfer initiated: {payoneer_result.get('transfer_id')} for payout {payout_record.id}")
            else:
                # Payoneer transfer failed - mark as failed
                payout_record.status = 'failed'
                payout_record.failure_reason = payoneer_result.get("error", "Payoneer transfer failed")
                logger.error(f"❌ Payoneer transfer failed: {payoneer_result.get('error')}")
                
                # Still commit the project approval but notify admin
                await db.commit()
                
                # Notify admin about failure
                await _notify_admin_payoneer_failure(
                    project_id=str(project.id),
                    project_title=project.title,
                    developer_name=developer.name,
                    developer_email=developer.email,
                    amount=developer_payout,
                    error=payoneer_result.get("error", "Unknown error")
                )
                
                return {
                    "success": True,
                    "message": "Project approved but Payoneer transfer failed. Admin has been notified.",
                    "project_id": str(project.id),
                    "project_status": project.status,
                    "developer_payout": float(developer_payout),
                    "platform_commission": float(platform_fee),
                    "gross_amount": float(project_amount),
                    "payoneer_status": "failed",
                    "failure_reason": payoneer_result.get("error", "Unknown error")
                }

        except Exception as payoneer_error:
            logger.error(f"❌ Payoneer error: {payoneer_error}")
            payout_record.status = 'failed'
            payout_record.failure_reason = str(payoneer_error)
            await db.commit()
            
            # Notify admin
            await _notify_admin_payoneer_failure(
                project_id=str(project.id),
                project_title=project.title,
                developer_name=developer.name,
                developer_email=developer.email,
                amount=developer_payout,
                error=str(payoneer_error)
            )
            
            return {
                "success": True,
                "message": "Project approved but Payoneer transfer failed. Admin has been notified.",
                "project_id": str(project.id),
                "project_status": project.status,
                "developer_payout": float(developer_payout),
                "platform_commission": float(platform_fee),
                "gross_amount": float(project_amount),
                "payoneer_status": "failed",
                "failure_reason": str(payoneer_error)
            }

        # ✅ COMMIT ALL CHANGES
        await db.commit()
        await db.refresh(project)
        await db.refresh(payout_record)

        # ✅ CREATE NOTIFICATION FOR DEVELOPER
        dev_notification = Notification(
            user_id=project.developer_id,
            type='project_approved',
            title='🎉 Project Approved!',
            message=f'Your project "{project.title}" has been approved by the investor. Payment of ${developer_payout:.2f} is being processed via Payoneer.',
            link=f"/projects/{project_id}",
            details={
                'project_id': str(project.id),
                'project_title': project.title,
                'amount': float(developer_payout),
                'platform_fee': float(platform_fee),
                'gross_amount': float(project_amount),
                'payoneer_transfer_id': payout_record.payoneer_transfer_id
            },
            read=False
        )
        db.add(dev_notification)

        # ✅ SEND SYSTEM MESSAGE TO CHAT
        try:
            room_query = await db.execute(
                select(ChatRoom).where(
                    and_(
                        ChatRoom.project_id == project.id,
                        ChatRoom.investor_id == user_uuid
                    )
                )
            )
            room = room_query.scalar_one_or_none()
            
            if room:
                system_message = ChatMessage(
                    room_id=room.id,
                    sender_id=project.developer_id,
                    message=f"✅ Project Completed!\n\n" +
                           f"Investor has confirmed project completion.\n" +
                           f"Payment Details:\n" +
                           f"• Gross Amount: ${project_amount:.2f}\n" +
                           f"• Platform Fee (6%): ${platform_fee:.2f}\n" +
                           f"• Developer Payout: ${developer_payout:.2f}\n\n" +
                           f"💳 Payment is being processed via Payoneer.\n" +
                           f"Transfer ID: {payout_record.payoneer_transfer_id or 'Pending'}",
                    message_type='system',
                    created_at=datetime.utcnow()
                )
                db.add(system_message)
                logger.info(f"✅ System message added to chat room {room.id}")
        except Exception as e:
            logger.error(f"Error adding system message to chat: {e}")

        await db.commit()

        # ✅ SEND LIVE NOTIFICATION
        try:
            await NotificationService.send_notification_to_user(
                str(project.developer_id),
                {
                    "type": "project_approved",
                    "title": "🎉 Project Approved!",
                    "message": f"Your project \"{project.title}\" has been approved! Payment of ${developer_payout:.2f} is being processed via Payoneer.",
                    "link": f"/projects/{project_id}",
                    "details": {
                        'project_id': str(project.id),
                        'project_title': project.title,
                        'amount': float(developer_payout),
                        'platform_fee': float(platform_fee),
                        'gross_amount': float(project_amount),
                        'payoneer_transfer_id': payout_record.payoneer_transfer_id
                    }
                },
                db
            )
            logger.info(f"✅ Live notification sent to developer {project.developer_id}")
        except Exception as e:
            logger.error(f"Failed to send live notification: {e}")

        # ✅ SEND EMAIL
        try:
            asyncio.create_task(
                send_project_approved_email(
                    to_email=developer.email,
                    developer_name=developer.name,
                    project_title=project.title,
                    amount=developer_payout,
                    platform_fee=platform_fee,
                    gross_amount=project_amount,
                    payoneer_transfer_id=payout_record.payoneer_transfer_id
                )
            )
            logger.info(f"✅ Project approval email queued for {developer.email}")
        except Exception as e:
            logger.error(f"Failed to queue approval email: {e}")

        logger.info(f"✅ Project {project_id} approved successfully by {current_user['name']}")
        
        return {
            "success": True,
            "message": "Project approved successfully! Payment is being processed via Payoneer.",
            "project_id": str(project.id),
            "project_status": project.status,
            "developer_payout": float(developer_payout),
            "platform_commission": float(platform_fee),
            "gross_amount": float(project_amount),
            "payoneer_transfer_id": payout_record.payoneer_transfer_id,
            "payoneer_status": payout_record.payoneer_transfer_status,
            "notification_sent": True,
            "email_sent": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error approving project: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to approve project: {str(e)}"
        )


# ============================================
# HELPER FUNCTIONS
# ============================================

async def _notify_admin_payoneer_failure(
    project_id: str,
    project_title: str,
    developer_name: str,
    developer_email: str,
    amount: float,
    error: str
):
    """Notify admin about Payoneer transfer failure"""
    try:
        admin_email = settings.ADMIN_EMAIL or "bitebids@gmail.com"
        
        subject = f"⚠️ Payoneer Transfer Failed - {project_title}"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #ef4444; color: white; padding: 30px; border-radius: 10px 10px 0 0; text-align: center; }}
                .content {{ background: #f8fafc; padding: 30px; border: 1px solid #e0e0e0; }}
                .detail {{ padding: 10px 0; border-bottom: 1px solid #e5e7eb; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>⚠️ Payoneer Transfer Failed</h1>
            </div>
            <div class="content">
                <h2>Project: {project_title}</h2>
                <div class="detail"><strong>Project ID:</strong> {project_id}</div>
                <div class="detail"><strong>Developer:</strong> {developer_name} ({developer_email})</div>
                <div class="detail"><strong>Amount:</strong> ${amount:.2f}</div>
                <div class="detail"><strong>Error:</strong> {error}</div>
                <p style="margin-top: 20px;">
                    Please check the Payoneer dashboard and retry the payout manually if needed.
                </p>
                <a href="{settings.FRONTEND_URL}/admin/payouts" 
                   style="display:inline-block; background:#ef4444; color:white; padding:12px 24px; border-radius:6px; text-decoration:none; margin-top:15px;">
                    View Payouts
                </a>
            </div>
        </body>
        </html>
        """
        
        await EmailService.send_email(
            to_email=admin_email,
            subject=subject,
            html_content=html_content
        )
        logger.info(f"📧 Admin notification sent for Payoneer failure on {project_title}")
        
    except Exception as e:
        logger.error(f"Failed to notify admin about Payoneer failure: {e}")
