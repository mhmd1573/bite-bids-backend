# app/api/v1/disputes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, or_
import uuid
from datetime import datetime

from app.database import get_db
from app.models.project import Project
from app.models.user import User
from app.models.chat import ChatRoom
from app.models.dispute import ProjectDisputeSimple
from app.models.notification import Notification
from app.models.payment import CheckoutSession, DeveloperPayout
from app.core.dependencies import get_current_user, get_current_admin
from app.core.exceptions import NotFoundException, ForbiddenException
from app.utils.converters import model_to_dict
from app.services.notification_service import NotificationService

router = APIRouter(prefix="/disputes", tags=["Disputes"])


@router.post("/projects/{project_id}/create")
async def create_project_dispute(
    project_id: str,
    data: dict,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a dispute for a project"""
    try:
        # Get project
        project_query = await db.execute(
            select(Project).where(Project.id == uuid.UUID(project_id))
        )
        project = project_query.scalar_one_or_none()
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Verify project is active
        if project.status not in ['fixed_price', 'in_progress', 'disputed']:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot dispute project with status '{project.status}'"
            )
        
        # Determine who's opening the dispute
        user_uuid = uuid.UUID(current_user['id'])
        is_developer = user_uuid == project.developer_id
        
        if is_developer:
            # Developer opening dispute - need to know which investor
            investor_id_str = data.get('investor_id')
            
            if not investor_id_str:
                # Check how many chat rooms exist
                rooms_query = await db.execute(
                    select(ChatRoom).where(
                        and_(
                            ChatRoom.project_id == project.id,
                            ChatRoom.developer_id == user_uuid
                        )
                    )
                )
                rooms = rooms_query.scalars().all()
                
                if len(rooms) > 1:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "code": "INVESTOR_ID_REQUIRED",
                            "message": "Multiple investors have purchased this project. Please specify which investor.",
                            "investors": [
                                {
                                    "investor_id": str(room.investor_id),
                                    "room_id": str(room.id)
                                }
                                for room in rooms
                            ]
                        }
                    )
                elif len(rooms) == 1:
                    investor_id = rooms[0].investor_id
                    room = rooms[0]
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="No chat room found. Cannot open dispute."
                    )
            else:
                try:
                    investor_id = uuid.UUID(investor_id_str)
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid investor_id format")
                
                room_query = await db.execute(
                    select(ChatRoom).where(
                        and_(
                            ChatRoom.project_id == project.id,
                            ChatRoom.developer_id == user_uuid,
                            ChatRoom.investor_id == investor_id
                        )
                    )
                )
                room = room_query.scalar_one_or_none()
                
                if not room:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No chat room found with investor {investor_id_str}"
                    )
        else:
            # Investor opening dispute
            investor_id = user_uuid
            
            room_query = await db.execute(
                select(ChatRoom).where(
                    and_(
                        ChatRoom.project_id == project.id,
                        ChatRoom.investor_id == investor_id
                    )
                )
            )
            room = room_query.scalar_one_or_none()
            
            if not room:
                raise HTTPException(
                    status_code=400,
                    detail="No chat room found. You must have an active chat to open a dispute."
                )
        
        # Check for existing active dispute
        existing_dispute_query = await db.execute(
            select(ProjectDisputeSimple).where(
                and_(
                    ProjectDisputeSimple.project_id == uuid.UUID(project_id),
                    ProjectDisputeSimple.investor_id == investor_id,
                    ProjectDisputeSimple.resolved == False
                )
            )
        )
        active_dispute = existing_dispute_query.scalar_one_or_none()
        
        if active_dispute:
            raise HTTPException(
                status_code=400,
                detail="An active dispute already exists with this investor."
            )
        
        reason = data.get('reason', '').strip()
        notes = data.get('notes', '').strip()
        
        if not reason:
            raise HTTPException(status_code=400, detail="Dispute reason is required")
        
        # Store old status
        old_status = project.status
        
        # Only change status for auction projects
        if project.status == 'in_progress':
            project.status = 'disputed'
        
        # Create dispute record
        dispute_record = ProjectDisputeSimple(
            project_id=uuid.UUID(project_id),
            reason=reason,
            notes=notes,
            disputed_by=user_uuid,
            investor_id=investor_id,
            disputed_at=datetime.utcnow(),
            previous_status=old_status,
            resolved=False
        )
        db.add(dispute_record)
        
        # Get developer and investor details
        developer_query = await db.execute(
            select(User).where(User.id == project.developer_id)
        )
        developer = developer_query.scalar_one()
        
        investor_query = await db.execute(
            select(User).where(User.id == investor_id)
        )
        investor = investor_query.scalar_one()
        
        # Notify the other party
        other_party_id = investor_id if is_developer else project.developer_id
        dispute_opener = "developer" if is_developer else "investor"
        
        other_party_notification = Notification(
            user_id=other_party_id,
            type='dispute_opened',
            title='⚠️ Dispute Opened',
            message=f'The {dispute_opener} has opened a dispute for project "{project.title}". Reason: {reason}.',
            link=f"/projects/{project_id}",
            details={
                'project_id': str(project.id),
                'dispute_id': str(dispute_record.id),
                'reason': reason,
                'opened_by': dispute_opener
            },
            read=False
        )
        db.add(other_party_notification)
        
        # Add system message to chat
        notes_part = f"Notes: {notes}\n\n" if notes else ""
        system_message_text = (
            f"⚠️ Dispute Opened\n\n"
            f"The {dispute_opener} has opened a dispute.\n\n"
            f"Reason: {reason}\n"
            f"{notes_part}"
            f"An admin will review this case and make a fair decision."
        )
        
        from app.models.chat import ChatMessage
        system_message = ChatMessage(
            room_id=room.id,
            sender_id=project.developer_id,
            message=system_message_text,
            message_type='system',
            created_at=datetime.utcnow()
        )
        db.add(system_message)
        
        await db.commit()
        
        return {
            "success": True,
            "message": "Dispute created successfully. An admin will review your case.",
            "dispute_id": str(dispute_record.id),
            "project_id": str(project.id),
            "investor_id": str(investor_id),
            "developer_id": str(project.developer_id)
        }
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create dispute: {str(e)}")


@router.get("/admin/disputes-simple")
async def get_simple_disputes(
    admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get all active disputes"""
    try:
        disputes_query = await db.execute(
            select(ProjectDisputeSimple)
            .where(ProjectDisputeSimple.resolved == False)
            .order_by(ProjectDisputeSimple.disputed_at.desc())
        )
        dispute_records = disputes_query.scalars().all()
        
        disputes = []
        for dispute_record in dispute_records:
            try:
                # Get project
                project_query = await db.execute(
                    select(Project).where(Project.id == dispute_record.project_id)
                )
                project = project_query.scalar_one_or_none()
                
                if not project:
                    continue
                
                # Get developer
                developer_query = await db.execute(
                    select(User).where(User.id == project.developer_id)
                )
                developer = developer_query.scalar_one_or_none()
                
                if not developer:
                    continue
                
                # Get investor
                investor_query = await db.execute(
                    select(User).where(User.id == dispute_record.investor_id)
                )
                investor = investor_query.scalar_one_or_none()
                
                if not investor:
                    continue
                
                # Get dispute opener
                dispute_opener_query = await db.execute(
                    select(User).where(User.id == dispute_record.disputed_by)
                )
                dispute_opener = dispute_opener_query.scalar_one_or_none()
                
                if not dispute_opener:
                    continue
                
                is_developer_dispute = dispute_record.disputed_by == project.developer_id
                opened_by_role = "developer" if is_developer_dispute else "investor"
                
                disputes.append({
                    "id": str(project.id),
                    "dispute_id": str(dispute_record.id),
                    "project": {
                        "id": str(project.id),
                        "title": project.title,
                        "amount": float(project.budget),
                        "status": project.status,
                        "previous_status": dispute_record.previous_status
                    },
                    "developer": {
                        "id": str(developer.id),
                        "name": developer.name,
                        "email": developer.email
                    },
                    "investor": {
                        "id": str(investor.id),
                        "name": investor.name,
                        "email": investor.email
                    },
                    "dispute_reason": dispute_record.reason or 'Unknown',
                    "dispute_notes": dispute_record.notes or '',
                    "disputed_at": dispute_record.disputed_at.isoformat() if dispute_record.disputed_at else None,
                    "opened_by": opened_by_role,
                    "opened_by_name": dispute_opener.name,
                    "platform_commission": float(project.budget) * 0.06,
                    "developer_payout": float(project.budget) - (float(project.budget) * 0.06),
                })
            except:
                continue
        
        return {"disputes": disputes}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch disputes: {str(e)}")


@router.post("/admin/disputes-simple/{project_id}/resolve")
async def resolve_simple_dispute(
    project_id: str,
    data: dict,
    admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Resolve a dispute"""
    try:
        dispute_id = data.get('dispute_id')
        
        # Get project
        project_query = await db.execute(
            select(Project).where(Project.id == uuid.UUID(project_id))
        )
        project = project_query.scalar_one_or_none()
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Get active dispute
        if dispute_id:
            dispute_query = await db.execute(
                select(ProjectDisputeSimple).where(ProjectDisputeSimple.id == uuid.UUID(dispute_id))
            )
        else:
            dispute_query = await db.execute(
                select(ProjectDisputeSimple)
                .where(
                    ProjectDisputeSimple.project_id == uuid.UUID(project_id),
                    ProjectDisputeSimple.resolved == False
                )
            )
        
        dispute = dispute_query.scalar_one_or_none()
        
        if not dispute:
            raise HTTPException(status_code=404, detail="Active dispute not found")
        
        resolution = data.get('resolution')
        admin_notes = data.get('admin_notes', '').strip()
        
        if not resolution or resolution not in ['refund_investor', 'refund_developer', 'continue_project']:
            raise HTTPException(
                status_code=400,
                detail="Invalid resolution. Must be 'refund_investor', 'refund_developer', or 'continue_project'"
            )
        
        # Mark dispute as resolved
        dispute.resolved = True
        dispute.resolution = resolution
        dispute.admin_notes = admin_notes
        dispute.resolved_by = uuid.UUID(admin['id'])
        dispute.resolved_at = datetime.utcnow()
        
        # Handle resolution
        is_fixed_price = dispute.previous_status == 'fixed_price'
        
        if is_fixed_price:
            # Check for other active disputes
            other_disputes_query = await db.execute(
                select(ProjectDisputeSimple)
                .where(
                    ProjectDisputeSimple.project_id == uuid.UUID(project_id),
                    ProjectDisputeSimple.resolved == False,
                    ProjectDisputeSimple.id != dispute.id
                )
            )
            other_active_disputes = other_disputes_query.scalars().all()
            
            if not other_active_disputes:
                project.status = 'fixed_price'
            
            resolution_message = "Admin has resolved this dispute. "
            if resolution == 'refund_investor':
                resolution_message += "Full refund issued to investor."
            elif resolution == 'refund_developer':
                resolution_message += "Full payment released to developer."
            else:
                resolution_message += "Both parties can continue working together."
        else:
            if resolution == 'refund_investor':
                project.status = 'cancelled'
                resolution_message = "Admin ruled in favor of the investor. Full refund issued."
            elif resolution == 'refund_developer':
                project.status = 'completed'
                resolution_message = "Admin ruled in favor of the developer. Full payment released."
            else:
                project.status = dispute.previous_status or 'in_progress'
                resolution_message = "Admin decided to give both parties another chance. Project continues."
        
        # Get users
        developer_query = await db.execute(
            select(User).where(User.id == project.developer_id)
        )
        developer = developer_query.scalar_one()
        
        investor_query = await db.execute(
            select(User).where(User.id == dispute.investor_id)
        )
        investor = investor_query.scalar_one()
        
        # Create payout if resolution is refund_developer
        if resolution == 'refund_developer':
            project_amount = float(project.budget)
            platform_fee_percentage = 6.0
            platform_fee = project_amount * (platform_fee_percentage / 100)
            developer_payout_amount = project_amount - platform_fee
            
            checkout_result = await db.execute(
                select(CheckoutSession).where(
                    and_(
                        CheckoutSession.project_id == project.id,
                        CheckoutSession.customer_id == dispute.investor_id,
                        CheckoutSession.status == 'completed'
                    )
                ).order_by(CheckoutSession.completed_at.desc())
            )
            checkout_session = checkout_result.scalar_one_or_none()
            
            payout_record = DeveloperPayout(
                developer_id=project.developer_id,
                project_id=project.id,
                checkout_session_id=checkout_session.id if checkout_session else None,
                investor_id=dispute.investor_id,
                gross_amount=Decimal(str(project_amount)),
                platform_fee=Decimal(str(platform_fee)),
                net_amount=Decimal(str(developer_payout_amount)),
                currency='USD',
                status='pending',
                description=f"Dispute resolution - Payment for project: {project.title}"
            )
            db.add(payout_record)
            
            developer.total_earnings = (developer.total_earnings or 0) + Decimal(str(developer_payout_amount))
            developer.projects_completed = (developer.projects_completed or 0) + 1
        
        # Create notifications
        for user in [developer, investor]:
            notification = Notification(
                user_id=user.id,
                type='dispute_resolved',
                title='✅ Dispute Resolved',
                message=f'The dispute for project "{project.title}" has been resolved. {resolution_message}',
                link=f"/projects/{project_id}",
                details={
                    'project_id': str(project.id),
                    'project_title': project.title,
                    'resolution': resolution,
                    'admin_notes': admin_notes
                },
                read=False
            )
            db.add(notification)
        
        # Add system message to chat
        room_query = await db.execute(
            select(ChatRoom).where(
                and_(
                    ChatRoom.project_id == project.id,
                    ChatRoom.investor_id == dispute.investor_id
                )
            )
        )
        room = room_query.scalar_one_or_none()
        
        if room:
            from app.models.chat import ChatMessage
            admin_notes_part = f"\n\nAdmin Notes: {admin_notes}" if admin_notes else ""
            system_message_text = (
                f"✅ Dispute Resolved\n\n"
                f"{resolution_message}"
                f"{admin_notes_part}"
            )
            
            system_message = ChatMessage(
                room_id=room.id,
                sender_id=project.developer_id,
                message=system_message_text,
                message_type='system',
                created_at=datetime.utcnow()
            )
            db.add(system_message)
        
        await db.commit()
        
        return {
            "success": True,
            "message": "Dispute resolved successfully",
            "resolution": resolution,
            "new_status": project.status
        }
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to resolve dispute: {str(e)}")


@router.get("/chat/rooms/{room_id}/has-active-dispute")
async def check_active_dispute(
    room_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Check if there's an active dispute for this chat room"""
    try:
        room_query = await db.execute(
            select(ChatRoom).where(ChatRoom.id == uuid.UUID(room_id))
        )
        room = room_query.scalar_one_or_none()
        
        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")
        
        user_uuid = uuid.UUID(current_user['id'])
        if user_uuid not in [room.developer_id, room.investor_id]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        dispute_query = await db.execute(
            select(ProjectDisputeSimple).where(
                and_(
                    ProjectDisputeSimple.project_id == room.project_id,
                    ProjectDisputeSimple.investor_id == room.investor_id,
                    ProjectDisputeSimple.resolved == False
                )
            )
        )
        active_dispute = dispute_query.scalar_one_or_none()
        
        if active_dispute:
            return {
                "has_active_dispute": True,
                "dispute_id": str(active_dispute.id),
                "disputed_by": str(active_dispute.disputed_by),
                "reason": active_dispute.reason,
                "disputed_at": active_dispute.disputed_at.isoformat()
            }
        
        return {"has_active_dispute": False, "dispute_id": None}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))