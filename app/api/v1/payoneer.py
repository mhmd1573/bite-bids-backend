# app/api/v1/payoneer.py
from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
import uuid
from datetime import datetime

from app.database import get_db
from app.models.user import User
from app.models.payment import DeveloperPayout
from app.models.project import Project
from app.models.notification import Notification
from app.services.payoneer_service import PayoneerService
from app.core.dependencies import get_current_user, get_current_admin
from app.core.exceptions import NotFoundException, ForbiddenException
from app.services.notification_service import NotificationService
from app.utils.converters import model_to_dict
from app.config import settings

router = APIRouter(prefix="/payoneer", tags=["Payoneer Payouts"])

payoneer_service = PayoneerService()


@router.get("/account-status")
async def get_payoneer_status(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current user's Payoneer account status"""
    user_result = await db.execute(
        select(User).where(User.id == uuid.UUID(current_user['id']))
    )
    user = user_result.scalar_one_or_none()
    
    if not user:
        raise NotFoundException("User not found")
    
    # If payee exists, fetch latest status from Payoneer
    if user.payoneer_payee_id:
        try:
            status_result = await payoneer_service.get_payee_status(user.payoneer_payee_id)
            if status_result.get("success"):
                user.payoneer_payee_status = status_result.get("status")
                user.payoneer_onboarding_completed = status_result.get("is_active", False)
                await db.commit()
        except Exception as e:
            pass
    
    return {
        "payoneer_payee_id": user.payoneer_payee_id,
        "payoneer_payee_status": user.payoneer_payee_status,
        "payoneer_onboarding_completed": user.payoneer_onboarding_completed or False,
        "payoneer_verified": user.payoneer_verified or False,
        "payoneer_currency": user.payoneer_currency or "USD",
        "total_earnings": float(user.total_earnings or 0)
    }


@router.post("/create-payee")
async def create_payee_account(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a Payoneer payee account and get signup link"""
    user_result = await db.execute(
        select(User).where(User.id == uuid.UUID(current_user['id']))
    )
    user = user_result.scalar_one_or_none()
    
    if not user:
        raise NotFoundException("User not found")
    
    # Check if user already has a payee account
    if user.payoneer_payee_id:
        # If onboarding not complete, generate new link
        if not user.payoneer_onboarding_completed:
            return await _generate_onboarding_link(user, db)
        else:
            return {
                "message": "Payoneer account already set up",
                "payee_id": user.payoneer_payee_id,
                "onboarding_completed": True
            }
    
    # Create unique payee ID
    payee_id = f"BITE_{user.id.hex[:16]}_{datetime.utcnow().strftime('%Y%m%d')}"
    
    # Generate signup link
    result = await payoneer_service.create_payee_signup_link(
        payee_id=payee_id,
        email=user.email,
        name=user.name,
        redirect_url=f"{settings.FRONTEND_URL}/payout-settings?payoneer_success=true"
    )
    
    if result.get("success"):
        user.payoneer_payee_id = payee_id
        user.payoneer_payee_status = "onboarding"
        user.payoneer_onboarding_url = result.get("link")
        await db.commit()
        await db.refresh(user)
        
        return {
            "success": True,
            "onboarding_url": result.get("link"),
            "payee_id": payee_id,
            "message": "Payoneer account created. Complete your KYC using the link."
        }
    else:
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to create Payoneer account"))


@router.post("/check-onboarding-status")
async def check_onboarding_status(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Check if the user's Payoneer onboarding is complete"""
    user_result = await db.execute(
        select(User).where(User.id == uuid.UUID(current_user['id']))
    )
    user = user_result.scalar_one_or_none()
    
    if not user or not user.payoneer_payee_id:
        return {"onboarding_completed": False, "status": "not_started"}
    
    try:
        status_result = await payoneer_service.get_payee_status(user.payoneer_payee_id)
        
        if status_result.get("success"):
            is_active = status_result.get("is_active", False)
            status = status_result.get("status", "unknown")
            
            user.payoneer_payee_status = status
            user.payoneer_onboarding_completed = is_active
            
            if is_active:
                user.payoneer_verified = True
            
            await db.commit()
            
            return {
                "onboarding_completed": is_active,
                "status": status,
                "verified": user.payoneer_verified
            }
        else:
            return {"onboarding_completed": False, "status": "error", "error": status_result.get("error")}
            
    except Exception as e:
        return {"onboarding_completed": False, "status": "error", "error": str(e)}


@router.post("/admin/initiate-payout/{payout_id}")
async def admin_initiate_payout(
    payout_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    admin: dict = Depends(get_current_admin)
):
    """Admin initiates a Payoneer payout"""
    try:
        payout_uuid = uuid.UUID(payout_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid payout ID")
    
    # Get payout with developer info
    payout_result = await db.execute(
        select(DeveloperPayout)
        .where(DeveloperPayout.id == payout_uuid)
    )
    payout = payout_result.scalar_one_or_none()
    
    if not payout:
        raise NotFoundException("Payout not found")
    
    if payout.status != 'pending':
        raise HTTPException(
            status_code=400,
            detail=f"Cannot process payout with status '{payout.status}'. Must be 'pending'."
        )
    
    # Get developer
    dev_result = await db.execute(
        select(User).where(User.id == payout.developer_id)
    )
    developer = dev_result.scalar_one_or_none()
    
    if not developer:
        raise NotFoundException("Developer not found")
    
    if not developer.payoneer_payee_id or not developer.payoneer_onboarding_completed:
        raise HTTPException(
            status_code=400,
            detail="Developer has not completed Payoneer onboarding"
        )
    
    # Initiate Payoneer transfer
    result = await payoneer_service.initiate_payout(
        payee_id=developer.payoneer_payee_id,
        amount=float(payout.net_amount),
        currency=payout.currency or "USD",
        description=f"Payment for project {payout.project_id}",
        reference_id=str(payout.id)
    )
    
    if result.get("success"):
        # Update payout record
        payout.payoneer_transfer_id = result.get("transfer_id")
        payout.payoneer_transfer_status = result.get("status", "pending")
        payout.payoneer_quote_id = result.get("quote_id")
        payout.payoneer_batch_id = result.get("batch_id")
        payout.status = 'processing'
        payout.processed_by = uuid.UUID(admin['id'])
        payout.processed_at = datetime.utcnow()
        payout.transaction_id = result.get("transfer_id")
        
        await db.commit()
        
        # Notify developer
        background_tasks.add_task(
            _notify_developer_payout,
            developer.id,
            payout.net_amount,
            result.get("transfer_id"),
            db
        )
        
        return {
            "success": True,
            "message": "Payoneer payout initiated successfully",
            "transfer_id": result.get("transfer_id"),
            "status": result.get("status")
        }
    else:
        # Mark as failed
        payout.status = 'failed'
        payout.failure_reason = result.get("error", "Payoneer transfer failed")
        await db.commit()
        
        raise HTTPException(
            status_code=400,
            detail=f"Payoneer transfer failed: {result.get('error')}"
        )


@router.post("/admin/retry-payout/{payout_id}")
async def admin_retry_payout(
    payout_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    admin: dict = Depends(get_current_admin)
):
    """Admin retries a failed Payoneer payout"""
    try:
        payout_uuid = uuid.UUID(payout_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid payout ID")
    
    payout_result = await db.execute(
        select(DeveloperPayout)
        .where(DeveloperPayout.id == payout_uuid)
    )
    payout = payout_result.scalar_one_or_none()
    
    if not payout:
        raise NotFoundException("Payout not found")
    
    if payout.status != 'failed':
        raise HTTPException(
            status_code=400,
            detail=f"Can only retry failed payouts. Current status: '{payout.status}'"
        )
    
    # Get developer
    dev_result = await db.execute(
        select(User).where(User.id == payout.developer_id)
    )
    developer = dev_result.scalar_one_or_none()
    
    if not developer or not developer.payoneer_payee_id:
        raise HTTPException(
            status_code=400,
            detail="Developer Payoneer account not configured"
        )
    
    # Increment retry count
    payout.retry_count = (payout.retry_count or 0) + 1
    payout.status = 'pending'
    payout.failure_reason = None
    await db.commit()
    
    # Trigger the payout again
    background_tasks.add_task(
        _process_payout,
        str(payout.id),
        admin['id']
    )
    
    return {
        "success": True,
        "message": "Payout retry initiated",
        "payout_id": str(payout.id)
    }


@router.post("/webhook")
async def payoneer_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Handle Payoneer webhook notifications"""
    try:
        payload = await request.json()
        event_type = payload.get("eventType")
        data = payload.get("data", {})
        
        logger.info(f"Payoneer webhook received: {event_type}")
        
        if event_type == "payout.completed":
            transfer_id = data.get("transferId")
            if transfer_id:
                # Find and update the payout
                payout_result = await db.execute(
                    select(DeveloperPayout)
                    .where(DeveloperPayout.payoneer_transfer_id == transfer_id)
                )
                payout = payout_result.scalar_one_or_none()
                
                if payout:
                    payout.status = 'completed'
                    payout.payoneer_transfer_status = 'completed'
                    payout.completed_at = datetime.utcnow()
                    await db.commit()
                    
                    # Notify developer
                    await _notify_developer_payout_completed(
                        str(payout.developer_id),
                        payout.net_amount,
                        transfer_id,
                        db
                    )
                    
                    logger.info(f"✅ Payout {payout.id} completed via Payoneer")
        
        elif event_type == "payout.failed":
            transfer_id = data.get("transferId")
            if transfer_id:
                payout_result = await db.execute(
                    select(DeveloperPayout)
                    .where(DeveloperPayout.payoneer_transfer_id == transfer_id)
                )
                payout = payout_result.scalar_one_or_none()
                
                if payout:
                    payout.status = 'failed'
                    payout.payoneer_transfer_status = 'failed'
                    payout.failure_reason = data.get("failureReason", "Unknown")
                    await db.commit()
                    
                    logger.error(f"❌ Payout {payout.id} failed: {payout.failure_reason}")
        
        elif event_type == "payee.onboarding.completed":
            payee_id = data.get("payeeId")
            if payee_id:
                # Find and update the user
                user_result = await db.execute(
                    select(User).where(User.payoneer_payee_id == payee_id)
                )
                user = user_result.scalar_one_or_none()
                
                if user:
                    user.payoneer_onboarding_completed = True
                    user.payoneer_payee_status = "active"
                    user.payoneer_verified = True
                    await db.commit()
                    
                    # Notify user
                    await _notify_user_onboarding_completed(str(user.id), db)
                    
                    logger.info(f"✅ User {user.email} completed Payoneer onboarding")
        
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Payoneer webhook error: {str(e)}")
        return {"status": "error", "error": str(e)}


# ============================================
# Helper Functions
# ============================================

async def _generate_onboarding_link(user: User, db: AsyncSession):
    """Generate a new onboarding link for an existing payee"""
    result = await payoneer_service.create_payee_signup_link(
        payee_id=user.payoneer_payee_id,
        email=user.email,
        name=user.name,
        redirect_url=f"{settings.FRONTEND_URL}/payout-settings?payoneer_success=true"
    )
    
    if result.get("success"):
        user.payoneer_onboarding_url = result.get("link")
        user.payoneer_payee_status = "onboarding"
        await db.commit()
        
        return {
            "success": True,
            "onboarding_url": result.get("link"),
            "payee_id": user.payoneer_payee_id,
            "message": "New onboarding link generated"
        }
    else:
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to generate link"))


async def _process_payout(payout_id: str, admin_id: str):
    """Background task to process a payout"""
    from app.database import AsyncSessionLocal
    
    async with AsyncSessionLocal() as db:
        try:
            payout_result = await db.execute(
                select(DeveloperPayout)
                .where(DeveloperPayout.id == uuid.UUID(payout_id))
            )
            payout = payout_result.scalar_one_or_none()
            
            if not payout or payout.status != 'pending':
                return
            
            dev_result = await db.execute(
                select(User).where(User.id == payout.developer_id)
            )
            developer = dev_result.scalar_one_or_none()
            
            if not developer or not developer.payoneer_payee_id:
                return
            
            result = await payoneer_service.initiate_payout(
                payee_id=developer.payoneer_payee_id,
                amount=float(payout.net_amount),
                currency=payout.currency or "USD",
                description=f"Payment for project {payout.project_id}",
                reference_id=str(payout.id)
            )
            
            if result.get("success"):
                payout.payoneer_transfer_id = result.get("transfer_id")
                payout.payoneer_transfer_status = result.get("status", "pending")
                payout.status = 'processing'
                payout.processed_by = uuid.UUID(admin_id)
                payout.processed_at = datetime.utcnow()
                payout.transaction_id = result.get("transfer_id")
                await db.commit()
            else:
                payout.status = 'failed'
                payout.failure_reason = result.get("error", "Unknown error")
                await db.commit()
                
        except Exception as e:
            logger.error(f"Background payout processing error: {str(e)}")


async def _notify_developer_payout(developer_id: uuid.UUID, amount: float, transfer_id: str, db: AsyncSession):
    """Notify developer that payout is being processed"""
    notification = Notification(
        user_id=developer_id,
        type='payout_processing',
        title='💳 Payout Processing',
        message=f'Your payout of ${float(amount):.2f} is being processed via Payoneer. Funds will arrive shortly.',
        link='/dashboard',
        details={
            'amount': float(amount),
            'transfer_id': transfer_id,
            'method': 'Payoneer'
        },
        read=False
    )
    db.add(notification)
    await db.commit()


async def _notify_developer_payout_completed(developer_id: str, amount: float, transfer_id: str, db: AsyncSession):
    """Notify developer that payout is completed"""
    notification = Notification(
        user_id=uuid.UUID(developer_id),
        type='payout_completed',
        title='✅ Payout Completed!',
        message=f'Your payout of ${float(amount):.2f} has been sent via Payoneer! Transaction ID: {transfer_id}',
        link='/dashboard',
        details={
            'amount': float(amount),
            'transfer_id': transfer_id,
            'method': 'Payoneer'
        },
        read=False
    )
    db.add(notification)
    await db.commit()


async def _notify_user_onboarding_completed(user_id: str, db: AsyncSession):
    """Notify user that Payoneer onboarding is complete"""
    notification = Notification(
        user_id=uuid.UUID(user_id),
        type='payoneer_onboarding_completed',
        title='✅ Payoneer Account Verified!',
        message='Your Payoneer account has been successfully verified. You are now ready to receive payouts.',
        link='/dashboard',
        details={
            'method': 'Payoneer',
            'status': 'active'
        },
        read=False
    )
    db.add(notification)
    await db.commit()