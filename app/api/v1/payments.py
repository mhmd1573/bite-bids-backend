# app/api/v1/payments.py
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import os
import asyncio
import logging

from app.database import get_db
from app.models.payment import CheckoutSession, DeveloperPayout
from app.models.project import Project
from app.models.user import User
from app.models.bid import Bid
from app.models.notification import Notification
from app.models.chat import ChatRoom, ChatMessage
from app.schemas.payment import StripeCheckoutRequest
from app.core.dependencies import get_current_user
from app.core.constants import PLATFORM_FEE_PERCENTAGE, PLATFORM_FIXED_FEE, PROJECT_POSTING_FEE
from app.core.exceptions import PaymentRequiredException
from app.utils.converters import model_to_dict

from app.services.notification_service import NotificationService
from app.services.email_service import EmailService
from app.core.websocket_manager import manager

from app.config import settings

# Setup logger
logger = logging.getLogger(__name__)

import stripe

router = APIRouter(prefix="/payments", tags=["Payments"])


@router.post("/create-post-project-session")
async def create_post_project_payment_session(
    request: dict,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Create a Stripe Checkout Session for project posting fee ($0.99)
    """
    
    if not settings.STRIPE_SECRET_KEY:
        raise HTTPException(
            status_code=500, 
            detail="Stripe is not configured. Please contact support."
        )
    
    try:
        stripe.api_key = settings.STRIPE_SECRET_KEY
        
        customer_email = request.get('customer_email')
        customer_name = request.get('customer_name')
        
        # Convert to cents for Stripe
        amount_cents = int(PROJECT_POSTING_FEE * 100)
        
        # Build metadata
        metadata = {
            'user_id': current_user.get('id'),
            'user_email': customer_email,
            'payment_type': 'project_posting',
            'amount': str(PROJECT_POSTING_FEE)
        }
        
        # Calculate expiry time (30 minutes)
        expiry = datetime.utcnow() + timedelta(minutes=30)
        expiry_ts = int(expiry.replace(tzinfo=timezone.utc).timestamp())

        # Create Stripe Checkout Session
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'unit_amount': amount_cents,
                    'product_data': {
                        'name': 'BiteBids Project Posting Fee',
                        'description': 'One-time fee to post a project on BiteBids marketplace',
                    },
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=f"{settings.FRONTEND_URL}/marketplace?session_id={{CHECKOUT_SESSION_ID}}&payment_status=success",
            cancel_url=f"{settings.FRONTEND_URL}/marketplace?payment_status=cancelled",
            customer_email=customer_email,
            metadata=metadata,
            payment_intent_data={
                'metadata': metadata,
                'description': 'BiteBids Project Posting Fee',
            },
            expires_at=expiry_ts,
        )
        
        # Create database record
        customer_id_uuid = uuid.UUID(current_user.get('id'))
        
        checkout_record = CheckoutSession(
            id=uuid.uuid4(),
            session_id=checkout_session.id,
            order_reference=f"post_project_{uuid.uuid4()}",
            external_reference=checkout_session.id,
            project_id=None,
            amount=Decimal(str(PROJECT_POSTING_FEE)),
            total_with_fees=Decimal(str(PROJECT_POSTING_FEE)),
            fees=Decimal('0'),
            payment_method='stripe_card',
            customer_id=customer_id_uuid,
            status='pending',
            payment_url=checkout_session.url,
            payment_method_used='stripe',
            expires_at=datetime.utcnow() + timedelta(minutes=30),
            extra_data={
                'stripe_session_id': checkout_session.id,
                'payment_type': 'project_posting',
                'metadata': metadata
            }
        )
        
        db.add(checkout_record)
        await db.commit()
        await db.refresh(checkout_record)
        
        return {
            "success": True,
            "session_id": checkout_session.id,
            "checkout_url": checkout_session.url,
            "order_reference": checkout_record.order_reference,
            "amount": PROJECT_POSTING_FEE
        }
        
    except Exception as e:
        logger.error(f"Error creating post project session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create payment session: {str(e)}")


@router.post("/stripe/create-checkout-session")
async def create_stripe_checkout_session(
    request: StripeCheckoutRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Create a Stripe Checkout Session for payment processing
    """
    
    if not settings.STRIPE_SECRET_KEY:
        raise HTTPException(
            status_code=500, 
            detail="Stripe is not configured. Please contact support."
        )
    
    try:
        stripe.api_key = settings.STRIPE_SECRET_KEY
        
        # Calculate fees
        base_amount = request.amount
        fee_percentage = PLATFORM_FEE_PERCENTAGE / 100
        fees = round(base_amount * fee_percentage + PLATFORM_FIXED_FEE, 2)
        total_amount = base_amount + fees
        
        # Convert to cents for Stripe
        amount_cents = int(total_amount * 100)
        
        # Build metadata
        metadata = {
            'item_id': request.item_id,
            'user_id': current_user.get('id'),
            'user_email': request.customer_email,
            'base_amount': str(base_amount),
            'fees': str(fees),
            'total_amount': str(total_amount),
        }
        
        if request.project_id:
            metadata['project_id'] = request.project_id
        if request.winner_bid_id:
            metadata['winner_bid_id'] = request.winner_bid_id
        if request.notification_id:
            metadata['notification_id'] = request.notification_id
        
        # Determine item name
        item_name = f"BiteBids {request.order_type.title()} Payment"
        if request.order_type == 'auction':
            item_name = "Auction Winning Bid Payment"

        # Parse UUIDs early (needed for duplicate-purchase check below)
        customer_id_uuid = uuid.UUID(current_user.get('id'))
        project_id_uuid = uuid.UUID(request.project_id) if request.project_id else None

        # ✅ Prevent duplicate purchases for fixed-price projects
        if request.order_type == 'fixed' and request.project_id:
            existing_purchase = await db.scalar(
                select(CheckoutSession).where(
                    and_(
                        CheckoutSession.project_id == project_id_uuid,
                        CheckoutSession.customer_id == customer_id_uuid,
                        CheckoutSession.status == 'completed'
                    )
                )
            )
            if existing_purchase:
                raise HTTPException(
                    status_code=400,
                    detail="You have already purchased this project."
                )

        expiry = datetime.utcnow() + timedelta(minutes=30)
        expiry_ts = int(expiry.replace(tzinfo=timezone.utc).timestamp())

        # Create Stripe Checkout Session
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'unit_amount': amount_cents,
                    'product_data': {
                        'name': item_name,
                        'description': f'Payment for project: {request.item_id}',
                    },
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=f"{settings.FRONTEND_URL}/payment/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{settings.FRONTEND_URL}/payment/cancel",
            customer_email=request.customer_email,
            metadata=metadata,
            billing_address_collection='required',
            payment_intent_data={
                'metadata': metadata,
                'description': f'BiteBids payment for {item_name}',
            },
            expires_at=expiry_ts,
        )
        
        # Create database record
        checkout_record = CheckoutSession(
            id=uuid.uuid4(),
            session_id=checkout_session.id,
            order_reference=f"stripe_{uuid.uuid4()}",
            external_reference=checkout_session.id,
            project_id=project_id_uuid,
            amount=base_amount,
            total_with_fees=total_amount,
            fees=fees,
            payment_method='stripe_card',
            customer_id=customer_id_uuid,
            status='pending',
            payment_url=checkout_session.url,
            payment_method_used='stripe',
            expires_at=datetime.utcnow() + timedelta(minutes=30),
            extra_data={
                'stripe_session_id': checkout_session.id,
                'metadata': metadata
            }
        )
        
        db.add(checkout_record)
        await db.commit()
        await db.refresh(checkout_record)
        
        return {
            "success": True,
            "session_id": checkout_session.id,
            "checkout_url": checkout_session.url,
            "order_reference": checkout_record.order_reference,
            "amount": base_amount,
            "fees": fees,
            "total": total_amount
        }
        
    except Exception as e:
        logger.error(f"Error creating checkout session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create checkout session: {str(e)}")


@router.post("/stripe/webhook")
async def stripe_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Handle Stripe webhook events
    This is called by Stripe when payment status changes
    """
    
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    webhook_secret = settings.STRIPE_WEBHOOK_SECRET or os.getenv('STRIPE_WEBHOOK_SECRET')
    
    logger.info(f"📨 Webhook received. Signature header present: {bool(sig_header)}")
    
    try:
        stripe.api_key = settings.STRIPE_SECRET_KEY
        
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        ) if webhook_secret else json.loads(payload)
        
        logger.info(f"📋 Webhook event type: {event['type']}")
        
    except ValueError as e:
        logger.error(f"Invalid payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid signature: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    # Handle the event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        session_id = session.get('id')
        
        logger.info(f"✅ Processing checkout.session.completed: {session_id}")
        
        metadata = session.get('metadata', {}) or {}
        payment_type = metadata.get('payment_type')
        project_id_str = metadata.get('project_id')
        winner_bid_id_str = metadata.get('winner_bid_id')
        notification_id_str = metadata.get('notification_id')
        investor_id_str = metadata.get('user_id')
        investor_email = metadata.get('user_email')
        
        amount_total = session.get('amount_total')
        amount_paid = (amount_total / 100.0) if amount_total is not None else float(metadata.get('amount', 0) or 0)

        # 1) Update checkout session record
        stmt = select(CheckoutSession).where(
            CheckoutSession.session_id == session_id
        )
        result = await db.execute(stmt)
        checkout_record = result.scalar_one_or_none()
        
        if checkout_record:
            checkout_record.status = 'completed'
            checkout_record.completed_at = datetime.utcnow()
            await db.commit()
            logger.info(f"Payment completed for session: {session_id}")
        else:
            logger.warning(f"Checkout session {session_id} not found in database")
        
        # ============================================
        # Handle project posting payment ($0.99)
        # ============================================
      
        if payment_type == 'project_posting':
            user_id = metadata.get('user_id')
            user_email = metadata.get('user_email')
            
            logger.info(f"🎯 Project posting payment verified for user {user_id}")
            
            if user_id:
                try:
                    user_uuid = uuid.UUID(user_id)
                    user_result = await db.execute(
                        select(User).where(User.id == user_uuid)
                    )
                    user = user_result.scalar_one_or_none()
                    
                    if user:
                        old_credits = user.posting_credits or 0
                        user.posting_credits = old_credits + 1
                        await db.commit()
                        await db.refresh(user)
                        
                        logger.info(f"✅ Added posting credit. User {user.email}: {old_credits} → {user.posting_credits}")
                        
                        # ✅ CORRECT: Use NotificationService.send_notification_to_user
                        await NotificationService.send_notification_to_user(
                            user_id,
                            {
                                "type": "payment_verified",
                                "title": "✅ Credit Added!",
                                "message": f"Payment confirmed! You have {user.posting_credits} credit(s) available.",
                                "link": "/dashboard",
                                "details": {
                                    "payment_type": "project_posting",
                                    "amount": float(amount_paid),
                                    "credits_added": 1,
                                    "total_credits": user.posting_credits
                                }
                            },
                            db
                        )
                        logger.info(f"✅ Notification sent to user {user.email}")
                        
                    else:
                        logger.error(f"❌ User {user_id} not found")
                        
                except Exception as e:
                    logger.error(f"❌ Failed to add posting credit: {e}")
                    import traceback
                    traceback.print_exc()
            
            return {"status": "success", "payment_type": "project_posting"}

        # ============================================
        # Handle project payment (auction/fixed price)
        # ============================================
        project = None
        investor_user = None
        developer_user = None

        try:
            if project_id_str:
                project = await db.scalar(
                    select(Project).where(Project.id == uuid.UUID(project_id_str))
                )
                logger.info(f"📁 Project found: {project.title if project else 'None'}")
        except ValueError:
            logger.warning(f"Invalid project_id in Stripe metadata: {project_id_str}")

        try:
            if investor_id_str:
                investor_user = await db.scalar(
                    select(User).where(User.id == uuid.UUID(investor_id_str))
                )
        except ValueError:
            logger.warning(f"Invalid investor user_id in Stripe metadata: {investor_id_str}")

        if project and project.developer_id:
            developer_user = await db.scalar(
                select(User).where(User.id == project.developer_id)
            )

        if project:
            if project.status == 'winner_selected':
                project.status = 'in_progress'
            project.updated_at = datetime.utcnow()
            await db.commit()
            logger.info(f"Project {project.id} status: {project.status} after payment")
            
            # 📡 Broadcast live update so viewers see the project go 'in_progress' after payment
            await manager.broadcast_data_change(
                event_type="project_updated",
                data=model_to_dict(project),
                project_id=str(project.id)
            )
            # 📡 Notify admin dashboards of a new payment
            await manager.broadcast_data_change(
                event_type="admin_update",
                data={"entity": "payment", "action": "completed", "project_id": str(project.id)},
                project_id=str(project.id)
            )

        if notification_id_str:
            try:
                notif_uuid = uuid.UUID(notification_id_str)
                notif_result = await db.execute(
                    select(Notification).where(Notification.id == notif_uuid)
                )
                existing_notif = notif_result.scalar_one_or_none()
                if existing_notif:
                    existing_notif.read = True
                    existing_notif.read_at = datetime.utcnow()
                    await db.commit()
                    logger.info(f"Notification {notification_id_str} marked as read after payment")
            except ValueError:
                logger.warning(f"Invalid notification_id in Stripe metadata: {notification_id_str}")

        if project and developer_user:
            try:
                formatted_amount = f"${amount_paid:,.2f}"
                dev_notif = Notification(
                    user_id=project.developer_id,
                    type="payment_received",
                    title=f"💰 Payment Received: {formatted_amount}",
                    message=f"{investor_user.name if investor_user else 'Investor'} paid {formatted_amount} for '{project.title[:50]}'. Funds secured in escrow.",
                    link=str(project.id),
                    details={
                        "project_id": str(project.id),
                        "project_title": project.title,
                        "amount": float(amount_paid),
                        "winner_bid_id": winner_bid_id_str,
                        "investor_id": investor_id_str,
                        "checkout_session_id": session_id,
                        "payment_type": "project_payment",
                        "platform_fee_percentage": PLATFORM_FEE_PERCENTAGE,
                        "platform_fixed_fee": PLATFORM_FIXED_FEE,
                    },
                    read=False
                )
                db.add(dev_notif)
                await db.commit()
                logger.info(
                    f"Developer notification created for payment (project={project.id}, developer={project.developer_id})"
                )
            except Exception as e:
                logger.error(f"Failed to create developer payment notification: {e}")

        # CREATE CHAT ROOM AFTER PAYMENT
        if project and project.developer_id and investor_id_str:
            try:
                room_result = await db.execute(
                    select(ChatRoom).where(
                        and_(
                            ChatRoom.project_id == project.id,
                            ChatRoom.investor_id == uuid.UUID(investor_id_str)
                        )
                    )
                )
                existing_room = room_result.scalar_one_or_none()
                
                if not existing_room:
                    chat_room = ChatRoom(
                        project_id=project.id,
                        developer_id=project.developer_id,
                        investor_id=uuid.UUID(investor_id_str),
                        status="active"
                    )
                    
                    db.add(chat_room)
                    await db.flush()
                    
                    formatted_amount = f"${amount_paid:,.2f}"
                    system_message = ChatMessage(
                        room_id=chat_room.id,
                        sender_id=project.developer_id,
                        message=f"Payment of {formatted_amount} completed successfully! 🎉 Chat room is now active. You can start discussing the project.",
                        message_type="system"
                    )
                    
                    db.add(system_message)
                    await db.commit()
                    
                    logger.info(f"✅ Chat room created: {chat_room.id} for project {project.id} with investor {investor_id_str}")
                    
                    # ✅ CORRECT: Use NotificationService.send_notification_to_user
                    await NotificationService.send_notification_to_user(
                        str(project.developer_id),
                        {
                            "type": "chat_room_created",
                            "title": "💬 Chat with Investor Ready",
                            "message": f"Payment received! Discuss '{project.title[:50]}' with {investor_user.name if investor_user else 'investor'}.",
                            "link": f"/chat/{chat_room.id}",
                            "details": {
                                "project_id": str(project.id),
                                "room_id": str(chat_room.id),
                                "amount": float(amount_paid),
                                "investor_name": investor_user.name if investor_user else "Investor"
                            }
                        },
                        db
                    )
                    
                    await NotificationService.send_notification_to_user(
                        investor_id_str,
                        {
                            "type": "chat_room_created",
                            "title": "💬 Chat with Developer Ready",
                            "message": f"Payment confirmed! Discuss '{project.title[:50]}' with {developer_user.name if developer_user else 'developer'}.",
                            "link": f"/chat/{chat_room.id}",
                            "details": {
                                "project_id": str(project.id),
                                "room_id": str(chat_room.id),
                                "amount": float(amount_paid),
                                "developer_name": developer_user.name if developer_user else "Developer"
                            }
                        },
                        db
                    )
                    
                    logger.info(f"✅ Chat room notifications sent for room {chat_room.id}")
                else:
                    logger.info(f"ℹ️ Chat room already exists for project {project.id} with investor {investor_id_str}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to create chat room: {e}")

        # Send emails
        try:
            project_title = project.title if project else "your project"

            if investor_email:
                asyncio.create_task(
                    EmailService.send_payment_confirmation_email(
                        to_email=investor_email,
                        project_title=project_title,
                        amount=amount_paid,
                        role="investor"
                    )
                )
            elif investor_user and investor_user.email:
                asyncio.create_task(
                    EmailService.send_payment_confirmation_email(
                        to_email=investor_user.email,
                        project_title=project_title,
                        amount=amount_paid,
                        role="investor"
                    )
                )

            if developer_user and developer_user.email:
                asyncio.create_task(
                    EmailService.send_payment_confirmation_email(
                        to_email=developer_user.email,
                        project_title=project_title,
                        amount=amount_paid,
                        role="developer"
                    )
                )

        except Exception as e:
            logger.error(f"Failed to schedule payment confirmation emails: {e}")
    
    elif event['type'] == 'checkout.session.expired':
        session = event['data']['object']
        session_id = session.get('id')
        
        logger.info(f"⏰ Checkout session expired: {session_id}")
        
        stmt = select(CheckoutSession).where(
            CheckoutSession.session_id == session_id
        )
        result = await db.execute(stmt)
        checkout_record = result.scalar_one_or_none()
        
        if checkout_record:
            checkout_record.status = 'expired'
            await db.commit()
            logger.info(f"Payment session expired: {session_id}")
    
    return {"status": "success"}


@router.get("/session/{session_id}")
async def get_checkout_session(
    session_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get checkout session details"""
    result = await db.execute(
        select(CheckoutSession).where(CheckoutSession.session_id == session_id)
    )
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail="Checkout session not found")
    
    if str(session.customer_id) != current_user["_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return model_to_dict(session)


@router.get("/stripe/verify-session/{session_id}")
async def verify_stripe_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Verify a Stripe checkout session status
    """
    try:
        if not settings.STRIPE_SECRET_KEY:
            raise HTTPException(
                status_code=500, 
                detail="Stripe is not configured. Please contact support."
            )
        
        stripe.api_key = settings.STRIPE_SECRET_KEY
        
        session = stripe.checkout.Session.retrieve(session_id)
        
        query = select(CheckoutSession).where(CheckoutSession.session_id == session_id)
        result = await db.execute(query)
        checkout_record = result.scalar_one_or_none()
        
        if not checkout_record:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if str(checkout_record.customer_id) != current_user.get('id'):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if session.payment_status == 'paid':
            if checkout_record.status != 'completed':
                checkout_record.status = 'completed'
                checkout_record.completed_at = datetime.utcnow()
                await db.commit()
            
            return {
                "success": True, 
                "payment_status": "completed",
                "session": {
                    "id": session.id,
                    "amount_total": session.amount_total / 100 if session.amount_total else 0,
                    "currency": session.currency,
                    "customer_email": session.customer_email,
                    "payment_status": session.payment_status,
                    "status": session.status
                }
            }
        
        return {
            "success": False,
            "payment_status": session.payment_status,
            "session": {
                "id": session.id,
                "amount_total": session.amount_total / 100 if session.amount_total else 0,
                "currency": session.currency,
                "customer_email": session.customer_email,
                "payment_status": session.payment_status,
                "status": session.status
            }
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {e}")
        raise HTTPException(status_code=400, detail=f"Stripe error: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to verify session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to verify session: {str(e)}")