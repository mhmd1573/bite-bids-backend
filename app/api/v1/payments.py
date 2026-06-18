# app/api/v1/payments.py
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.database import get_db
from app.models.payment import CheckoutSession, DeveloperPayout
from app.models.project import Project
from app.models.user import User
from app.models.bid import Bid
from app.models.notification import Notification
from app.schemas.payment import StripeCheckoutRequest
from app.core.dependencies import get_current_user
from app.core.constants import PLATFORM_FEE_PERCENTAGE, PLATFORM_FIXED_FEE, PROJECT_POSTING_FEE
from app.core.exceptions import PaymentRequiredException
from app.utils.converters import model_to_dict

from app.services.notification_service import NotificationService
from app.services.email_service import EmailService

from app.config import settings

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
        import stripe
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
        import stripe
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
        customer_id_uuid = uuid.UUID(current_user.get('id'))
        project_id_uuid = uuid.UUID(request.project_id) if request.project_id else None
        
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
        raise HTTPException(status_code=500, detail=f"Failed to create checkout session: {str(e)}")


@router.post("/stripe/webhook")
async def stripe_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Handle Stripe webhook events
    """
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    webhook_secret = settings.STRIPE_WEBHOOK_SECRET
    
    try:
        import stripe
        stripe.api_key = settings.STRIPE_SECRET_KEY
        
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        ) if webhook_secret else json.loads(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Handle the event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        
        # Stripe metadata
        metadata = session.get('metadata', {}) or {}
        payment_type = metadata.get('payment_type')
        project_id_str = metadata.get('project_id')
        winner_bid_id_str = metadata.get('winner_bid_id')
        notification_id_str = metadata.get('notification_id')
        investor_id_str = metadata.get('user_id')
        investor_email = metadata.get('user_email')
        
        amount_total = session.get('amount_total')
        amount_paid = (amount_total / 100.0) if amount_total is not None else float(metadata.get('amount', 0) or 0)

        # Update checkout session record
        stmt = select(CheckoutSession).where(
            CheckoutSession.session_id == session['id']
        )
        result = await db.execute(stmt)
        checkout_record = result.scalar_one_or_none()
        
        if checkout_record:
            checkout_record.status = 'completed'
            checkout_record.completed_at = datetime.utcnow()
            await db.commit()
        
        # Handle project posting payment
        if payment_type == 'project_posting':
            user_id = metadata.get('user_id')
            
            if user_id:
                try:
                    user_uuid = uuid.UUID(user_id)
                    user_result = await db.execute(
                        select(User).where(User.id == user_uuid)
                    )
                    user = user_result.scalar_one_or_none()
                    
                    if user:
                        # Add 1 posting credit
                        user.posting_credits = (user.posting_credits or 0) + 1
                        await db.commit()
                        
                        # Send notification
                        await NotificationService.send_notification_to_user(
                            user_id,
                            {
                                "type": "payment_verified",
                                "title": "Posting Credit Added! ✓",
                                "message": f"You now have {user.posting_credits} posting credit(s). Ready to post your project!",
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
                except Exception as e:
                    pass
            
            return {"status": "success", "payment_type": "project_posting"}

        # Handle project payment
        project = None
        investor_user = None
        developer_user = None

        if project_id_str:
            project = await db.scalar(
                select(Project).where(Project.id == uuid.UUID(project_id_str))
            )
        
        if investor_id_str:
            investor_user = await db.scalar(
                select(User).where(User.id == uuid.UUID(investor_id_str))
            )
        
        if project and project.developer_id:
            developer_user = await db.scalar(
                select(User).where(User.id == project.developer_id)
            )

        # Update project status
        if project:
            if project.status == 'winner_selected':
                project.status = 'in_progress'
            project.updated_at = datetime.utcnow()
            await db.commit()

        # Mark notification as read
        if notification_id_str:
            try:
                notif_uuid = uuid.UUID(notification_id_str)
                await db.execute(
                    update(Notification)
                    .where(Notification.id == notif_uuid)
                    .values(read=True, read_at=datetime.utcnow())
                )
                await db.commit()
            except:
                pass

        # Create notification for developer
        if project and developer_user:
            try:
                formatted_amount = f"${amount_paid:,.2f}"
                dev_notif = Notification(
                    user_id=project.developer_id,
                    type="payment_received",
                    title="💰 Investor payment received",
                    message=(
                        f"The investor {investor_user.name if investor_user else 'a client'} "
                        f"has completed a payment of {formatted_amount} for your project "
                        f"'{project.title}'. You can now start working on the project."
                    ),
                    link=str(project.id),
                    details={
                        "project_id": str(project.id),
                        "project_title": project.title,
                        "amount": float(amount_paid),
                        "winner_bid_id": winner_bid_id_str,
                        "investor_id": investor_id_str,
                    },
                    read=False
                )
                db.add(dev_notif)
                await db.commit()
            except:
                pass

        # Create chat room
        if project and project.developer_id and investor_id_str:
            try:
                from app.models.chat import ChatRoom
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
                    await db.commit()
                    await db.refresh(chat_room)
                    
                    # System message
                    system_message = ChatMessage(
                        room_id=chat_room.id,
                        sender_id=project.developer_id,
                        message=f"Payment of {formatted_amount} completed successfully! 🎉 Chat room is now active.",
                        message_type="system"
                    )
                    db.add(system_message)
                    await db.commit()
                    
                    # Send notifications
                    await send_notification_to_user(
                        str(project.developer_id),
                        {
                            "type": "chat_room_created",
                            "title": "Chat Room Created! 💬",
                            "message": f"Payment received! Chat room for '{project.title}' is now active.",
                            "link": f"/chat/{chat_room.id}",
                            "details": {
                                "project_id": str(project.id),
                                "room_id": str(chat_room.id),
                                "amount": float(amount_paid),
                            }
                        },
                        db
                    )
                    
                    await send_notification_to_user(
                        investor_id_str,
                        {
                            "type": "chat_room_created",
                            "title": "Chat Room Created! 💬",
                            "message": f"Payment successful! Chat room for '{project.title}' is now active.",
                            "link": f"/chat/{chat_room.id}",
                            "details": {
                                "project_id": str(project.id),
                                "room_id": str(chat_room.id),
                                "amount": float(amount_paid),
                            }
                        },
                        db
                    )
            except:
                pass

        # Send confirmation emails
        try:
            project_title = project.title if project else "your project"
            
            if investor_email:
                await send_payment_confirmation_email(
                    to_email=investor_email,
                    project_title=project_title,
                    amount=amount_paid,
                    role="investor"
                )
            
            if developer_user and developer_user.email:
                await send_payment_confirmation_email(
                    to_email=developer_user.email,
                    project_title=project_title,
                    amount=amount_paid,
                    role="developer"
                )
        except:
            pass
    
    elif event['type'] == 'checkout.session.expired':
        session = event['data']['object']
        
        stmt = select(CheckoutSession).where(
            CheckoutSession.session_id == session['id']
        )
        result = await db.execute(stmt)
        checkout_record = result.scalar_one_or_none()
        
        if checkout_record:
            checkout_record.status = 'expired'
            await db.commit()
    
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