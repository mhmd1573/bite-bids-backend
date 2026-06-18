# app/api/v1/chat.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, or_, func
import uuid
import os
import shutil
from datetime import datetime, timezone
from typing import Optional

from app.database import get_db
from app.models.chat import ChatRoom, ChatMessage
from app.models.project import Project
from app.models.user import User
from app.models.payment import CheckoutSession
from app.models.notification import Notification
from app.schemas.chat import ChatMessageCreate
from app.core.dependencies import get_current_user, get_current_admin
from app.core.websocket_manager import manager
from app.core.security import generate_verification_token
from app.core.constants import ALLOWED_IMAGE_EXTENSIONS, MAX_UPLOAD_SIZE
from app.utils.converters import model_to_dict
from app.services.notification_service import NotificationService
from app.services.moderation_service import ModerationService
from app.config import settings

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.get("/rooms")
async def get_user_chat_rooms(
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get all chat rooms for current user"""
    try:
        user_id = uuid.UUID(current_user["id"])
        
        result = await db.execute(
            select(ChatRoom)
            .where(
                or_(
                    ChatRoom.developer_id == user_id,
                    ChatRoom.investor_id == user_id
                )
            )
            .order_by(ChatRoom.created_at.desc())
        )
        rooms = result.scalars().all()
        
        return [model_to_dict(room) for room in rooms]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rooms/create/{project_id}")
async def create_chat_room(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Create a chat room for a project after payment is completed"""
    try:
        # Verify project exists
        project_result = await db.execute(
            select(Project).where(Project.id == uuid.UUID(project_id))
        )
        project = project_result.scalar_one_or_none()
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Check if payment is completed for this project by this investor
        payment_result = await db.execute(
            select(CheckoutSession)
            .where(
                and_(
                    CheckoutSession.project_id == uuid.UUID(project_id),
                    CheckoutSession.customer_id == uuid.UUID(current_user['id']),
                    CheckoutSession.status == "completed"
                )
            )
        )
        payment = payment_result.scalar_one_or_none()
        
        if not payment:
            raise HTTPException(
                status_code=403, 
                detail="Payment must be completed before creating chat room"
            )
        
        # Get investor_id from payment
        investor_id = payment.customer_id
        
        # Check if chat room already exists for this developer-investor pair
        room_result = await db.execute(
            select(ChatRoom).where(
                and_(
                    ChatRoom.project_id == uuid.UUID(project_id),
                    ChatRoom.investor_id == investor_id
                )
            )
        )
        existing_room = room_result.scalar_one_or_none()
        
        if existing_room:
            return {
                "room_id": str(existing_room.id),
                "message": "Chat room already exists for this investor"
            }
        
        # Create new chat room
        chat_room = ChatRoom(
            project_id=uuid.UUID(project_id),
            developer_id=project.developer_id,
            investor_id=investor_id,
            status="active"
        )
        
        db.add(chat_room)
        await db.commit()
        await db.refresh(chat_room)
        
        # Create system message
        system_message = ChatMessage(
            room_id=chat_room.id,
            sender_id=project.developer_id,
            message="Chat room created. Payment completed successfully. You can now communicate about the project.",
            message_type="system"
        )
        
        db.add(system_message)
        await db.commit()
        
        # Send notifications
        await send_notification_to_user(
            str(project.developer_id),
            {
                "type": "chat_room_created",
                "title": "Chat Room Created",
                "message": f"Chat room for project '{project.title}' is now active",
                "link": f"/chat/{chat_room.id}",
                "details": {
                    "project_id": project_id,
                    "room_id": str(chat_room.id)
                }
            },
            db
        )
        
        await send_notification_to_user(
            str(investor_id),
            {
                "type": "chat_room_created",
                "title": "Chat Room Created",
                "message": f"Chat room for project '{project.title}' is now active",
                "link": f"/chat/{chat_room.id}",
                "details": {
                    "project_id": project_id,
                    "room_id": str(chat_room.id)
                }
            },
            db
        )
        
        return {
            "room_id": str(chat_room.id),
            "project_id": project_id,
            "developer_id": str(project.developer_id),
            "investor_id": str(investor_id),
            "status": "active",
            "created_at": chat_room.created_at.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rooms/{room_id}")
async def get_chat_room(
    room_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get chat room details"""
    try:
        result = await db.execute(
            select(ChatRoom).where(ChatRoom.id == uuid.UUID(room_id))
        )
        room = result.scalar_one_or_none()
        
        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")
        
        # Verify user is part of this chat
        user_id = uuid.UUID(current_user["id"])
        if user_id not in [room.developer_id, room.investor_id]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return model_to_dict(room)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rooms/{room_id}/messages")
async def send_chat_message(
    room_id: str,
    message_data: ChatMessageCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Send a chat message with optional moderation"""
    try:
        user_id = uuid.UUID(current_user['id'])
        message_text = message_data.message.strip() if message_data.message else ""

        if not message_text and not message_data.file_url:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Verify room exists and user is a participant
        room_query = select(ChatRoom).where(ChatRoom.id == uuid.UUID(room_id))
        result = await db.execute(room_query)
        room = result.scalar_one_or_none()
        
        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")
        
        if user_id not in [room.developer_id, room.investor_id]:
            raise HTTPException(status_code=403, detail="Not a participant in this chat")
        
        # Create message with pending moderation status
        new_message = ChatMessage(
            room_id=uuid.UUID(room_id),
            sender_id=user_id,
            message=message_text if message_text else None,
            file_url=message_data.file_url,
            file_name=message_data.file_name,
            file_size=message_data.file_size,
            file_type=message_data.file_type,
            message_type=message_data.message_type or "text",
            moderation_status='pending',
            flagged=False
        )
        db.add(new_message)

        recipient_id = room.investor_id if user_id == room.developer_id else room.developer_id

        # Update room's updated_at
        room.updated_at = datetime.utcnow()

        await db.commit()
        await db.refresh(new_message)

        message_dict = model_to_dict(new_message)

        # Send via WebSocket IMMEDIATELY
        await manager.send_chat_message(room_id, message_dict)

        # Schedule background moderation for text messages
        if message_text:
            from app.services.moderation_service import moderate_message_async
            background_tasks.add_task(
                moderate_message_async,
                str(new_message.id),
                message_text,
                room_id,
                str(user_id),
                current_user['email']
            )

        if recipient_id:
            # Update unread count
            total_unread = await get_total_unread_chat_count(db, recipient_id)
            await manager.send_ws_event(
                str(recipient_id),
                {
                    "type": "chat_unread_count",
                    "total_unread_count": total_unread,
                    "room_id": room_id
                }
            )
        
        return message_dict
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rooms/{room_id}/messages")
async def get_chat_messages(
    room_id: str,
    before: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get messages from a chat room"""
    try:
        # Verify room exists and user has access
        room_result = await db.execute(
            select(ChatRoom).where(ChatRoom.id == uuid.UUID(room_id))
        )
        room = room_result.scalar_one_or_none()

        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")

        user_id = uuid.UUID(current_user["id"])
        if user_id not in [room.developer_id, room.investor_id]:
            raise HTTPException(status_code=403, detail="Access denied")

        # Build query
        query = select(ChatMessage).where(ChatMessage.room_id == uuid.UUID(room_id))

        if before:
            query = query.where(ChatMessage.created_at < datetime.fromisoformat(before))

        query = query.order_by(ChatMessage.created_at.asc())
        
        result = await db.execute(query)
        messages = list(result.scalars().all())

        # Mark messages as read
        unread_ids = [msg.id for msg in messages if not msg.read and msg.sender_id != user_id]
        
        if unread_ids:
            await db.execute(
                update(ChatMessage)
                .where(ChatMessage.id.in_(unread_ids))
                .values(read=True, read_at=func.now())
            )
            await db.commit()
            total_unread = await get_total_unread_chat_count(db, user_id)
            await manager.send_ws_event(
                str(user_id),
                {"type": "chat_unread_count", "total_unread_count": total_unread}
            )
        
        # Serialize messages
        serialized_messages = []
        for msg in messages:
            msg_dict = {
                'id': str(msg.id),
                'room_id': str(msg.room_id),
                'sender_id': str(msg.sender_id) if msg.sender_id else None,
                'message': msg.message,
                'message_type': msg.message_type,
                'file_url': msg.file_url,
                'file_name': msg.file_name,
                'file_size': msg.file_size,
                'read': msg.read,
                'read_at': msg.read_at.isoformat() + 'Z' if msg.read_at else None,
                'created_at': msg.created_at.isoformat() + 'Z' if msg.created_at else None,
                'flagged': getattr(msg, 'flagged', False),
                'moderation_status': getattr(msg, 'moderation_status', 'approved'),
                'moderation_reason': getattr(msg, 'moderation_reason', None)
            }
            serialized_messages.append(msg_dict)

        return serialized_messages
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rooms/{room_id}/upload")
async def upload_chat_file(
    room_id: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Upload a file to a chat room"""
    temp_path = None

    try:
        # Authorization
        room_query = await db.execute(select(ChatRoom).where(ChatRoom.id == uuid.UUID(room_id)))
        room = room_query.scalar_one_or_none()
        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")

        user_id = uuid.UUID(current_user["id"])
        if user_id not in [room.developer_id, room.investor_id]:
            raise HTTPException(status_code=403, detail="Access denied")

        # File validation
        filename = file.filename
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext not in ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Only PNG and JPEG images are allowed.")

        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Cannot upload empty file.")

        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=400, detail=f"File too large. Max size: {MAX_UPLOAD_SIZE / 1024 / 1024} MB")

        # Save temporary file
        temp_dir = "uploads/temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_filename = f"{uuid.uuid4()}_{filename}"
        temp_path = os.path.join(temp_dir, temp_filename)
        with open(temp_path, 'wb') as f:
            f.write(content)

        # OpenAI moderation
        moderation_result = await ModerationService.moderate_image(temp_path)

        if moderation_result["contains_harmful_content"]:
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail="This image contains harmful content.")

        if moderation_result["contains_contact_info"]:
            os.remove(temp_path)
            raise HTTPException(
                status_code=400,
                detail="This image contains contact information (phone, email, social media, or URLs)."
            )

        # Save to permanent location
        upload_dir = settings.UPLOAD_DIR or "uploads/chat_files"
        os.makedirs(upload_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_filename = os.path.basename(filename).replace("..", "")
        safe_filename = f"{user_id}_{timestamp}_{clean_filename}"
        final_path = os.path.join(upload_dir, safe_filename)
        shutil.move(temp_path, final_path)
        temp_path = None

        # Create database message
        new_message = ChatMessage(
            id=uuid.uuid4(),
            room_id=uuid.UUID(room_id),
            sender_id=user_id,
            message=f"🖼️ Shared an image: {filename}",
            message_type='file',
            file_url=f"/uploads/chat_files/{safe_filename}",
            file_name=filename,
            file_type=file_ext,
            file_size=len(content),
            created_at=datetime.utcnow()
        )
        db.add(new_message)
        await db.commit()
        await db.refresh(new_message)

        # Broadcast via WebSocket
        message_dict = model_to_dict(new_message)
        await manager.send_chat_message(room_id, message_dict)

        return {
            "success": True,
            "message_id": str(new_message.id),
            "file_url": new_message.file_url,
            "file_name": new_message.file_name,
            "file_size": len(content),
            "message": message_dict
        }

    except HTTPException:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/unread-count/total")
async def get_total_unread_count(
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get total unread message count across all chat rooms"""
    try:
        user_id = uuid.UUID(current_user["id"])
        count = await get_total_unread_chat_count(db, user_id)
        return {"total_unread_count": count or 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rooms/{room_id}/unread-count")
async def get_unread_count(
    room_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get unread message count for a chat room"""
    try:
        user_id = uuid.UUID(current_user["id"])
        
        room_result = await db.execute(
            select(ChatRoom).where(ChatRoom.id == uuid.UUID(room_id))
        )
        room = room_result.scalar_one_or_none()
        
        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")
        
        if user_id not in [room.developer_id, room.investor_id]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        result = await db.execute(
            select(func.count(ChatMessage.id))
            .where(
                and_(
                    ChatMessage.room_id == uuid.UUID(room_id),
                    ChatMessage.sender_id != user_id,
                    ChatMessage.read == False
                )
            )
        )
        count = result.scalar()
        
        return {"unread_count": count}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/messages/{message_id}/read")
async def mark_message_as_read(
    message_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Mark a chat message as read"""
    try:
        message_uuid = uuid.UUID(message_id)
        user_id = uuid.UUID(current_user["id"])
        
        # Get the message
        result = await db.execute(
            select(ChatMessage).where(ChatMessage.id == message_uuid)
        )
        message = result.scalar_one_or_none()
        
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        
        # Get the chat room to verify user is a participant
        room_result = await db.execute(
            select(ChatRoom).where(ChatRoom.id == message.room_id)
        )
        room = room_result.scalar_one_or_none()
        
        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")
        
        # Verify user is a participant
        if user_id not in [room.developer_id, room.investor_id]:
            raise HTTPException(status_code=403, detail="Not authorized")
        
        # Only allow marking if you're NOT the sender
        if message.sender_id == user_id:
            raise HTTPException(status_code=403, detail="Cannot mark own message as read")
        
        # Update read status
        message.read = True
        message.read_at = datetime.utcnow()
        await db.commit()
        
        total_unread = await get_total_unread_chat_count(db, user_id)
        await manager.send_ws_event(
            str(user_id),
            {"type": "chat_unread_count", "total_unread_count": total_unread}
        )
        
        return {
            "success": True,
            "message_id": str(message_id),
            "read_at": message.read_at.isoformat()
        }
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# Helper function
async def get_total_unread_chat_count(db: AsyncSession, user_id: uuid.UUID) -> int:
    """Return total unread chat messages for a user."""
    result = await db.execute(
        select(func.count(ChatMessage.id))
        .select_from(ChatMessage)
        .join(ChatRoom, ChatMessage.room_id == ChatRoom.id)
        .where(
            and_(
                ChatMessage.sender_id != user_id,
                ChatMessage.read == False,
                or_(
                    ChatRoom.developer_id == user_id,
                    ChatRoom.investor_id == user_id
                )
            )
        )
    )
    return result.scalar() or 0