# app/api/v1/websocket.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select, update, and_, or_, func
from datetime import datetime, timezone
import uuid
import json
import logging

from app.database import AsyncSessionLocal
from app.models.chat import ChatRoom, ChatMessage
from app.core.websocket_manager import manager
from app.services.moderation_service import ContentFilter

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/notifications/{user_id}")
async def websocket_notifications(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for live notifications"""
    await manager.connect(websocket, user_id)
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "user_id": user_id
        })
        
        while True:
            # Keep connection alive and listen for messages
            data = await websocket.receive_json()
            
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            
            elif data.get("type") == "mark_read":
                # Handle marking notification as read
                notif_id = data.get("notification_id")
                # Update notification in database
                async with AsyncSessionLocal() as db:
                    from app.models.notification import Notification
                    await db.execute(
                        update(Notification)
                        .where(Notification.id == uuid.UUID(notif_id))
                        .values(read=True, read_at=datetime.utcnow())
                    )
                    await db.commit()
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        manager.disconnect(websocket)


@router.websocket("/ws/chat/{room_id}/{user_id}")
async def websocket_chat(websocket: WebSocket, room_id: str, user_id: str):
    """WebSocket endpoint for chat rooms"""
    await manager.connect(websocket, user_id)
    
    try:
        user_uuid = uuid.UUID(user_id)
        room_uuid = uuid.UUID(room_id)
    except ValueError:
        await websocket.close(code=1008)
        manager.disconnect(websocket)
        return
    
    async with AsyncSessionLocal() as db:
        room_result = await db.execute(
            select(ChatRoom).where(ChatRoom.id == room_uuid)
        )
        room = room_result.scalar_one_or_none()
        
        if not room or user_uuid not in [room.developer_id, room.investor_id]:
            await websocket.close(code=1008)
            manager.disconnect(websocket)
            return
        
        # Mark messages as read
        await db.execute(
            update(ChatMessage)
            .where(
                and_(
                    ChatMessage.room_id == room_uuid,
                    ChatMessage.sender_id != user_uuid,
                    ChatMessage.read == False
                )
            )
            .values(read=True, read_at=func.now())
        )
        await db.commit()
        
        # Send unread count update
        from app.api.v1.chat import get_total_unread_chat_count
        total_unread = await get_total_unread_chat_count(db, user_uuid)
        await manager.send_ws_event(
            user_id,
            {"type": "chat_unread_count", "total_unread_count": total_unread}
        )
    
    await manager.join_chat_room(websocket, room_id)
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "room_id": room_id,
            "user_id": user_id
        })
        
        # Send notification about user joining
        await manager.broadcast_to_room(room_id, {
            "type": "user_joined",
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, exclude=websocket)
        
        while True:
            # Listen for incoming messages
            data = await websocket.receive_json()
            
            if data.get("type") == "message":
                message_text = data.get("message", "").strip()
                
                # Content filtering
                filter_result = ContentFilter.check_message(message_text)
                
                if not filter_result['is_safe']:
                    # Send error back to user
                    await websocket.send_json({
                        "type": "error",
                        "message": "Message blocked: Contains prohibited content",
                        "violations": [v['type'] for v in filter_result['violations']],
                        "detail": "Please do not share phone numbers, emails, or social media. All communication must happen through BiteBids."
                    })
                    logger.warning(f"Message blocked for user {user_id}: {[v['type'] for v in filter_result['violations']]}")
                    continue
                
                # Message is safe - broadcast to room
                message_data = {
                    "id": str(uuid.uuid4()),
                    "room_id": room_id,
                    "sender_id": user_id,
                    "message": message_text,
                    "message_type": data.get("message_type", "text"),
                    "read": False,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                await manager.send_chat_message(room_id, message_data)
            
            elif data.get("type") == "typing":
                await manager.broadcast_to_room(room_id, {
                    "type": "typing",
                    "user_id": user_id,
                    "is_typing": data.get("is_typing", True)
                }, exclude=websocket)
            
            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        await manager.broadcast_to_room(room_id, {
            "type": "user_left",
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, exclude=websocket)
        await manager.leave_chat_room(websocket, room_id)
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket chat error for room {room_id}, user {user_id}: {e}")
        await manager.leave_chat_room(websocket, room_id)
        manager.disconnect(websocket)