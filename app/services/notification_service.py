# app/services/notification_service.py
import uuid
import logging
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, update, delete
from datetime import datetime

from app.models.notification import Notification
from app.core.websocket_manager import manager
from app.utils.converters import model_to_dict

logger = logging.getLogger(__name__)


class NotificationService:
    """Service for managing notifications"""
    
    @staticmethod
    async def send_notification_to_user(
        user_id: str,
        notification_data: Dict[str, Any],
        db: AsyncSession
    ) -> Optional[Dict]:
        """Send notification to a user (both DB and WebSocket)"""
        try:
            # Save notification to database
            notification = Notification(
                user_id=uuid.UUID(user_id),
                type=notification_data.get("type", "general"),
                title=notification_data.get("title"),
                message=notification_data.get("message"),
                link=notification_data.get("link"),
                details=notification_data.get("details")
            )
            
            db.add(notification)
            await db.commit()
            await db.refresh(notification)
            
            # Send via WebSocket if user is connected
            notif_dict = model_to_dict(notification)
            await manager.send_personal_notification(user_id, notif_dict)
            
            return notif_dict
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return None

    @staticmethod
    async def get_user_notifications(
        user_id: str,
        db: AsyncSession,
        limit: int = 50
    ) -> Dict:
        """Get all notifications for a user"""
        try:
            user_uuid = uuid.UUID(user_id)
            
            result = await db.execute(
                select(Notification)
                .where(Notification.user_id == user_uuid)
                .order_by(Notification.created_at.desc())
                .limit(limit)
            )
            notifications = result.scalars().all()
            
            unread_result = await db.execute(
                select(func.count(Notification.id))
                .where(
                    and_(
                        Notification.user_id == user_uuid,
                        Notification.read == False
                    )
                )
            )
            unread_count = unread_result.scalar() or 0
            
            return {
                "notifications": [model_to_dict(n) for n in notifications],
                "unread_count": unread_count
            }
            
        except Exception as e:
            logger.error(f"Error fetching notifications: {str(e)}")
            return {"notifications": [], "unread_count": 0}

    @staticmethod
    async def mark_as_read(notification_id: str, db: AsyncSession) -> bool:
        """Mark a notification as read"""
        try:
            notif_uuid = uuid.UUID(notification_id)
            
            result = await db.execute(
                update(Notification)
                .where(Notification.id == notif_uuid)
                .values(read=True, read_at=datetime.utcnow())
                .returning(Notification.id)
            )
            await db.commit()
            
            return result.scalar_one_or_none() is not None
            
        except Exception as e:
            logger.error(f"Error marking notification as read: {str(e)}")
            return False

    @staticmethod
    async def mark_all_as_read(user_id: str, db: AsyncSession) -> bool:
        """Mark all notifications as read for a user"""
        try:
            user_uuid = uuid.UUID(user_id)
            
            await db.execute(
                update(Notification)
                .where(
                    and_(
                        Notification.user_id == user_uuid,
                        Notification.read == False
                    )
                )
                .values(read=True, read_at=datetime.utcnow())
            )
            await db.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error marking all notifications as read: {str(e)}")
            return False

    @staticmethod
    async def delete_notification(notification_id: str, db: AsyncSession) -> bool:
        """Delete a notification"""
        try:
            notif_uuid = uuid.UUID(notification_id)
            
            result = await db.execute(
                delete(Notification)
                .where(Notification.id == notif_uuid)
                .returning(Notification.id)
            )
            await db.commit()
            
            return result.scalar_one_or_none() is not None
            
        except Exception as e:
            logger.error(f"Error deleting notification: {str(e)}")
            return False

    @staticmethod
    async def get_total_unread_count(user_id: str, db: AsyncSession) -> int:
        """Get total unread notifications for a user"""
        try:
            user_uuid = uuid.UUID(user_id)
            
            result = await db.execute(
                select(func.count(Notification.id))
                .where(
                    and_(
                        Notification.user_id == user_uuid,
                        Notification.read == False
                    )
                )
            )
            return result.scalar() or 0
            
        except Exception as e:
            logger.error(f"Error getting unread count: {str(e)}")
            return 0