# app/api/v1/notifications.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, func
import uuid
from datetime import datetime

from app.database import get_db
from app.models.notification import Notification
from app.core.dependencies import get_current_user
from app.utils.converters import model_to_dict

router = APIRouter(prefix="/notifications", tags=["Notifications"])


@router.get("/{user_id}")
async def get_user_notifications(
    user_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get all notifications for a specific user"""
    try:
        user_uuid = uuid.UUID(user_id)
        
        result = await db.execute(
            select(Notification)
            .where(Notification.user_id == user_uuid)
            .order_by(Notification.created_at.desc())
            .limit(50)
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
        unread_count = unread_result.scalar()
        
        notification_list = [
            {
                "id": str(notif.id),
                "type": notif.type,
                "title": notif.title,
                "message": notif.message,
                "link": notif.link,
                "metadata": notif.details or {},
                "read": notif.read,
                "read_at": notif.read_at.isoformat() if notif.read_at else None,
                "created_at": notif.created_at.isoformat() if notif.created_at else None,
            }
            for notif in notifications
        ]
        
        return {
            "success": True,
            "notifications": notification_list,
            "unread_count": unread_count or 0
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch notifications: {str(e)}")


@router.put("/{notification_id}/read")
async def mark_notification_read(
    notification_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Mark a specific notification as read"""
    try:
        notif_uuid = uuid.UUID(notification_id)
        
        result = await db.execute(
            update(Notification)
            .where(Notification.id == notif_uuid)
            .values(
                read=True,
                read_at=datetime.utcnow()
            )
            .returning(Notification.id)
        )
        
        await db.commit()
        
        updated = result.scalar_one_or_none()
        
        if not updated:
            raise HTTPException(status_code=404, detail="Notification not found")
        
        return {
            "success": True,
            "message": "Notification marked as read"
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid notification ID format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update notification: {str(e)}")


@router.put("/{user_id}/read-all")
async def mark_all_notifications_read(
    user_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Mark all notifications as read for a specific user"""
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
            .values(
                read=True,
                read_at=datetime.utcnow()
            )
        )
        
        await db.commit()
        
        return {
            "success": True,
            "message": "All notifications marked as read"
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update notifications: {str(e)}")


@router.delete("/{notification_id}")
async def delete_notification(
    notification_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a specific notification"""
    try:
        notif_uuid = uuid.UUID(notification_id)
        
        result = await db.execute(
            delete(Notification)
            .where(Notification.id == notif_uuid)
            .returning(Notification.id)
        )
        
        await db.commit()
        
        deleted = result.scalar_one_or_none()
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Notification not found")
        
        return {
            "success": True,
            "message": "Notification deleted successfully"
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid notification ID format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete notification: {str(e)}")