# app/core/websocket_manager.py
import logging
from typing import Dict, List, Optional
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for live notifications and chat"""
    
    def __init__(self):
        # Store active connections per user
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # Store chat room connections (room_id -> list of WebSockets)
        self.chat_rooms: Dict[str, List[WebSocket]] = {}
        # Map WebSocket to user_id for easy lookup
        self.websocket_to_user: Dict[WebSocket, str] = {}
        
    async def connect(self, websocket: WebSocket, user_id: str):
        """Connect a user's WebSocket"""
        await websocket.accept()
        
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        
        self.active_connections[user_id].append(websocket)
        self.websocket_to_user[websocket] = user_id
        logger.info(f"User {user_id} connected via WebSocket")
        
    def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket"""
        user_id = self.websocket_to_user.get(websocket)
        
        if user_id and user_id in self.active_connections:
            self.active_connections[user_id].remove(websocket)
            
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        
        if websocket in self.websocket_to_user:
            del self.websocket_to_user[websocket]
            
        # Remove from any chat rooms
        for room_id in list(self.chat_rooms.keys()):
            if websocket in self.chat_rooms[room_id]:
                self.chat_rooms[room_id].remove(websocket)
                if not self.chat_rooms[room_id]:
                    del self.chat_rooms[room_id]
        
        logger.info(f"User {user_id} disconnected from WebSocket")
    
    async def join_chat_room(self, websocket: WebSocket, room_id: str):
        """Join a chat room for a project"""
        if room_id not in self.chat_rooms:
            self.chat_rooms[room_id] = []
        
        if websocket not in self.chat_rooms[room_id]:
            self.chat_rooms[room_id].append(websocket)
        
        user_id = self.websocket_to_user.get(websocket)
        logger.info(f"User {user_id} joined chat room {room_id}")
    
    async def leave_chat_room(self, websocket: WebSocket, room_id: str):
        """Leave a chat room"""
        if room_id in self.chat_rooms and websocket in self.chat_rooms[room_id]:
            self.chat_rooms[room_id].remove(websocket)
            
            if not self.chat_rooms[room_id]:
                del self.chat_rooms[room_id]
        
        user_id = self.websocket_to_user.get(websocket)
        logger.info(f"User {user_id} left chat room {room_id}")
    
    def is_user_in_room(self, room_id: str, user_id: str) -> bool:
        """Check if a user has any active WebSocket in a chat room."""
        connections = self.chat_rooms.get(room_id, [])
        for connection in connections:
            if self.websocket_to_user.get(connection) == user_id:
                return True
        return False
    
    async def send_personal_notification(self, user_id: str, notification: dict):
        """Send notification to a specific user"""
        if user_id in self.active_connections:
            disconnected = []
            
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json({
                        "type": "notification",
                        "data": notification
                    })
                except Exception as e:
                    logger.error(f"Error sending notification to user {user_id}: {e}")
                    disconnected.append(connection)
            
            # Clean up disconnected connections
            for conn in disconnected:
                self.disconnect(conn)
    
    async def send_ws_event(self, user_id: str, event: dict):
        """Send a raw WebSocket event to a specific user"""
        if user_id in self.active_connections:
            disconnected = []
            
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json(event)
                except Exception as e:
                    logger.error(f"Error sending ws event to user {user_id}: {e}")
                    disconnected.append(connection)
            
            for conn in disconnected:
                self.disconnect(conn)
    
    async def send_chat_message(self, room_id: str, message: dict):
        """Send message to all users in a chat room"""
        if room_id in self.chat_rooms:
            disconnected = []
            
            for connection in self.chat_rooms[room_id]:
                try:
                    await connection.send_json({
                        "type": "chat_message",
                        "data": message
                    })
                except Exception as e:
                    logger.error(f"Error sending chat message to room {room_id}: {e}")
                    disconnected.append(connection)
            
            # Clean up disconnected connections
            for conn in disconnected:
                self.disconnect(conn)
    
    async def broadcast_to_all(self, message: dict):
        """Broadcast message to all connected users"""
        for user_id, connections in list(self.active_connections.items()):
            disconnected = []
            
            for connection in connections:
                try:
                    await connection.send_json({
                        "type": "broadcast",
                        "data": message
                    })
                except Exception as e:
                    logger.error(f"Error broadcasting to user {user_id}: {e}")
                    disconnected.append(connection)
            
            # Clean up disconnected connections
            for conn in disconnected:
                self.disconnect(conn)

    async def broadcast_to_room(self, room_id: str, message: dict, exclude: WebSocket = None):
        """Broadcast message to all users in a chat room, optionally excluding one WebSocket"""
        if room_id in self.chat_rooms:
            disconnected = []
            
            for connection in self.chat_rooms[room_id]:
                # Skip the excluded connection
                if exclude and connection == exclude:
                    continue
                    
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to room {room_id}: {e}")
                    disconnected.append(connection)
            
            # Clean up disconnected connections
            for conn in disconnected:
                self.disconnect(conn)


# Create a global instance
manager = ConnectionManager()