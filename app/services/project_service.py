# app/services/project_service.py
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm import joinedload

from app.models.project import Project
from app.models.user import User
from app.models.chat import ChatRoom
from app.core.constants import PROJECT_POSTING_FEE
from app.core.exceptions import NotFoundException, ForbiddenException, PaymentRequiredException
from app.utils.converters import model_to_dict


class ProjectService:
    """Service for project management operations"""
    
    @staticmethod
    async def create_project(
        project_data: Dict[str, Any],
        developer_id: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Create a new project"""
        
        try:
            user_uuid = uuid.UUID(developer_id)
        except ValueError:
            raise NotFoundException("Invalid developer ID")
        
        # Check user and credits
        user_result = await db.execute(select(User).where(User.id == user_uuid))
        user = user_result.scalar_one_or_none()
        
        if not user:
            raise NotFoundException("User not found")
        
        if user.role != "developer":
            raise ForbiddenException("Only developers can create projects")
        
        # Check credits
        if (user.posting_credits or 0) <= 0:
            raise PaymentRequiredException(
                detail={
                    "message": "You need to purchase posting credit before creating a project",
                    "code": "NO_POSTING_CREDITS",
                    "posting_fee": PROJECT_POSTING_FEE
                }
            )
        
        # Deduct credit
        user.posting_credits = (user.posting_credits or 0) - 1
        
        # Parse deadline
        deadline = project_data.get("deadline")
        if deadline:
            try:
                deadline = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
            except:
                deadline = datetime.utcnow() + timedelta(days=30)
        else:
            deadline = datetime.utcnow() + timedelta(days=30)
        
        # Create project
        new_project = Project(
            title=project_data.get("title"),
            status=project_data.get("status", "open"),
            description=project_data.get("description"),
            tech_stack=project_data.get("tech_stack", []),
            requirements=project_data.get("requirements", ""),
            budget=project_data.get("budget"),
            lowest_bid=project_data.get("lowest_bid", 0),
            deadline=deadline,
            location=project_data.get("location", "Remote"),
            developer_id=user_uuid,
            category=project_data.get("category", "Machine Learning"),
            images=project_data.get("images", [])
        )
        
        db.add(new_project)
        await db.commit()
        await db.refresh(new_project)
        
        return {
            **model_to_dict(new_project),
            "remaining_credits": user.posting_credits
        }
    
    @staticmethod
    async def get_projects(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 20,
        status: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get projects with pagination"""
        
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
    
    @staticmethod
    async def get_project_by_id(
        project_id: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Get a project by ID"""
        
        try:
            project_uuid = uuid.UUID(project_id)
        except ValueError:
            raise NotFoundException("Invalid project ID")
        
        project = await db.scalar(
            select(Project)
            .options(joinedload(Project.developer))
            .where(Project.id == project_uuid)
        )
        
        if not project:
            raise NotFoundException("Project not found")
        
        data = model_to_dict(project)
        if project.developer:
            data["developer"] = {
                "id": str(project.developer.id),
                "name": project.developer.name,
                "company": project.developer.company,
                "avatar": project.developer.avatar,
            }
        
        return data
    
    @staticmethod
    async def update_project(
        project_id: str,
        update_data: Dict[str, Any],
        user_id: str,
        user_role: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Update a project"""
        
        try:
            project_uuid = uuid.UUID(project_id)
        except ValueError:
            raise NotFoundException("Invalid project ID")
        
        project = await db.scalar(select(Project).where(Project.id == project_uuid))
        if not project:
            raise NotFoundException("Project not found")
        
        # Check permissions
        is_owner = str(project.developer_id) == user_id
        is_admin = user_role == "admin"
        
        if not is_owner and not is_admin:
            raise ForbiddenException("Only the project owner or admin can edit this project")
        
        # Apply updates
        if update_data.get("deadline"):
            try:
                update_data["deadline"] = datetime.fromisoformat(
                    update_data["deadline"].replace("Z", "+00:00")
                )
            except:
                pass
        
        for key, value in update_data.items():
            if value is not None:
                setattr(project, key, value)
        
        project.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(project)
        
        return model_to_dict(project)
    
    @staticmethod
    async def delete_project(
        project_id: str,
        user_id: str,
        user_role: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Delete a project"""
        
        try:
            project_uuid = uuid.UUID(project_id)
        except ValueError:
            raise NotFoundException("Invalid project ID")
        
        project = await db.scalar(select(Project).where(Project.id == project_uuid))
        if not project:
            raise NotFoundException("Project not found")
        
        if str(project.developer_id) != user_id and user_role != "admin":
            raise ForbiddenException("Unauthorized")
        
        project_title = project.title
        
        await db.delete(project)
        await db.commit()
        
        return {
            "message": "Project deleted",
            "project_id": project_id,
            "project_title": project_title
        }