# app/api/v1/projects.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.orm import joinedload
import uuid
from datetime import datetime, timedelta

from app.database import get_db
from app.models.user import User
from app.models.project import Project
from app.schemas.project import ProjectCreate, ProjectUpdate
from app.core.dependencies import get_current_user, get_current_admin
from app.core.constants import PROJECT_POSTING_FEE
from app.core.exceptions import PaymentRequiredException, NotFoundException, ForbiddenException
from app.utils.converters import model_to_dict
from app.services.email_service import EmailService

router = APIRouter(prefix="/projects", tags=["Projects"])


@router.get("")
async def get_projects(
    skip: int = 0,
    limit: int = 20,
    status: str = None,
    db: AsyncSession = Depends(get_db)
):
    """Get projects with developer info"""
    
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


@router.post("")
async def create_project(
    project_data: ProjectCreate,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new project"""
    
    if current_user["role"] != "developer":
        raise ForbiddenException("Only developers can create projects")
    
    user_id = uuid.UUID(current_user["id"])
    
    # Check posting credits
    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalar_one_or_none()
    
    if not user:
        raise NotFoundException("User not found")
    
    # Check if user has credits
    if (user.posting_credits or 0) <= 0:
        raise PaymentRequiredException(
            detail={
                "message": "You need to purchase posting credit before creating a project",
                "code": "NO_POSTING_CREDITS",
                "posting_fee": PROJECT_POSTING_FEE
            }
        )
    
    # Deduct 1 credit
    user.posting_credits = (user.posting_credits or 0) - 1
    
    # Parse deadline
    deadline = None
    if project_data.deadline:
        try:
            deadline = datetime.fromisoformat(project_data.deadline.replace("Z", "+00:00"))
        except:
            deadline = datetime.utcnow() + timedelta(days=30)
    
    # Create project
    new_project = Project(
        title=project_data.title,
        status=project_data.status,
        description=project_data.description,
        tech_stack=project_data.tech_stack,
        requirements=project_data.requirements,
        budget=project_data.budget,
        lowest_bid=project_data.lowest_bid,
        deadline=deadline,
        location=project_data.location or "Remote",
        developer_id=current_user["id"],
        category=project_data.category or "Machine Learning",
        images=project_data.images if hasattr(project_data, 'images') else []
    )
    
    db.add(new_project)
    await db.commit()
    await db.refresh(new_project)
    
    # Send admin notification
    background_tasks.add_task(
        EmailService.send_admin_project_notification,
        action="created",
        project_id=str(new_project.id),
        project_title=new_project.title,
        developer_name=current_user.get('name', 'Unknown'),
        developer_email=current_user.get('email', 'Unknown'),
        project_data={
            'category': new_project.category,
            'budget': float(new_project.budget),
            'status': new_project.status,
            'location': new_project.location,
            'tech_stack': new_project.tech_stack
        }
    )
    
    return {
        **model_to_dict(new_project),
        "remaining_credits": user.posting_credits
    }


@router.get("/check-posting-status")
async def check_posting_status(
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Check if user has available posting credits"""
    
    user_id = uuid.UUID(current_user["id"])
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise NotFoundException("User not found")
    
    if user.role != "developer":
        raise ForbiddenException("Only developers can post projects")
    
    credits = user.posting_credits or 0
    has_credit = credits > 0
    
    return {
        "success": True,
        "has_credit": has_credit,
        "credits": credits,
        "needs_payment": not has_credit,
        "posting_fee": PROJECT_POSTING_FEE
    }


@router.get("/{project_id}")
async def get_project(
    project_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific project"""
    
    project = await db.scalar(
        select(Project)
        .options(joinedload(Project.developer))
        .where(Project.id == uuid.UUID(project_id))
    )
    
    if not project:
        raise NotFoundException("Project not found")
    
    data = model_to_dict(project)
    data["developer"] = {
        "id": str(project.developer.id),
        "name": project.developer.name,
        "company": project.developer.company,
        "avatar": project.developer.avatar,
    }
    return data


@router.put("/{project_id}")
async def update_project(
    project_id: str,
    project_data: ProjectUpdate,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update a project"""
    
    project = await db.scalar(select(Project).where(Project.id == uuid.UUID(project_id)))
    if not project:
        raise NotFoundException("Project not found")
    
    # Check permissions
    is_owner = str(project.developer_id) == str(current_user["id"])
    is_admin = current_user.get("role") == "admin"
    
    if not is_owner and not is_admin:
        raise ForbiddenException("Only the project owner or admin can edit this project")
    
    # Track changes if admin is editing
    changes = {}
    admin_editing_others_project = is_admin and not is_owner
    
    if admin_editing_others_project:
        for field, value in project_data.dict(exclude_unset=True).items():
            if field == "deadline" and value:
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            
            old_value = getattr(project, field, None)
            
            if old_value != value:
                if field == 'images':
                    old_images = old_value or []
                    new_images = value or []
                    if len(old_images) != len(new_images):
                        changes['images'] = {
                            'old': old_images,
                            'new': new_images,
                            'reason': 'Image(s) removed for policy compliance'
                        }
                elif field == 'tech_stack':
                    changes['tech_stack'] = {'old': old_value or [], 'new': value or []}
                elif field == 'budget':
                    changes['budget'] = {'old': float(old_value) if old_value else 0, 'new': float(value) if value else 0}
                elif field == 'description':
                    changes['description'] = {'old': old_value or '', 'new': value or '', 'reason': 'Content updated for policy compliance'}
                elif field == 'title':
                    changes['title'] = {'old': old_value or '', 'new': value or ''}
                elif field == 'status':
                    changes['status'] = {'old': old_value or '', 'new': value or ''}
                elif field == 'category':
                    changes['category'] = {'old': old_value or '', 'new': value or ''}
                elif field == 'location':
                    changes['location'] = {'old': old_value or '', 'new': value or ''}
    
    # Apply updates
    for field, value in project_data.dict(exclude_unset=True).items():
        if field == "deadline" and value:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        setattr(project, field, value)
    
    await db.commit()
    await db.refresh(project)
    
    # Send admin notification
    background_tasks.add_task(
        send_admin_project_notification,
        action="updated",
        project_id=str(project.id),
        project_title=project.title,
        developer_name=current_user.get('name', 'Unknown'),
        developer_email=current_user.get('email', 'Unknown'),
        project_data={
            'category': project.category,
            'budget': float(project.budget),
            'status': project.status,
            'location': project.location,
            'tech_stack': project.tech_stack
        }
    )
    
    # Send email to developer if admin edited
    if admin_editing_others_project and changes:
        developer = await db.scalar(select(User).where(User.id == project.developer_id))
        
        if developer and developer.email:
            background_tasks.add_task(
                EmailService.send_developer_edit_notification,
                developer_email=developer.email,
                developer_name=developer.name,
                project_title=project.title,
                project_id=str(project.id),
                admin_name=current_user.get('name', 'Admin'),
                changes=changes
            )
    
    return model_to_dict(project)


@router.delete("/{project_id}")
async def delete_project(
    project_id: str,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a project"""
    
    project = await db.scalar(select(Project).where(Project.id == uuid.UUID(project_id)))
    
    if not project:
        raise NotFoundException("Project not found")
    
    if str(project.developer_id) != str(current_user["id"]) and current_user["role"] != "admin":
        raise ForbiddenException("Unauthorized")
    
    # Capture data before delete
    project_title = project.title
    project_category = project.category
    project_budget = project.budget
    
    await db.delete(project)
    await db.commit()
    
    # Send admin notification
    background_tasks.add_task(
        send_admin_project_notification,
        action="deleted",
        project_id=project_id,
        project_title=project_title,
        developer_name=current_user.get('name', 'Unknown'),
        developer_email=current_user.get('email', 'Unknown'),
        project_data={
            'category': project_category,
            'budget': project_budget,
            'status': 'deleted',
            'location': 'N/A',
            'tech_stack': []
        }
    )
    
    return {"message": "Project deleted", "project_id": project_id}


@router.get("/developer/{developer_id}")
async def get_developer_projects(
    developer_id: str,
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """Get all projects for a developer"""
    
    result = await db.execute(
        select(Project)
        .where(Project.developer_id == uuid.UUID(developer_id))
        .order_by(Project.created_at.desc())
        .offset(skip).limit(limit)
    )
    projects = result.scalars().all()
    
    # Check for chat rooms for each project
    projects_with_chat_info = []
    for project in projects:
        project_dict = model_to_dict(project)
        
        # Check if project has any chat rooms
        from app.models.chat import ChatRoom
        chat_room_result = await db.execute(
            select(ChatRoom).where(ChatRoom.project_id == project.id).limit(1)
        )
        has_chat_room = chat_room_result.scalar() is not None
        project_dict["has_chat_room"] = has_chat_room
        
        projects_with_chat_info.append(project_dict)
    
    return {
        "projects": projects_with_chat_info,
        "total": len(projects),
        "skip": skip,
        "limit": limit
    }