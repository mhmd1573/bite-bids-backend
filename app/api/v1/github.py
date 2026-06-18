# app/api/v1/github.py
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
import uuid
import requests
import base64
import os
from datetime import datetime
from typing import Optional, List, Dict

from app.database import get_db
from app.models.chat import ChatRoom
from app.models.project import Project
from app.models.user import User
from app.models.github import ProjectGithubRepo
from app.models.payment import DeveloperPayout
from app.core.dependencies import get_current_user, get_current_admin
from app.core.security import encrypt_token, decrypt_token
from app.core.exceptions import NotFoundException, ForbiddenException
from app.config import settings

router = APIRouter(prefix="/github", tags=["GitHub"])


@router.post("/chat/rooms/{room_id}/submit-github-repo")
async def submit_github_repository(
    room_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Submit GitHub repository URL for project (Developer only)"""
    try:
        body = await request.json()
        repo_url = (body.get("repo_url") or "").strip()
        access_token = (body.get("access_token") or "").strip()

        if not repo_url:
            raise HTTPException(status_code=400, detail="Repository URL is required")

        # Validate GitHub URL format
        try:
            parsed = parse_github_url(repo_url)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid GitHub repository URL")

        # Check if repo is private
        is_private = await check_if_repo_is_private(
            parsed["owner"],
            parsed["repo"],
            access_token
        )

        # If private and no token provided -> error
        if is_private and not access_token:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "PRIVATE_REPO_TOKEN_REQUIRED",
                    "message": (
                        "This repository is private. Please provide a GitHub "
                        "Personal Access Token with 'repo' scope."
                    )
                }
            )

        # Verify room exists
        room_query = await db.execute(
            select(ChatRoom).where(ChatRoom.id == uuid.UUID(room_id))
        )
        room = room_query.scalar_one_or_none()

        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")

        # Verify developer permissions
        user_id = uuid.UUID(current_user["id"])
        if user_id != room.developer_id:
            raise HTTPException(
                status_code=403,
                detail="Only the developer can submit repository"
            )

        # Encrypt token if provided
        encrypted_token = None
        if access_token:
            try:
                encrypted_token = encrypt_token(access_token)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to encrypt access token"
                )

        # Check if repository already exists
        existing_query = await db.execute(
            select(ProjectGithubRepo).where(
                ProjectGithubRepo.room_id == uuid.UUID(room_id)
            )
        )
        existing = existing_query.scalar_one_or_none()

        if existing:
            existing.repo_url = repo_url
            existing.is_private = is_private
            existing.encrypted_access_token = encrypted_token
            existing.submitted_at = datetime.utcnow()
        else:
            github_repo = ProjectGithubRepo(
                id=uuid.uuid4(),
                room_id=uuid.UUID(room_id),
                repo_url=repo_url,
                is_private=is_private,
                encrypted_access_token=encrypted_token,
                submitted_by=user_id,
                submitted_at=datetime.utcnow()
            )
            db.add(github_repo)

        await db.commit()

        return {
            "success": True,
            "message": "Repository submitted successfully",
            "repo_url": repo_url,
            "is_private": is_private
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/chat/rooms/{room_id}/github-repo")
async def get_github_repository(
    room_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get GitHub repository for a room"""
    try:
        # Verify room exists
        room_query = await db.execute(
            select(ChatRoom).where(ChatRoom.id == uuid.UUID(room_id))
        )
        room = room_query.scalar_one_or_none()

        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")

        user_id = uuid.UUID(current_user["id"])
        user_role = current_user.get("role", "")

        # Allow admins to access any room
        if user_role != "admin":
            if user_id not in [room.developer_id, room.investor_id]:
                raise HTTPException(status_code=403, detail="Access denied")

        # Get repository
        repo_query = await db.execute(
            select(ProjectGithubRepo).where(
                ProjectGithubRepo.room_id == uuid.UUID(room_id)
            )
        )
        repo = repo_query.scalar_one_or_none()

        if not repo:
            return {
                "exists": False,
                "message": "No repository submitted yet"
            }

        return {
            "exists": True,
            "repo_url": repo.repo_url,
            "submitted_at": repo.submitted_at.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/repo-structure")
async def get_repo_structure(
    request: Request,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get GitHub repository structure"""
    try:
        body = await request.json()
        repo_url = body.get('repo_url')

        if not repo_url:
            raise HTTPException(status_code=400, detail="Repository URL is required")

        user_role = current_user.get("role", "")
        user_id = uuid.UUID(current_user["id"])

        # Parse GitHub URL
        try:
            parsed = parse_github_url(repo_url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Check if we have stored token for this repo
        repo_query = await db.execute(
            select(ProjectGithubRepo).where(
                ProjectGithubRepo.repo_url == repo_url
            )
        )
        repo_record = repo_query.scalar_one_or_none()

        # Admins can access any repo
        if repo_record and user_role != "admin":
            room_query = await db.execute(
                select(ChatRoom).where(ChatRoom.id == repo_record.room_id)
            )
            room = room_query.scalar_one_or_none()

            if room and user_id not in [room.developer_id, room.investor_id]:
                raise HTTPException(status_code=403, detail="Access denied")

        encrypted_token = repo_record.encrypted_access_token if repo_record else None

        # Fetch tree
        tree = fetch_github_tree(parsed['owner'], parsed['repo'], encrypted_token=encrypted_token)

        if tree is None:
            raise HTTPException(status_code=404, detail="Could not fetch repository structure. Repository may be private or rate limit exceeded.")

        return {
            "success": True,
            "owner": parsed['owner'],
            "repo": parsed['repo'],
            "tree": tree
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/file-content")
async def get_file_content(
    request: Request,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get file content from GitHub repository"""
    try:
        body = await request.json()
        repo_url = body.get('repo_url')
        file_path = body.get('file_path')

        if not repo_url or not file_path:
            raise HTTPException(status_code=400, detail="Repository URL and file path are required")

        user_role = current_user.get("role", "")
        user_id = uuid.UUID(current_user["id"])

        # Parse GitHub URL
        try:
            parsed = parse_github_url(repo_url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Check if we have stored token for this repo
        repo_query = await db.execute(
            select(ProjectGithubRepo).where(
                ProjectGithubRepo.repo_url == repo_url
            )
        )
        repo_record = repo_query.scalar_one_or_none()

        # Admins can access any file
        if repo_record and user_role != "admin":
            room_query = await db.execute(
                select(ChatRoom).where(ChatRoom.id == repo_record.room_id)
            )
            room = room_query.scalar_one_or_none()

            if room and user_id not in [room.developer_id, room.investor_id]:
                raise HTTPException(status_code=403, detail="Access denied")

        encrypted_token = repo_record.encrypted_access_token if repo_record else None

        # Fetch file content
        content = fetch_github_file_content(
            parsed['owner'],
            parsed['repo'],
            file_path,
            encrypted_token=encrypted_token
        )

        return {
            "success": True,
            "content": content,
            "file_path": file_path
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download-repo/{room_id}")
async def download_github_repo(
    room_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Download GitHub repository as ZIP file"""
    try:
        user_role = current_user.get("role", "")
        user_id = uuid.UUID(current_user["id"])
        room_uuid = uuid.UUID(room_id)

        # Get the chat room
        room_query = await db.execute(
            select(ChatRoom).where(ChatRoom.id == room_uuid)
        )
        room = room_query.scalar_one_or_none()

        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")

        # Authorization check (admin bypass)
        if user_role != "admin":
            if user_id not in [room.developer_id, room.investor_id]:
                raise HTTPException(status_code=403, detail="Access denied to this chat room")

        # Get the project and check status
        project_query = await db.execute(
            select(Project).where(Project.id == room.project_id)
        )
        project = project_query.scalar_one_or_none()

        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Check if project is completed
        is_investor = user_id == room.investor_id

        if user_role != "admin":
            # Check if there's a completed payout for this investor
            payout_query = await db.execute(
                select(DeveloperPayout).where(
                    DeveloperPayout.project_id == project.id,
                    DeveloperPayout.investor_id == user_id
                )
            )
            payout = payout_query.scalar_one_or_none()

            if is_investor and not payout:
                raise HTTPException(
                    status_code=403,
                    detail="You must confirm the project before downloading. Please click 'Confirm' first."
                )

        # Get GitHub repo for this room
        repo_query = await db.execute(
            select(ProjectGithubRepo).where(ProjectGithubRepo.room_id == room_uuid)
        )
        repo_record = repo_query.scalar_one_or_none()

        if not repo_record or not repo_record.repo_url:
            raise HTTPException(status_code=404, detail="No GitHub repository found for this project")

        # Parse GitHub URL
        try:
            parsed = parse_github_url(repo_record.repo_url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid GitHub URL: {str(e)}")

        owner = parsed['owner']
        repo = parsed['repo']

        # Prepare headers for GitHub API
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "BiteBids-Platform"
        }

        # Add auth token if available
        if repo_record.encrypted_access_token:
            try:
                token = decrypt_token(repo_record.encrypted_access_token)
                headers["Authorization"] = f"Bearer {token}"
            except Exception as e:
                pass

        # Get default branch
        branch = "main"
        try:
            repo_info_url = f"https://api.github.com/repos/{owner}/{repo}"
            repo_response = requests.get(repo_info_url, headers=headers, timeout=10)
            if repo_response.status_code == 200:
                repo_info = repo_response.json()
                branch = repo_info.get("default_branch", "main")
        except:
            pass

        # Download ZIP from GitHub
        zip_url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"

        response = requests.get(zip_url, headers=headers, stream=True, timeout=180)

        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="Repository not found or access denied")
        elif response.status_code == 401:
            raise HTTPException(status_code=401, detail="Authentication required for this private repository")
        elif response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"GitHub API error: {response.text}")

        # Get filename
        content_disposition = response.headers.get('Content-Disposition', '')
        if 'filename=' in content_disposition:
            filename = content_disposition.split('filename=')[-1].strip('"')
        else:
            filename = f"{repo}-{branch}.zip"

        # Stream the response
        response_headers = {
            "Content-Disposition": f'attachment; filename="{filename}"',
        }

        content_length = response.headers.get('Content-Length')
        if content_length and content_length.strip():
            response_headers["Content-Length"] = content_length

        return StreamingResponse(
            iter(response.iter_content(chunk_size=65536)),
            media_type="application/zip",
            headers=response_headers
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download repository: {str(e)}")


# Helper functions
def parse_github_url(repo_url: str) -> dict:
    """Parse GitHub repository URL"""
    import re
    pattern = r'github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$'
    match = re.search(pattern, repo_url)

    if not match:
        raise ValueError("Invalid GitHub repository URL")

    return {
        'owner': match.group(1),
        'repo': match.group(2)
    }


async def check_if_repo_is_private(owner: str, repo: str, access_token: str = None) -> bool:
    """Check if a GitHub repository is private"""
    try:
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "BiteBids-App"
        }

        token = access_token or settings.GITHUB_TOKEN
        if token:
            headers["Authorization"] = f"token {token}"

        url = f"https://api.github.com/repos/{owner}/{repo}"
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 404:
            return True

        if response.status_code == 200:
            data = response.json()
            return data.get('private', False)

        return False
    except Exception:
        return False


def get_github_headers() -> dict:
    """Get headers for GitHub API requests"""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "BiteBids-App"
    }

    if settings.GITHUB_TOKEN:
        headers["Authorization"] = f"token {settings.GITHUB_TOKEN}"

    return headers


def build_tree_structure(items: List[Dict]) -> List[Dict]:
    """Build hierarchical tree structure from GitHub API flat list"""
    files_dict = {}
    folders_dict = {}

    for item in items:
        path = item['path']

        if item['type'] == 'blob':
            files_dict[path] = {
                'path': path,
                'type': 'blob',
                'size': item.get('size', 0)
            }
        elif item['type'] == 'tree':
            if path not in folders_dict:
                folders_dict[path] = {
                    'path': path,
                    'type': 'tree',
                    'size': item.get('size', 0),
                    'children': []
                }

    # Build hierarchy
    root_items = []

    for path, folder in folders_dict.items():
        folder_path_prefix = path + '/'

        # Add child files
        for file_path, file_data in files_dict.items():
            if file_path.startswith(folder_path_prefix):
                relative_path = file_path[len(folder_path_prefix):]
                if '/' not in relative_path:
                    folder['children'].append(file_data)

        # Add child folders
        for child_path, child_folder in folders_dict.items():
            if child_path.startswith(folder_path_prefix) and child_path != path:
                relative_path = child_path[len(folder_path_prefix):]
                if '/' not in relative_path:
                    folder['children'].append(child_folder)

        # Root level folder
        if '/' not in path:
            root_items.append(folder)

    # Add root level files
    for file_path, file_data in files_dict.items():
        if '/' not in file_path:
            root_items.append(file_data)

    return root_items


def fetch_github_tree(owner: str, repo: str, branch: str = "main", encrypted_token: str = None) -> Optional[List[Dict]]:
    """Fetch repository tree structure from GitHub API"""
    try:
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        headers = get_github_headers()

        if encrypted_token:
            try:
                token = decrypt_token(encrypted_token)
                headers["Authorization"] = f"token {token}"
            except:
                return None

        response = requests.get(url, headers=headers, timeout=15)

        # Try master branch if main doesn't exist
        if response.status_code == 404:
            url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/master?recursive=1"
            response = requests.get(url, headers=headers, timeout=15)

        if response.status_code == 401:
            return None

        if response.status_code == 404:
            return None

        if response.status_code == 403:
            return None

        response.raise_for_status()
        data = response.json()

        items = data.get('tree', [])

        if not items:
            return []

        # Build hierarchical structure
        tree_structure = build_tree_structure(items)

        return tree_structure

    except Exception:
        return None


def fetch_github_file_content(owner: str, repo: str, file_path: str, branch: str = "main", encrypted_token: str = None) -> str:
    """Fetch file content from GitHub repository"""
    try:
        encoded_path = requests.utils.quote(file_path)
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{encoded_path}?ref={branch}"
        headers = get_github_headers()

        if encrypted_token:
            try:
                token = decrypt_token(encrypted_token)
                headers["Authorization"] = f"token {token}"
            except:
                return "Error: Failed to decrypt access token"

        response = requests.get(url, headers=headers, timeout=15)

        # Try master branch if main doesn't exist
        if response.status_code == 404:
            url = f"https://api.github.com/repos/{owner}/{repo}/contents/{encoded_path}?ref=master"
            response = requests.get(url, headers=headers, timeout=15)

        if response.status_code == 401:
            return "Error: Invalid or expired access token."

        if response.status_code == 404:
            return "Error: File not found or no access to this repository."

        response.raise_for_status()
        data = response.json()

        # Decode content from base64
        content = base64.b64decode(data['content']).decode('utf-8')

        return content

    except Exception:
        return "Error loading file"