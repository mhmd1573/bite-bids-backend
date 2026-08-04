# app/api/v1/uploads.py
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
import uuid
import os
import base64
import requests
from datetime import datetime
from PIL import Image
import boto3
from botocore.config import Config
import zipfile
import io
import tempfile

from app.database import get_db
from app.models.chat import ChatRoom
from app.models.project import Project
from app.models.user import User
from app.models.upload import ProjectUpload
from app.models.payment import DeveloperPayout
from app.core.dependencies import get_current_user, get_current_admin
from app.core.exceptions import NotFoundException, ForbiddenException
from app.core.constants import ALLOWED_IMAGE_EXTENSIONS, MAX_UPLOAD_SIZE
from app.config import settings

router = APIRouter(prefix="/upload", tags=["Uploads"])


# Initialize R2 client
r2_client = None
if settings.R2_ACCOUNT_ID and settings.R2_ACCESS_KEY_ID and settings.R2_SECRET_ACCESS_KEY:
    r2_client = boto3.client(
        's3',
        endpoint_url=f'https://{settings.R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
        aws_access_key_id=settings.R2_ACCESS_KEY_ID,
        aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
        config=Config(signature_version='s3v4'),
        region_name='auto'
    )


@router.post("/presigned-url/{room_id}")
async def get_upload_presigned_url(
    room_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Generate a presigned URL for direct upload to R2"""
    try:
        if not r2_client:
            raise HTTPException(status_code=503, detail="Cloud storage not configured")

        user_id = uuid.UUID(current_user["id"])
        room_uuid = uuid.UUID(room_id)

        # Get chat room
        room_query = await db.execute(
            select(ChatRoom).where(ChatRoom.id == room_uuid)
        )
        room = room_query.scalar_one_or_none()

        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")

        # Only developer can upload
        if user_id != room.developer_id:
            raise HTTPException(status_code=403, detail="Only the developer can upload project files")

        # Get request body
        body = await request.json()
        file_name = body.get("file_name", "project.zip")
        file_size = body.get("file_size", 0)
        content_type = body.get("content_type", "application/zip")

        # Validate file size (max 5GB)
        max_size = 5 * 1024 * 1024 * 1024
        if file_size > max_size:
            raise HTTPException(status_code=400, detail="File size exceeds 5GB limit")

        # Generate unique file key
        file_key = f"projects/{room_id}/{uuid.uuid4()}/{file_name}"

        # Generate presigned URL for upload
        presigned_url = r2_client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': settings.R2_BUCKET_NAME,
                'Key': file_key,
                'ContentType': content_type,
            },
            ExpiresIn=3600
        )

        return {
            "presigned_url": presigned_url,
            "file_key": file_key,
            "expires_in": 3600
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/complete/{room_id}")
async def complete_upload(
    room_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Complete the upload process by saving file tree metadata"""
    try:
        user_id = uuid.UUID(current_user["id"])
        room_uuid = uuid.UUID(room_id)

        # Get chat room
        room_query = await db.execute(
            select(ChatRoom).where(ChatRoom.id == room_uuid)
        )
        room = room_query.scalar_one_or_none()

        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")

        # Only developer can complete upload
        if user_id != room.developer_id:
            raise HTTPException(status_code=403, detail="Only the developer can upload project files")

        body = await request.json()
        file_key = body.get("file_key")
        file_name = body.get("file_name")
        file_size = body.get("file_size")
        file_tree = body.get("file_tree")

        if not all([file_key, file_name, file_tree]):
            raise HTTPException(status_code=400, detail="Missing required fields")

        # Check if upload already exists
        existing_query = await db.execute(
            select(ProjectUpload).where(ProjectUpload.room_id == room_uuid)
        )
        existing = existing_query.scalar_one_or_none()

        if existing:
            existing.file_key = file_key
            existing.file_name = file_name
            existing.file_size = file_size
            existing.file_tree = file_tree
            existing.uploaded_at = datetime.utcnow()
        else:
            upload = ProjectUpload(
                room_id=room_uuid,
                file_key=file_key,
                file_name=file_name,
                file_size=file_size,
                file_tree=file_tree,
                uploaded_by=user_id,
                status='pending'
            )
            db.add(upload)

        await db.commit()

        return {
            "success": True,
            "message": "Project uploaded successfully",
            "file_name": file_name,
            "file_size": file_size
        }

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info/{room_id}")
async def get_upload_info(
    room_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get upload information including file tree for preview"""
    try:
        user_id = uuid.UUID(current_user["id"])
        user_role = current_user.get("role", "")
        room_uuid = uuid.UUID(room_id)

        # Get chat room
        room_query = await db.execute(
            select(ChatRoom).where(ChatRoom.id == room_uuid)
        )
        room = room_query.scalar_one_or_none()

        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")

        # Authorization check
        if user_role != "admin" and user_id not in [room.developer_id, room.investor_id]:
            raise HTTPException(status_code=403, detail="Access denied")

        # Get upload record
        upload_query = await db.execute(
            select(ProjectUpload).where(ProjectUpload.room_id == room_uuid)
        )
        upload = upload_query.scalar_one_or_none()

        if not upload:
            return {"upload": None}

        return {
            "upload": {
                "id": str(upload.id),
                "file_name": upload.file_name,
                "file_size": upload.file_size,
                "file_tree": upload.file_tree,
                "uploaded_at": upload.uploaded_at.isoformat() if upload.uploaded_at else None,
                "status": upload.status
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/file-content/{room_id}")
async def get_uploaded_file_content(
    room_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Read a specific file's content from the uploaded ZIP in R2 cloud storage.
    This is used by the "Cloud View" feature so investors can browse files
    without downloading the entire project.
    """
    try:
        if not r2_client:
            raise HTTPException(status_code=503, detail="Cloud storage not configured")

        user_id = uuid.UUID(current_user["id"])
        user_role = current_user.get("role", "")
        room_uuid = uuid.UUID(room_id)

        # Get request body
        body = await request.json()
        file_path = body.get("file_path")
        if not file_path:
            raise HTTPException(status_code=400, detail="file_path is required")

        # Get chat room
        room_query = await db.execute(
            select(ChatRoom).where(ChatRoom.id == room_uuid)
        )
        room = room_query.scalar_one_or_none()

        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")

        # Authorization check
        if user_role != "admin" and user_id not in [room.developer_id, room.investor_id]:
            raise HTTPException(status_code=403, detail="Access denied")

        # Get upload record
        upload_query = await db.execute(
            select(ProjectUpload).where(ProjectUpload.room_id == room_uuid)
        )
        upload = upload_query.scalar_one_or_none()

        if not upload:
            raise HTTPException(status_code=404, detail="No uploaded project found")

        # Generate a presigned URL to download the ZIP
        presigned_url = r2_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': settings.R2_BUCKET_NAME,
                'Key': upload.file_key,
            },
            ExpiresIn=300  # 5 minutes
        )

        # Download the ZIP file from R2
        zip_response = requests.get(presigned_url, timeout=30)
        if zip_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to download project archive from cloud storage")

        # Read the ZIP file from memory
        try:
            zip_data = io.BytesIO(zip_response.content)
            with zipfile.ZipFile(zip_data, 'r') as zf:
                # List all files in the ZIP to find the matching one
                all_files = zf.namelist()
                
                # Find the file that matches the requested path
                # The file_path comes from the file tree which uses relative paths
                # ZIP entries may have the full webkitRelativePath or just the relative path
                matched_file = None
                for zip_entry in all_files:
                    # Check if the entry ends with the requested file_path
                    if zip_entry.endswith(file_path) or zip_entry == file_path:
                        matched_file = zip_entry
                        break
                    # Also check if the entry contains the file_path somewhere
                    if f"/{file_path}" in zip_entry or zip_entry == file_path:
                        matched_file = zip_entry
                        break
                
                if not matched_file:
                    # Try a more flexible search - look for the filename at the end
                    file_name_only = file_path.split('/')[-1]
                    for zip_entry in all_files:
                        if zip_entry.endswith(f"/{file_name_only}") or zip_entry == file_name_only:
                            matched_file = zip_entry
                            break
                
                if not matched_file:
                    raise HTTPException(
                        status_code=404, 
                        detail=f"File '{file_path}' not found in the uploaded project archive"
                    )
                
                # Check if it's a directory
                info = zf.getinfo(matched_file)
                if info.is_dir():
                    raise HTTPException(status_code=400, detail="Cannot read a directory as file content")
                
                # Read the file content
                with zf.open(matched_file, 'r') as f:
                    content_bytes = f.read()
                
                # Try to decode as text
                try:
                    content = content_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        content = content_bytes.decode('latin-1')
                    except:
                        # If it's binary, return a message
                        content = f"[Binary file: {file_name_only} - {len(content_bytes)} bytes]"
                
                return {
                    "success": True,
                    "content": content,
                    "file_path": file_path,
                    "matched_zip_path": matched_file
                }

        except zipfile.BadZipFile:
            raise HTTPException(status_code=500, detail="Invalid or corrupted project archive")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading file from archive: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download-url/{room_id}")
async def get_download_presigned_url(
    room_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Generate a presigned URL for downloading from R2"""
    try:
        if not r2_client:
            raise HTTPException(status_code=503, detail="Cloud storage not configured")

        user_id = uuid.UUID(current_user["id"])
        user_role = current_user.get("role", "")
        room_uuid = uuid.UUID(room_id)

        # Get chat room
        room_query = await db.execute(
            select(ChatRoom).where(ChatRoom.id == room_uuid)
        )
        room = room_query.scalar_one_or_none()

        if not room:
            raise HTTPException(status_code=404, detail="Chat room not found")

        # Authorization check
        if user_role != "admin" and user_id not in [room.developer_id, room.investor_id]:
            raise HTTPException(status_code=403, detail="Access denied")

        # Check if project is confirmed (for investor)
        is_investor = user_id == room.investor_id

        if user_role != "admin" and is_investor:
            payout_query = await db.execute(
                select(DeveloperPayout).where(
                    DeveloperPayout.project_id == room.project_id,
                    DeveloperPayout.investor_id == user_id
                )
            )
            payout = payout_query.scalar_one_or_none()

            if not payout:
                raise HTTPException(
                    status_code=403,
                    detail="You must confirm the project before downloading"
                )

        # Get upload record
        upload_query = await db.execute(
            select(ProjectUpload).where(ProjectUpload.room_id == room_uuid)
        )
        upload = upload_query.scalar_one_or_none()

        if not upload:
            raise HTTPException(status_code=404, detail="No uploaded project found")

        # Generate presigned download URL
        presigned_url = r2_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': settings.R2_BUCKET_NAME,
                'Key': upload.file_key,
            },
            ExpiresIn=3600
        )

        # Update status
        upload.status = 'downloaded'
        await db.commit()

        return {
            "download_url": presigned_url,
            "file_name": upload.file_name,
            "file_size": upload.file_size,
            "expires_in": 3600
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-image")
async def upload_project_image(
    image: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload and validate project image with OpenAI moderation"""
    temp_path = None

    try:
        # Validate file type
        filename = image.filename
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext not in ['.jpg', '.jpeg', '.png']:
            raise HTTPException(status_code=400, detail="Only JPEG and PNG images are allowed.")

        # Read file
        content = await image.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Cannot upload empty file.")

        # Validate size
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {MAX_UPLOAD_SIZE / 1024 / 1024} MB"
            )

        # Save temporary file
        temp_dir = "uploads/temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_filename = f"{uuid.uuid4()}_{filename}"
        temp_path = os.path.join(temp_dir, temp_filename)

        with open(temp_path, 'wb') as f:
            f.write(content)

        # OpenAI moderation
        from app.services.moderation_service import ModerationService
        moderation_result = await ModerationService.moderate_image(temp_path)


        if moderation_result["contains_harmful_content"]:
            os.remove(temp_path)
            raise HTTPException(
                status_code=400,
                detail="This image contains harmful or inappropriate content."
            )

        if moderation_result["contains_contact_info"]:
            os.remove(temp_path)
            raise HTTPException(
                status_code=400,
                detail="This image contains contact information (phone, email, social media, or URLs)."
            )

        # Optimize image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        user_id = current_user.get('id')
        clean_filename = os.path.splitext(os.path.basename(filename).replace("..", ""))[0]

        optimized_path = temp_path
        try:
            img = Image.open(temp_path)

            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background

            max_width = 1920
            if img.width > max_width:
                ratio = max_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

            optimized_path = temp_path.replace('.png', '.jpg').replace('.jpeg', '.jpg')
            img.save(optimized_path, format='JPEG', quality=85, optimize=True)
        except:
            optimized_path = temp_path

        # Upload to Cloudflare R2
        if not r2_client:
            raise HTTPException(
                status_code=503,
                detail="Cloud storage not configured. Please contact administrator."
            )

        # Generate unique file key for R2
        file_ext = os.path.splitext(optimized_path)[1] or '.jpg'
        file_key = f"project-images/{user_id}/{timestamp}_{clean_filename}{file_ext}"

        # Upload the optimized image to R2
        with open(optimized_path, 'rb') as img_file:
            r2_client.put_object(
                Bucket=settings.R2_BUCKET_NAME,
                Key=file_key,
                Body=img_file,
                ContentType='image/jpeg'
            )

        # Generate image URL - use permanent public URL if configured, otherwise presigned URL
        if settings.R2_PUBLIC_URL:
            # Permanent public URL (requires R2 bucket to be public)
            image_url = f"{settings.R2_PUBLIC_URL.rstrip('/')}/{file_key}"
        else:
            # Fallback: presigned URL (expires after 7 days)
            image_url = r2_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': settings.R2_BUCKET_NAME,
                    'Key': file_key,
                },
                ExpiresIn=604800  # 7 days
            )

        # Clean up temp files
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if optimized_path != temp_path and os.path.exists(optimized_path):
            os.remove(optimized_path)

        return {
            "success": True,
            "image_url": image_url,
            "filename": clean_filename,
            "delete_url": ""
        }

    except HTTPException:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")