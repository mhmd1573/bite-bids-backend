# app/api/v1/oauth.py
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
import json
import urllib.parse
import httpx
from datetime import datetime
import logging

from app.database import get_db
from app.models.user import User
from app.schemas.auth import OAuthValidation
from app.core.security import create_jwt_token, hash_password  # ✅ Added hash_password
from app.core.dependencies import get_current_user
from app.core.exceptions import UnauthorizedException
from app.utils.converters import model_to_dict
from app.config import settings

# ✅ Added logger
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["OAuth"])


@router.post("/oauth/validate")
async def validate_oauth_session(
    oauth_data: OAuthValidation,
    db: AsyncSession = Depends(get_db)
):
    """Validate OAuth session"""
    try:
        # Call Emergent auth API
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
                headers={"X-Session-ID": oauth_data.session_id},
                timeout=10
            )
        
        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid OAuth session")
        
        oauth_user = response.json()
        required_fields = ["id", "email", "name"]
        if not all(field in oauth_user for field in required_fields):
            raise HTTPException(status_code=400, detail="Incomplete OAuth user data")
        
        # Check if user exists
        result = await db.execute(select(User).where(User.email == oauth_user["email"]))
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            user_dict = model_to_dict(existing_user)
        else:
            # Create new user
            new_user = User(
                email=oauth_user["email"],
                password_hash=None,
                role="developer",
                status="active",
                name=oauth_user["name"],
                reputation_score=0,
                profile={
                    "cosmic_theme": "default",
                    "avatar": oauth_user.get("picture"),
                    "bio": "",
                    "oauth_provider": True
                },
                oauth_id=oauth_user["id"]
            )
            db.add(new_user)
            await db.commit()
            await db.refresh(new_user)
            user_dict = model_to_dict(new_user)
        
        # Generate JWT token
        token = create_jwt_token(user_dict)
        
        return {
            "token": token,
            "user": {
                "id": user_dict["_id"],
                "email": user_dict["email"],
                "role": user_dict["role"],
                "name": user_dict["name"],
                "avatar": user_dict.get("profile", {}).get("avatar") if isinstance(user_dict.get("profile"), dict) else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"OAuth validation failed: {str(e)}")


@router.post("/oauth/complete")
async def complete_oauth_registration(
    request: dict, 
    db: AsyncSession = Depends(get_db)
):
    """Complete OAuth registration"""
    try:
        oauth_data = request.get("oauth_data")
        provider = request.get("provider")
        selected_role = request.get("role")
        company = request.get("company", "")
        password = request.get("password")  

        if not oauth_data or not provider or not selected_role:
            raise HTTPException(status_code=400, detail="OAuth data, provider, and role are required")
        
        email = oauth_data.get("email")
        if not email:
            raise HTTPException(status_code=400, detail="Email is required")
        
        # ✅ FIX: Ensure name is never NULL
        name = oauth_data.get("name")
        if not name or not name.strip():
            # Fallback 1: Use email username
            name = email.split("@")[0]
        
        # Additional validation
        if not company or not company.strip():
            raise HTTPException(status_code=400, detail="Company is required")
        
        # Check if user exists
        result = await db.execute(select(User).where(User.email == email))
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Hash Password
        password_hash = hash_password(password)

        # ✅ FIX: Create new user with guaranteed name value
        new_user = User(
            email=email,
            name=name,  # Now guaranteed to have a value
            password_hash=password_hash,
            role=selected_role,
            email_verified=True,
            verified=True,  # OAuth users are automatically verified
            status="active",
            company=company,
            oauth_provider=provider,
            oauth_id=str(oauth_data.get("provider_id", "")),
            reputation_score=0,
            profile={
                "avatar_url": oauth_data.get("avatar_url") or oauth_data.get("picture", "")
            }
        )
        
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        
        # Generate JWT token
        user_dict = model_to_dict(new_user)
        token = create_jwt_token(user_dict)
        
        return {
            "token": token,
            "user": {
                "id": str(new_user.id),
                "email": new_user.email,
                "name": new_user.name,
                "role": new_user.role
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OAuth registration error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to complete registration")


@router.get("/login/{provider}")
async def oauth_login(provider: str, request: Request):
    """Initiate OAuth login"""
    if provider not in ['github', 'google']:
        raise HTTPException(status_code=400, detail="Unsupported OAuth provider")
    
    if provider == 'github' and not (settings.GITHUB_CLIENT_ID and settings.GITHUB_CLIENT_SECRET):
        raise HTTPException(status_code=400, detail="GitHub OAuth not configured")
    
    if provider == 'google' and not (settings.GOOGLE_CLIENT_ID and settings.GOOGLE_CLIENT_SECRET):
        raise HTTPException(status_code=400, detail="Google OAuth not configured")
    
    # Build OAuth authorization URL
    if provider == 'github':
        auth_url = (f"https://github.com/login/oauth/authorize?"
                   f"client_id={settings.GITHUB_CLIENT_ID}&"
                   f"scope=user:email&"
                   f"redirect_uri={settings.BASE_URL}/api/auth/callback/github&"
                   f"state={uuid.uuid4()}")
    elif provider == 'google':
        auth_url = (f"https://accounts.google.com/o/oauth2/v2/auth?"
                   f"client_id={settings.GOOGLE_CLIENT_ID}&"
                   f"response_type=code&"
                   f"scope=openid email profile&"
                   f"redirect_uri={settings.BASE_URL}/api/auth/callback/google&"
                   f"state={uuid.uuid4()}")
    
    return RedirectResponse(url=auth_url)


@router.get("/callback/github")
async def github_callback(code: str, request: Request, db: AsyncSession = Depends(get_db)):
    """Handle GitHub OAuth callback"""
    if not (settings.GITHUB_CLIENT_ID and settings.GITHUB_CLIENT_SECRET):
        raise HTTPException(status_code=400, detail="GitHub OAuth not configured")
    
    try:
        # Exchange code for access token
        token_url = "https://github.com/login/oauth/access_token"
        token_data = {
            "client_id": settings.GITHUB_CLIENT_ID,
            "client_secret": settings.GITHUB_CLIENT_SECRET,
            "code": code
        }
        headers = {"Accept": "application/json"}
        
        async with httpx.AsyncClient() as client:
            token_response = await client.post(token_url, data=token_data, headers=headers)
            token_json = token_response.json()
            
            if "access_token" not in token_json:
                error_description = token_json.get("error_description", "Unknown error")
                error_message = urllib.parse.quote(f"Failed to get access token: {error_description}")
                return RedirectResponse(url=f"{settings.FRONTEND_URL}?auth=error&message={error_message}")
            
            access_token = token_json["access_token"]
            
            # Get user info
            user_response = await client.get(
                "https://api.github.com/user",
                headers={"Authorization": f"token {access_token}"}
            )
            user_data = user_response.json()
            
            # Get user email
            email_response = await client.get(
                "https://api.github.com/user/emails",
                headers={"Authorization": f"token {access_token}"}
            )
            emails = email_response.json()
            primary_email = next((email["email"] for email in emails if email["primary"]), None)
            
            if not primary_email:
                error_message = urllib.parse.quote("No primary email found")
                return RedirectResponse(url=f"{settings.FRONTEND_URL}?auth=error&message={error_message}")
            
            # Check if user exists
            result = await db.execute(select(User).where(User.email == primary_email))
            existing_user = result.scalar_one_or_none()
            
            if existing_user:
                existing_user.last_login = datetime.utcnow()
                existing_user.verified = True
                await db.commit()
                await db.refresh(existing_user)
                
                user_dict = model_to_dict(existing_user)
                token = create_jwt_token(user_dict)
                
                user_info = {
                    "id": str(existing_user.id),
                    "email": existing_user.email,
                    "name": existing_user.name,
                    "role": existing_user.role
                }
                
                # Role-based redirect
                if existing_user.role == "investor":
                    redirect_page = "home"
                elif existing_user.role == "developer":
                    redirect_page = "dashboard"
                elif existing_user.role == "admin":
                    redirect_page = "dashboard-admin"
                else:
                    redirect_page = "home"
                
                encoded_user = urllib.parse.quote(json.dumps(user_info))
                
                return RedirectResponse(
                    url=f"{settings.FRONTEND_URL}/{redirect_page}?token={token}&user={encoded_user}&auth=success"
                )
            else:
                # New user - redirect to registration
                github_data = {
                    "email": primary_email,
                    "name": user_data.get("name", user_data.get("login", primary_email.split("@")[0])),
                    "provider": "github",
                    "provider_id": str(user_data["id"]),
                    "avatar_url": user_data.get("avatar_url", "")
                }
                
                encoded_data = urllib.parse.quote(json.dumps(github_data))
                
                return RedirectResponse(
                    url=f"{settings.FRONTEND_URL}?oauth_data={encoded_data}&auth=register&provider=github"
                )
                
    except Exception as e:
        error_message = urllib.parse.quote(str(e))
        return RedirectResponse(url=f"{settings.FRONTEND_URL}?auth=error&message={error_message}")


@router.get("/callback/google")
async def google_callback(code: str, request: Request, db: AsyncSession = Depends(get_db)):
    """Handle Google OAuth callback"""
    if not (settings.GOOGLE_CLIENT_ID and settings.GOOGLE_CLIENT_SECRET):
        raise HTTPException(status_code=400, detail="Google OAuth not configured")
    
    try:
        # Exchange code for access token
        token_url = "https://oauth2.googleapis.com/token"
        token_data = {
            "client_id": settings.GOOGLE_CLIENT_ID,
            "client_secret": settings.GOOGLE_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": f"{settings.BASE_URL}/api/auth/callback/google"
        }
        
        async with httpx.AsyncClient() as client:
            token_response = await client.post(token_url, data=token_data)
            token_json = token_response.json()
            
            if "access_token" not in token_json:
                error_description = token_json.get("error_description", "Unknown error")
                error_message = urllib.parse.quote(f"Failed to get access token: {error_description}")
                return RedirectResponse(url=f"{settings.FRONTEND_URL}?auth=error&message={error_message}")
            
            access_token = token_json["access_token"]
            
            # Get user info
            user_response = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            user_data = user_response.json()
            
            email = user_data.get("email")
            if not email:
                error_message = urllib.parse.quote("No email found in Google account")
                return RedirectResponse(url=f"{settings.FRONTEND_URL}?auth=error&message={error_message}")
            
            # Check if user exists
            result = await db.execute(select(User).where(User.email == email))
            existing_user = result.scalar_one_or_none()
            
            if existing_user:
                existing_user.last_login = datetime.utcnow()
                existing_user.verified = True
                await db.commit()
                await db.refresh(existing_user)
                
                user_dict = model_to_dict(existing_user)
                token = create_jwt_token(user_dict)
                
                user_info = {
                    "id": str(existing_user.id),
                    "email": existing_user.email,
                    "name": existing_user.name,
                    "role": existing_user.role
                }
                
                if existing_user.role == "investor":
                    redirect_page = "home"
                elif existing_user.role == "developer":
                    redirect_page = "dashboard"
                elif existing_user.role == "admin":
                    redirect_page = "dashboard-admin"
                else:
                    redirect_page = "home"
                
                encoded_user = urllib.parse.quote(json.dumps(user_info))
                
                return RedirectResponse(
                    url=f"{settings.FRONTEND_URL}/{redirect_page}?token={token}&user={encoded_user}&auth=success"
                )
            else:
                google_data = {
                    "email": email,
                    "name": user_data.get("name", email.split("@")[0]),
                    "provider": "google",
                    "provider_id": user_data.get("id", ""),
                    "avatar_url": user_data.get("picture", "")
                }
                
                encoded_data = urllib.parse.quote(json.dumps(google_data))
                
                return RedirectResponse(
                    url=f"{settings.FRONTEND_URL}?oauth_data={encoded_data}&auth=register&provider=google"
                )
                
    except Exception as e:
        error_message = urllib.parse.quote(str(e))
        return RedirectResponse(url=f"{settings.FRONTEND_URL}?auth=error&message={error_message}")