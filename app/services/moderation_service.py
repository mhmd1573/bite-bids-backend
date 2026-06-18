# app/services/moderation_service.py
import json
import re
import logging
from typing import Dict, List, Optional
from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None


class ContentFilter:
    """Filter messages to prevent sharing contact information"""
    
    PHONE_PATTERNS = [
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
        r'\b\d{10,}\b',
        r'\+\d{1,3}\s?\d{6,}',
        r'\(\d{3}\)\s?\d{3}[-.\s]?\d{4}',
    ]
    
    EMAIL_PATTERNS = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        r'\b[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\s*\.\s*[A-Z|a-z]{2,}\b',
        r'\b[A-Za-z0-9._%+-]+\s*\[at\]\s*[A-Za-z0-9.-]+\s*\[dot\]\s*[A-Z|a-z]{2,}\b',
    ]
    
    SOCIAL_PATTERNS = [
        r'@[A-Za-z0-9_]{1,15}\b',
        r'facebook\.com/[A-Za-z0-9._]+',
        r'linkedin\.com/in/[A-Za-z0-9-]+',
        r'telegram\.me/[A-Za-z0-9_]+',
        r'wa\.me/\d+',
    ]
    
    BYPASS_PATTERNS = [
        r'\b(contact|call|text|email|phone|whatsapp|telegram|skype)\s+(me|us)\b',
        r'\b(my|our)\s+(email|phone|number|whatsapp)\b',
        r'\b(reach|contact|call)\s+(me|us)\s+(at|on)\b',
        r'\b(dm|message)\s+(me|us)\b',
        r'\b(let\'?s\s+)?(talk|chat|discuss)\s+(outside|off|directly)\b',
    ]
    
    @classmethod
    def check_message(cls, message: str) -> dict:
        """Check message for prohibited content"""
        violations = []
        filtered_message = message
        
        for pattern in cls.PHONE_PATTERNS:
            matches = re.findall(pattern, message, re.IGNORECASE)
            if matches:
                violations.append({'type': 'phone', 'matches': matches})
                for match in matches:
                    filtered_message = filtered_message.replace(match, '[PHONE REMOVED]')
        
        for pattern in cls.EMAIL_PATTERNS:
            matches = re.findall(pattern, message, re.IGNORECASE)
            if matches:
                violations.append({'type': 'email', 'matches': matches})
                for match in matches:
                    filtered_message = filtered_message.replace(match, '[EMAIL REMOVED]')
        
        for pattern in cls.SOCIAL_PATTERNS:
            matches = re.findall(pattern, message, re.IGNORECASE)
            if matches:
                violations.append({'type': 'social', 'matches': matches})
                for match in matches:
                    filtered_message = filtered_message.replace(match, '[SOCIAL MEDIA REMOVED]')
        
        for pattern in cls.BYPASS_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                violations.append({'type': 'bypass_attempt', 'pattern': pattern})
        
        return {
            'is_safe': len(violations) == 0,
            'violations': violations,
            'filtered_message': filtered_message if violations else message,
            'original_message': message
        }


class ModerationService:
    """Service for content moderation using OpenAI"""
    
    @staticmethod
    async def moderate_chat_message(message: str) -> Dict:
        """Moderate a chat message using OpenAI"""
        if not openai_client:
            logger.error("OpenAI client not initialized")
            return {
                "is_safe": True,
                "contains_harmful_content": False,
                "contains_contact_info": False,
                "violations": [],
                "reason": ""
            }
        
        try:
            # First, use regex-based filtering
            filter_result = ContentFilter.check_message(message)
            
            if not filter_result['is_safe']:
                return {
                    "is_safe": False,
                    "contains_harmful_content": False,
                    "contains_contact_info": True,
                    "violations": filter_result['violations'],
                    "reason": "Contains prohibited content"
                }
            
            # Then use OpenAI moderation API
            moderation_response = openai_client.moderations.create(
                model="omni-moderation-latest",
                input=message
            )
            
            moderation_result = moderation_response.results[0]
            
            if moderation_result.flagged:
                flagged_categories = [
                    category for category, flagged in moderation_result.categories.model_dump().items()
                    if flagged
                ]
                
                return {
                    "is_safe": False,
                    "contains_harmful_content": True,
                    "contains_contact_info": False,
                    "violations": [{
                        "type": "openai_moderation",
                        "categories": flagged_categories
                    }],
                    "reason": f"Content violates guidelines: {', '.join(flagged_categories)}"
                }
            
            return {
                "is_safe": True,
                "contains_harmful_content": False,
                "contains_contact_info": False,
                "violations": [],
                "reason": ""
            }
            
        except Exception as e:
            logger.error(f"Moderation error: {e}")
            return {
                "is_safe": True,  # Fail open on errors
                "contains_harmful_content": False,
                "contains_contact_info": False,
                "violations": [],
                "reason": "Moderation service temporarily unavailable"
            }

    @staticmethod
    async def moderate_image(image_path: str) -> Dict:
        """Moderate an image using OpenAI GPT-4 Vision"""
        if not openai_client:
            logger.error("OpenAI client not initialized")
            return {
                "contains_harmful_content": True,
                "contains_contact_info": True,
            }
        
        try:
            import base64
            import os
            
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
            
            ext = os.path.splitext(image_path)[1].lower()
            mime_type = "image/png" if ext == ".png" else "image/jpeg"
            data_url = f"data:{mime_type};base64,{image_base64}"
            
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an image moderation system for BiteBids, a private project-bidding platform.\n\n"
                            "ALLOWED images: Screenshots of code, diagrams, UI mockups, architecture, flowcharts, "
                            "software dashboards, terminals, databases, APIs, project-related technical graphics.\n\n"
                            "DISALLOWED images (flag as harmful OR contact info): Any human faces, bodies, hands, selfies, "
                            "avatars, profile photos, real-world photos, any text containing emails, phone numbers, URLs, "
                            "QR codes, usernames, social handles, names, logos, watermarks, signatures, "
                            "any non-software or non-technical content, any harmful, illegal, sexual, violent, or unsafe content.\n\n"
                            "If the image is NOT clearly a software or technical project image, it MUST be flagged.\n"
                            "When uncertain, always return true for both fields.\n\n"
                            "Reply ONLY in valid JSON exactly:\n"
                            '{"contains_harmful_content": true/false, "contains_contact_info": true/false}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this image strictly."},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                max_tokens=50,
                temperature=0.0,
            )
            
            content = response.choices[0].message.content.strip()
            match = re.search(r"\{[\s\S]*\}", content)
            
            if not match:
                return {"contains_harmful_content": True, "contains_contact_info": True}
            
            result = json.loads(match.group())
            
            return {
                "contains_harmful_content": bool(result.get("contains_harmful_content", True)),
                "contains_contact_info": bool(result.get("contains_contact_info", True)),
            }
            
        except Exception as e:
            logger.error(f"Image moderation error: {e}")
            return {"contains_harmful_content": True, "contains_contact_info": True}