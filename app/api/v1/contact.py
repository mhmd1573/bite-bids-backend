# app/api/v1/contact.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from app.database import get_db
from app.models.contact import ContactFormRecord
from app.schemas.contact import ContactFormSubmission
from app.services.email_service import EmailService
from app.config import settings

router = APIRouter(prefix="/contact", tags=["Contact"])


@router.post("/submit")
async def submit_contact_form(
    form_data: ContactFormSubmission,
    db: AsyncSession = Depends(get_db)
):
    """Submit contact form - sends email to admin team"""
    try:
        # Category display names
        category_names = {
            'general': 'General Inquiry',
            'technical': 'Technical Support',
            'billing': 'Billing & Payments',
            'partnership': 'Partnership',
            'feedback': 'Feedback'
        }
        
        category_display = category_names.get(form_data.category, form_data.category.title())
        
        # Save to database
        record = ContactFormRecord(
            name=form_data.name,
            email=form_data.email,
            subject=form_data.subject,
            category=form_data.category,
            message=form_data.message
        )
        db.add(record)
        await db.commit()
        
        # Prepare email content for admin
        subject = f"[Contact Form] {category_display}: {form_data.subject}"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; text-align: center; }}
                .content {{ background: #f8f9fa; padding: 30px; border: 1px solid #e0e0e0; border-top: none; }}
                .field {{ margin-bottom: 20px; background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #667eea; }}
                .field-label {{ font-weight: bold; color: #667eea; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
                .field-value {{ color: #333; font-size: 15px; }}
                .message-box {{ background: white; padding: 20px; border-radius: 8px; border-left: 4px solid #667eea; white-space: pre-wrap; }}
                .footer {{ background: #f8f9fa; padding: 20px; text-align: center; color: #666; font-size: 13px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📧 New Contact Form Submission</h1>
            </div>
            <div class="content">
                <div class="field">
                    <div class="field-label">Category</div>
                    <div class="field-value">{category_display}</div>
                </div>
                <div class="field">
                    <div class="field-label">From</div>
                    <div class="field-value">{form_data.name}</div>
                </div>
                <div class="field">
                    <div class="field-label">Email</div>
                    <div class="field-value">{form_data.email}</div>
                </div>
                <div class="field">
                    <div class="field-label">Subject</div>
                    <div class="field-value">{form_data.subject}</div>
                </div>
                <div class="field-label" style="margin-bottom: 10px;">Message</div>
                <div class="message-box">{form_data.message}</div>
            </div>
            <div class="footer">
                <p>Received on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            </div>
        </body>
        </html>
        """
        
        # Send email to admin
        admin_email = settings.ADMIN_EMAIL or "bitebids@gmail.com"
        
        await EmailService.send_email(
            to_email=admin_email,
            subject=subject,
            html_content=html_content
        )
        
        # Send auto-reply to user
        auto_reply_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; text-align: center; }}
                .content {{ background: #f8f9fa; padding: 30px; border: 1px solid #e0e0e0; }}
                .footer {{ background: #f8f9fa; padding: 20px; text-align: center; color: #666; font-size: 13px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>✅ Thank You for Contacting BiteBids!</h1>
            </div>
            <div class="content">
                <p>Hi {form_data.name},</p>
                <p>Thank you for reaching out to us! We've received your message and our team will review it shortly.</p>
                <p><strong>Your submission details:</strong></p>
                <ul>
                    <li><strong>Category:</strong> {category_display}</li>
                    <li><strong>Subject:</strong> {form_data.subject}</li>
                </ul>
                <p>We typically respond within 24 hours during business days.</p>
                <p>Best regards,<br>The BiteBids Team</p>
            </div>
        </body>
        </html>
        """
        
        await EmailService.send_email(
            to_email=form_data.email,
            subject=f"We received your message: {form_data.subject}",
            html_content=auto_reply_html
        )
        
        return {
            "success": True,
            "message": "Thank you for your message! We'll get back to you within 24 hours."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")