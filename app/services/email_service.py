# app/services/email_service.py
import smtplib
import asyncio
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from datetime import datetime

from app.config import settings

logger = logging.getLogger(__name__)


class EmailService:
    """Service for sending emails"""
    
    @staticmethod
    async def send_email(
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None
    ) -> bool:
        """Send an email using SMTP"""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = settings.EMAIL_FROM
            msg["To"] = to_email

            if text_content:
                msg.attach(MIMEText(text_content, "plain"))
            msg.attach(MIMEText(html_content, "html"))

            server = smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT)
            server.starttls()
            server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            server.sendmail(settings.EMAIL_FROM, to_email, msg.as_string())
            server.quit()

            logger.info(f"📧 Email sent successfully to {to_email}")
            return True

        except Exception as e:
            logger.error(f"❌ Email sending failed to {to_email}: {str(e)}")
            return False

    @staticmethod
    async def send_verification_email(to_email: str, token: str) -> bool:
        """Send email verification link to the user"""
        try:
            verification_link = f"{settings.FRONTEND_URL}/verify-email?token={token}"

            html_content = f"""\
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1.0" />
                <title>Email Verification</title>
                <style>
                    body {{ margin:0; padding:0; background:#f8fafc; font-family:'Segoe UI', Arial, sans-serif; }}
                    .container {{ max-width:600px; margin:0 auto; padding:40px 20px; }}
                    .card {{ background:white; border-radius:16px; padding:40px; box-shadow:0 4px 20px rgba(0,0,0,0.06); }}
                    .btn {{ display:inline-block; background:#4f46e5; color:white; padding:14px 40px; border-radius:10px; text-decoration:none; font-weight:600; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="card">
                        <h1 style="color:#4f46e5; text-align:center;">BiteBids</h1>
                        <h2 style="text-align:center;">Welcome to BiteBids! 🎉</h2>
                        <p style="text-align:center;">Please verify your email address to continue.</p>
                        <div style="text-align:center; margin:30px 0;">
                            <a href="{verification_link}" class="btn">Verify Email</a>
                        </div>
                        <p style="color:#6b7280; font-size:13px;">
                            If the button doesn't work, copy the link below:
                        </p>
                        <p style="word-break:break-all;">
                            <a href="{verification_link}" style="color:#4f46e5;">{verification_link}</a>
                        </p>
                        <hr style="border:0; border-top:1px solid #e5e7eb; margin:20px 0;" />
                        <p style="text-align:center; font-size:11px; color:#9ca3af;">
                            © 2024 BiteBids. All rights reserved.
                        </p>
                    </div>
                </div>
            </body>
            </html>
            """

            return await EmailService.send_email(
                to_email=to_email,
                subject="Verify your BiteBids account 🚀",
                html_content=html_content
            )

        except Exception as e:
            logger.error(f"❌ Failed to send verification email: {str(e)}")
            return False

    @staticmethod
    async def send_payment_confirmation_email(
        to_email: str,
        project_title: str,
        amount: float,
        role: str = "investor"
    ) -> bool:
        """Send payment confirmation email"""
        try:
            subject = (
                f"✅ Payment successful for '{project_title}'"
                if role == "investor"
                else f"💰 Client payment received for '{project_title}'"
            )

            role_message = (
                "Your payment was successfully processed and held in BiteBids escrow."
                if role == "investor"
                else "Your client has completed payment. Funds are now held in BiteBids escrow."
            )

            html_content = f"""\
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ margin:0; padding:0; background:#f8fafc; font-family:'Segoe UI', Arial, sans-serif; }}
                    .container {{ max-width:600px; margin:0 auto; padding:40px 20px; }}
                    .card {{ background:white; border-radius:16px; padding:40px; box-shadow:0 4px 20px rgba(0,0,0,0.06); }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="card">
                        <h1 style="color:#4f46e5; text-align:center;">BiteBids</h1>
                        <h2 style="text-align:center;">{subject}</h2>
                        <p style="text-align:center;">{role_message}</p>
                        <p style="text-align:center; font-size:18px; font-weight:600;">
                            Project: <span style="color:#4f46e5;">{project_title}</span><br/>
                            Amount: <span style="color:#059669;">${float(amount):,.2f}</span>
                        </p>
                        <hr style="border:0; border-top:1px solid #e5e7eb; margin:20px 0;" />
                        <p style="text-align:center; font-size:11px; color:#9ca3af;">
                            © 2024 BiteBids. All rights reserved.
                        </p>
                    </div>
                </div>
            </body>
            </html>
            """

            return await EmailService.send_email(
                to_email=to_email,
                subject=subject,
                html_content=html_content
            )

        except Exception as e:
            logger.error(f"❌ Failed to send payment confirmation email: {str(e)}")
            return False

    @staticmethod
    async def send_admin_project_notification(
        action: str,
        project_id: str,
        project_title: str,
        developer_name: str,
        developer_email: str,
        project_data: dict = None
    ) -> bool:
        """Send notification to admin when project is created/updated/deleted"""
        try:
            admin_email = settings.ADMIN_EMAIL or "bitebids@gmail.com"
            
            action_config = {
                "created": {"emoji": "🎉", "color": "#22c55e", "title": "New Project Posted"},
                "updated": {"emoji": "✏️", "color": "#3b82f6", "title": "Project Updated"},
                "deleted": {"emoji": "🗑️", "color": "#ef4444", "title": "Project Deleted"}
            }
            
            config = action_config.get(action, action_config["created"])
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ margin:0; padding:0; background:#f8fafc; font-family:'Segoe UI', Arial, sans-serif; }}
                    .container {{ max-width:600px; margin:0 auto; padding:40px 20px; }}
                    .card {{ background:white; border-radius:16px; padding:40px; box-shadow:0 4px 20px rgba(0,0,0,0.06); }}
                    .badge {{ display:inline-block; background:{config['color']}; color:white; padding:12px 24px; border-radius:8px; font-weight:600; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="card">
                        <h1 style="color:#4f46e5; text-align:center;">BiteBids Admin</h1>
                        <div style="text-align:center; margin:20px 0;">
                            <span class="badge">{config['emoji']} {config['title']}</span>
                        </div>
                        <h2 style="text-align:center;">{project_title}</h2>
                        <p><strong>Developer:</strong> {developer_name} ({developer_email})</p>
                        <p><strong>Project ID:</strong> {project_id}</p>
                        <p style="text-align:center; margin-top:20px;">
                            <a href="{settings.FRONTEND_URL}/dashboard-admin" 
                               style="display:inline-block; background:#4f46e5; color:white; padding:14px 32px; border-radius:8px; text-decoration:none; font-weight:600;">
                                View in Admin Dashboard →
                            </a>
                        </p>
                        <hr style="border:0; border-top:1px solid #e5e7eb; margin:20px 0;" />
                        <p style="text-align:center; font-size:11px; color:#9ca3af;">
                            © 2024 BiteBids. All rights reserved.
                        </p>
                    </div>
                </div>
            </body>
            </html>
            """

            return await EmailService.send_email(
                to_email=admin_email,
                subject=f"{config['emoji']} {config['title']} - {project_title}",
                html_content=html_content
            )

        except Exception as e:
            logger.error(f"❌ Failed to send admin notification: {str(e)}")
            return False