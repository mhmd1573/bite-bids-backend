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
    async def send_verification_email(to_email: str, token: str, is_email_change: bool = False) -> bool:
        """Send email verification link to the user"""
        try:
            if is_email_change:
                verification_link = f"{settings.FRONTEND_URL}/verify-email-change?token={token}"
                subject = "Verify Your New Email Address"
                title = "Email Change Request"
                message = "You requested to change your email address. Please verify your new email."
            else:
                verification_link = f"{settings.FRONTEND_URL}/verify-email?token={token}"
                subject = "Verify your BiteBids account 🚀"
                title = "Welcome to BiteBids! 🎉"
                message = "Please verify your email address to continue."

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
                        <h2 style="text-align:center;">{title}</h2>
                        <p style="text-align:center;">{message}</p>
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
                subject=subject,
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

    @staticmethod
    async def send_project_approved_email(
        to_email: str,
        developer_name: str,
        project_title: str,
        amount: float,
        platform_fee: float,
        gross_amount: float,
        payoneer_transfer_id: str = None
    ) -> bool:
        """
        Send project approval email to developer
        ✅ UPDATED: Includes Payoneer transfer ID
        """
        try:
            subject = f"✅ Project Approved: '{project_title}'"

            safe_amount = float(amount or 0)
            safe_fee = float(platform_fee or 0)
            safe_gross = float(gross_amount or 0)

            transfer_status = "Being processed via Payoneer"
            if payoneer_transfer_id:
                transfer_status = f"Processing via Payoneer (Transfer ID: {payoneer_transfer_id[:16]}...)"

            html_content = f"""\
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; text-align: center; }}
                    .content {{ background: #f8fafc; padding: 30px; border: 1px solid #e0e0e0; border-top: none; }}
                    .amount-box {{ background: #d1fae5; padding: 20px; border-radius: 8px; text-align: center; margin: 15px 0; }}
                    .transfer-id {{ background: #f3f4f6; padding: 10px; border-radius: 6px; font-family: monospace; font-size: 0.9rem; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🎉 Project Approved!</h1>
                </div>
                <div class="content">
                    <h2>Hi {developer_name},</h2>
                    <p>Your project <strong>"{project_title}"</strong> has been approved by the investor!</p>
                    
                    <div class="amount-box">
                        <h3>Payment Details</h3>
                        <p><strong>Gross Amount:</strong> ${safe_gross:,.2f}</p>
                        <p><strong>Platform Fee (6%):</strong> -${safe_fee:,.2f}</p>
                        <p style="font-size: 24px; font-weight: bold; color: #059669;">
                            Your Payout: ${safe_amount:,.2f}
                        </p>
                    </div>
                    
                    <p><strong>Payment Method:</strong> Payoneer</p>
                    <p><strong>Status:</strong> {transfer_status}</p>
                    
                    <p>Funds typically arrive within 1-2 business days.</p>
                    
                    <a href="{settings.FRONTEND_URL}/dashboard" 
                       style="display:inline-block; background:#10b981; color:white; padding:12px 24px; border-radius:6px; text-decoration:none; margin-top:15px;">
                        View Dashboard
                    </a>
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
            logger.error(f"❌ Failed to send project approval email: {str(e)}")
            return False

    @staticmethod
    async def send_developer_edit_notification(
        developer_email: str,
        developer_name: str,
        project_title: str,
        project_id: str,
        admin_name: str,
        changes: dict
    ) -> bool:
        """
        Send email notification to developer when admin edits their project
        """
        try:
            subject = f"⚠️ Admin Updated Your Project: {project_title}"
            
            # Build changes HTML
            changes_html = ""
            for field, change_data in changes.items():
                old_value = change_data.get('old')
                new_value = change_data.get('new')
                reason = change_data.get('reason', '')
                
                field_name = field.replace('_', ' ').title()
                
                if field == 'images':
                    old_count = len(old_value) if old_value else 0
                    new_count = len(new_value) if new_value else 0
                    changes_html += f"""
                    <tr>
                        <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                            <strong style="color: #ef4444;">🗑️ {field_name}</strong>
                        </td>
                        <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                            <span style="color: #6b7280;">{old_count} images</span>
                        </td>
                        <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                            <span style="color: #059669;">{new_count} images</span>
                        </td>
                    </tr>
                    """
                    if reason:
                        changes_html += f"""
                        <tr>
                            <td colspan="3" style="padding: 8px 12px; background: #fef3c7; border-bottom: 1px solid #e5e7eb;">
                                <em style="color: #92400e; font-size: 0.875rem;">Reason: {reason}</em>
                            </td>
                        </tr>
                        """
                elif field == 'tech_stack':
                    old_stack = ', '.join(old_value) if old_value else 'None'
                    new_stack = ', '.join(new_value) if new_value else 'None'
                    changes_html += f"""
                    <tr>
                        <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                            <strong style="color: #3b82f6;">🔧 {field_name}</strong>
                        </td>
                        <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                            <span style="color: #6b7280; font-size: 0.875rem;">{old_stack}</span>
                        </td>
                        <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                            <span style="color: #059669; font-size: 0.875rem;">{new_stack}</span>
                        </td>
                    </tr>
                    """
                elif field == 'budget':
                    changes_html += f"""
                    <tr>
                        <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                            <strong style="color: #10b981;">💰 {field_name}</strong>
                        </td>
                        <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                            <span style="color: #6b7280;">${float(old_value or 0):,.2f}</span>
                        </td>
                        <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                            <span style="color: #059669;">${float(new_value or 0):,.2f}</span>
                        </td>
                    </tr>
                    """
                else:
                    changes_html += f"""
                    <tr>
                        <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                            <strong style="color: #6366f1;">📝 {field_name}</strong>
                        </td>
                        <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                            <span style="color: #6b7280;">{old_value or 'N/A'}</span>
                        </td>
                        <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                            <span style="color: #059669;">{new_value or 'N/A'}</span>
                        </td>
                    </tr>
                    """
                    if reason:
                        changes_html += f"""
                        <tr>
                            <td colspan="3" style="padding: 8px 12px; background: #fef3c7; border-bottom: 1px solid #e5e7eb;">
                                <em style="color: #92400e; font-size: 0.875rem;">Reason: {reason}</em>
                            </td>
                        </tr>
                        """

            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; text-align: center; }}
                    .content {{ background: #f8fafc; padding: 30px; border: 1px solid #e0e0e0; border-top: none; }}
                    .changes-table {{ width: 100%; border-collapse: collapse; }}
                    .changes-table th {{ background: #f3f4f6; padding: 12px; text-align: left; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>⚠️ Admin Updated Your Project</h1>
                </div>
                <div class="content">
                    <h2>Hi {developer_name},</h2>
                    <p>An administrator (<strong>{admin_name}</strong>) has made changes to your project <strong>"{project_title}"</strong>.</p>
                    <p>Please review the changes below:</p>
                    
                    <table class="changes-table">
                        <thead>
                            <tr>
                                <th>Field</th>
                                <th>Previous Value</th>
                                <th>New Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            {changes_html}
                        </tbody>
                    </table>
                    
                    <hr />
                    <p>
                        <a href="{settings.FRONTEND_URL}/dashboard" 
                           style="display:inline-block; background:#4f46e5; color:white; padding:12px 24px; border-radius:6px; text-decoration:none; margin-top:15px;">
                            View Your Dashboard →
                        </a>
                    </p>
                </div>
            </body>
            </html>
            """

            return await EmailService.send_email(
                to_email=developer_email,
                subject=subject,
                html_content=html_content
            )

        except Exception as e:
            logger.error(f"❌ Failed to send developer edit notification: {str(e)}")
            return False