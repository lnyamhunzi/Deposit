import smtplib
from email.mime.text import MIMEText
from typing import List, Dict, Any, Optional
import os
from datetime import datetime, timedelta

class NotificationService:
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.example.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        self.smtp_username = os.getenv("SMTP_USERNAME", "your_email@example.com")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "your_email_password")
        self.sender_email = os.getenv("SENDER_EMAIL", "no-reply@example.com")

    def send_email(self, recipients: List[str], subject: str, body: str, html_body: str = None) -> Dict[str, Any]:
        """Sends an email to the specified recipients."""
        try:
            msg = MIMEText(html_body if html_body else body, 'html' if html_body else 'plain')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = ", ".join(recipients)

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.sendmail(self.sender_email, recipients, msg.as_string())
            
            return {"success": True, "message": "Email sent successfully."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def send_task_assignment_notification(self, recipient_email: str, task_name: str, deadline: str) -> Dict[str, Any]:
        """Sends a notification for a task assignment."""
        subject = f"Task Assignment: {task_name}"
        body = f"""
        <html>
        <body>
            <p>Dear User,</p>
            <p>You have been assigned a new task: <b>{task_name}</b>.</p>
            <p>The deadline for this task is: <b>{deadline}</b>.</p>
            <p>Please log in to the system to view more details.</p>
            <p>Regards,</p>
            <p>Your System</p>
        </body>
        </html>
        """
        return self.send_email(recipients=[recipient_email], subject=subject, html_body=body)

    def send_deadline_reminder(self, recipient_email: str, task_name: str, deadline: str) -> Dict[str, Any]:
        """Sends a deadline reminder notification."""
        subject = f"Reminder: Task Deadline Approaching for {task_name}"
        body = f"""
        <html>
        <body>
            <p>Dear User,</p>
            <p>This is a reminder that the deadline for your task <b>{task_name}</b> is approaching.</p>
            <p>The deadline is: <b>{deadline}</b>.</p>
            <p>Please ensure to complete it on time.</p>
            <p>Regards,</p>
            <p>Your System</p>
        </body>
        </html>
        """
        return self.send_email(recipients=[recipient_email], subject=subject, html_body=body)

    # Placeholder for getting recipient emails
    def get_admin_emails(self) -> List[str]:
        """Retrieves a list of admin emails for notifications."""
        # In a real application, this would query a database or configuration
        return os.getenv("ADMIN_EMAILS", "admin@example.com").split(',')

    def get_user_email(self, user_id: str) -> Optional[str]:
        """Retrieves the email of a specific user."""
        # Placeholder: In a real app, query the user database
        # For now, return a dummy email or None
        if user_id == "test_user_id":
            return "test_user@example.com"
        return None