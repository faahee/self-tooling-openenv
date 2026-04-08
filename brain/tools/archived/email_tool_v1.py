import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional
import pyperclip
import requests
import psutil
import time

def email_tool(
    sender_email: str,
    receiver_email: str,
    subject: str,
    message: str,
    password: str
) -> dict:
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject

        body = message
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()

        return {
            "success": True,
            "result": None,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "error": str(e)
        }

# Example usage:
email_tool(
    sender_email="maahirfaheem666@gmail.com",
    receiver_email="maahirfaheem666@gmail.com",
    subject="Test Email",
    message="This is a test email.",
    password="your_password"
).get("result")