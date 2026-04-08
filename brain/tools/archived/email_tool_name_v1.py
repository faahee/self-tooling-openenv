import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import requests
import pyperclip
import datetime
import psutil
import time

def send_email(subject: str, message: str) -> dict:
    """
    Send an email to maahirfaheem666@gmail.com.

    Args:
        subject (str): The subject of the email.
        message (str): The body of the email.

    Returns:
        dict: A dictionary containing 'success', 'result', and 'error'.
    """

    try:
        # Create a new email message
        msg = MIMEMultipart()
        msg['From'] = 'your-email@gmail.com'
        msg['To'] = 'maahirfaheem666@gmail.com'
        msg['Subject'] = subject

        # Attach the message to the email
        msg.attach(MIMEText(message, 'plain'))

        # Send the email using smtplib
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(msg['From'], 'your-password')
        text = msg.as_string()
        server.sendmail(msg['From'], msg['To'], text)
        server.quit()

        # Return success with the email content
        return {'success': True, 'result': msg.as_string(), 'error': None}

    except Exception as e:
        # Handle any exceptions that occur during the process
        return {'success': False, 'result': None, 'error': str(e)}