# notifier.py
"""
Notification module to send alerts via Email, SMS (Twilio), and Slack.

Classes:
    - EmailNotifier(smtp_server, smtp_port, username, password)
    - SMSNotifier(account_sid, auth_token, from_number)
    - SlackNotifier(webhook_url)

Methods:
    - send_email(to: str, subject: str, body: str)
    - send_sms(to: str, message: str)
    - send_slack(channel: str, message: str)
"""
import smtplib
from email.mime.text import MIMEText
from typing import Optional

try:
    from twilio.rest import Client as TwilioClient
except ImportError:
    TwilioClient = None

import requests


class EmailNotifier:
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str):
        self.server = smtp_server
        self.port = smtp_port
        self.username = username
        self.password = password

    def send_email(self, to: str, subject: str, body: str):
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = self.username
        msg['To'] = to
        with smtplib.SMTP(self.server, self.port) as s:
            s.starttls()
            s.login(self.username, self.password)
            s.send_message(msg)


class SMSNotifier:
    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        if TwilioClient is None:
            raise ImportError("twilio library is required for SMSNotifier.")
        self.client = TwilioClient(account_sid, auth_token)
        self.from_number = from_number

    def send_sms(self, to: str, message: str):
        self.client.messages.create(
            to=to,
            from_=self.from_number,
            body=message
        )


class SlackNotifier:
    def __init__(self, webhook_url: str, default_channel: Optional[str] = None):
        self.webhook_url = webhook_url
        self.default_channel = default_channel

    def send_slack(self, message: str, channel: Optional[str] = None):
        payload = {
            'text': message
        }
        if channel or self.default_channel:
            payload['channel'] = channel or self.default_channel
        requests.post(self.webhook_url, json=payload)
