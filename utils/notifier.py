'''
notifier.py

Notification module to send alerts via Email, SMS (Twilio), and Slack.

Classes:
    - EmailNotifier(smtp_server, smtp_port, username, password)
    - SMSNotifier(account_sid, auth_token, from_number)
    - SlackNotifier(webhook_url, default_channel=None)
    - Notifier()

Usage:
    notifier = Notifier()
    notifier.send_order_notification(order_result)
'''
import os
import smtplib
from email.mime.text import MIMEText
from typing import Optional, Any, Dict

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

    def send_email(self, recipient, subject, message):
        """Send an email notification."""
        sender = self.username
        password = self.password
        from email.mime.text import MIMEText
        import smtplib
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = recipient
        try:
            server = smtplib.SMTP_SSL(self.server, self.port)
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
            server.quit()
            return True
        except Exception as e:
            print(f"Email sending error: {e}")
            return False


class SMSNotifier:
    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        if TwilioClient is None:
            raise ImportError("twilio library is required for SMSNotifier.")
        self.client = TwilioClient(account_sid, auth_token)
        self.from_number = from_number

    def send_sms(self, to: str, message: str) -> None:
        try:
            self.client.messages.create(
                to=to,
                from_=self.from_number,
                body=message
            )
        except Exception as e:
            print(f"SMS sending error: {e}")


class SlackNotifier:
    def __init__(self, webhook_url: str, default_channel: Optional[str] = None):
        self.webhook_url = webhook_url
        self.default_channel = default_channel

    def send_slack(self, message: str, channel: Optional[str] = None) -> None:
        payload: Dict[str, Any] = {'text': message}
        if channel or self.default_channel:
            payload['channel'] = channel or self.default_channel
        response = requests.post(self.webhook_url, json=payload)
        response.raise_for_status()


class Notifier:
    """
    Facade for sending order notifications via configured channels: Email, SMS, and Slack.
    Reads configuration from environment variables.
    """

    def __init__(self) -> None:
        # Email config
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port   = os.getenv("SMTP_PORT")
        smtp_user   = os.getenv("SMTP_USER")
        smtp_pass   = os.getenv("SMTP_PASS")

        # SMS config
        twilio_sid      = os.getenv("TWILIO_ACCOUNT_SID")
        twilio_token    = os.getenv("TWILIO_AUTH_TOKEN")
        twilio_from_num = os.getenv("TWILIO_FROM_NUMBER")
        twilio_to_num   = os.getenv("TWILIO_TO_NUMBER")

        # Slack config
        slack_url = os.getenv("SLACK_WEBHOOK_URL")

        # Instantiate notifiers if configs are present
        self.email_notifier = (
            EmailNotifier(smtp_server, int(smtp_port), smtp_user, smtp_pass)
            if smtp_server and smtp_port and smtp_user and smtp_pass
            else None
        )
        self.sms_notifier = (
            SMSNotifier(twilio_sid, twilio_token, twilio_from_num)
            if twilio_sid and twilio_token and twilio_from_num and twilio_to_num
            else None
        )
        self.sms_to_number = twilio_to_num
        self.slack_notifier = (
            SlackNotifier(slack_url)
            if slack_url
            else None
        )

    def send_order_notification(self, order_result: Dict[str, Any]) -> None:
        """
        Send a unified order notification to all configured channels.
        """
        # Build message text
        symbol = order_result.get('symbol')
        side   = order_result.get('side')
        qty    = order_result.get('quantity')
        price  = order_result.get('filled_price')
        ts     = order_result.get('timestamp')

        message = (
            f"Order Executed:\n"
            f"Symbol: {symbol}\n"
            f"Side:   {side}\n"
            f"Qty:    {qty}\n"
            f"Price:  {price}\n"
            f"Time:   {ts}"
        )

        # Email
        if self.email_notifier:
            try:
                self.email_notifier.send_email(
                    self.email_notifier.username,
                    "StockTrader Notification",
                    message
                )
            except Exception as e:
                # Log or ignore
                pass

        # SMS
        if self.sms_notifier and self.sms_to_number:
            try:
                self.sms_notifier.send_sms(self.sms_to_number, message)
            except Exception:
                pass

        # Slack
        if self.slack_notifier:
            try:
                self.slack_notifier.send_slack(message)
            except Exception:
                pass

    def send_notification(self, subject: str, message: str) -> None:
        """
        Send a general notification to all configured channels (Email, SMS, Slack).
        """
        # Email
        if self.email_notifier:
            try:
                self.email_notifier.send_email(
                    self.email_notifier.username,
                    subject,
                    message
                )
            except Exception:
                pass
        # SMS
        if self.sms_notifier and self.sms_to_number:
            try:
                self.sms_notifier.send_sms(self.sms_to_number, f"{subject}: {message}")
            except Exception:
                pass
        # Slack
        if self.slack_notifier:
            try:
                self.slack_notifier.send_slack(f"{subject}: {message}")
            except Exception:
                pass
