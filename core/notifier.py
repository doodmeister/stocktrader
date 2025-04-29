"""Notifier Utility

Handles user notifications, future extensible (email, SMS, Slack, etc.)
"""

import logging

logger = logging.getLogger(__name__)

class Notifier:
    """Simple notifier placeholder."""

    def send(self, message: str) -> None:
        """Send a notification (placeholder)."""
        try:
            logger.info(f"Notifier: {message}")
            # In production, extend here (email, Slack webhook, etc.)
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")