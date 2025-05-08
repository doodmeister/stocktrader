"""Logging configuration for the dashboard."""

import logging
from logging.handlers import RotatingFileHandler
from .config import DashboardConfig

def setup_logger(name: str) -> logging.Logger:
    """Configure and return a logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # File handler
    file_handler = RotatingFileHandler(
        DashboardConfig.LOG_FILE,
        maxBytes=DashboardConfig.LOG_MAX_SIZE,
        backupCount=DashboardConfig.LOG_BACKUP_COUNT
    )
    file_handler.setFormatter(formatter)
    
    # Stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger