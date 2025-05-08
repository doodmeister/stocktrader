"""Unified logging configuration for the StockTrader project."""

import logging
import os
from logging.handlers import RotatingFileHandler
from utils.config.config import DashboardConfig

_DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_DEFAULT_LEVEL = os.environ.get("STOCKTRADER_LOG_LEVEL", "INFO").upper()

def _get_log_level():
    try:
        return getattr(logging, _DEFAULT_LEVEL)
    except AttributeError:
        return logging.INFO

def _ensure_logs_dir():
    log_dir = os.path.dirname(DashboardConfig.LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

def setup_logger(name: str = None, level: int = None) -> logging.Logger:
    """
    Configure and return a logger instance.
    Ensures handlers are only added once per logger.
    """
    _ensure_logs_dir()  # Ensure logs directory exists

    logger = logging.getLogger(name)
    log_level = level if level is not None else _get_log_level()
    logger.setLevel(log_level)

    if not getattr(logger, "_stocktrader_handlers_set", False):
        formatter = logging.Formatter(_DEFAULT_FORMAT)

        # File handler
        file_handler = RotatingFileHandler(
            DashboardConfig.LOG_FILE,  # <-- Uses log path from config.py
            maxBytes=DashboardConfig.LOG_MAX_SIZE,
            backupCount=DashboardConfig.LOG_BACKUP_COUNT,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

        # Stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(log_level)
        logger.addHandler(stream_handler)

        logger._stocktrader_handlers_set = True

    return logger

def init_root_logger(level: int = None):
    """
    Initialize the root logger for legacy modules using logging.getLogger(__name__).
    Should be called once at program entry point.
    """
    _ensure_logs_dir()  # Ensure logs directory exists

    root_logger = logging.getLogger()
    log_level = level if level is not None else _get_log_level()
    root_logger.setLevel(log_level)

    # Remove all existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(_DEFAULT_FORMAT)

    file_handler = RotatingFileHandler(
        DashboardConfig.LOG_FILE,  # <-- Uses log path from config.py
        maxBytes=DashboardConfig.LOG_MAX_SIZE,
        backupCount=DashboardConfig.LOG_BACKUP_COUNT,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    root_logger.addHandler(stream_handler)

# Optionally, initialize root logger on import for scripts that don't call setup_logger
if os.environ.get("STOCKTRADER_AUTO_INIT_LOGGING", "1") == "1":
    init_root_logger()