"""Unified logging configuration for the StockTrader project."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import os
import threading

# Global lock for thread-safe logging setup
_logger_lock = threading.Lock()
_configured_loggers = set()

def setup_logger(
    name: str = __name__,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    use_rotation: bool = False  # Disable rotation by default for Streamlit
) -> logging.Logger:
    """
    Set up a logger with optional file output and rotation.
    
    Args:
        name: Logger name (usually __name__)
        log_file: Optional log file path
        level: Logging level
        format_string: Custom format string
        max_bytes: Max bytes before rotation (only if use_rotation=True)
        backup_count: Number of backup files to keep
        use_rotation: Whether to use rotating file handler (disabled for Streamlit)
    
    Returns:
        Configured logger instance
    """
    with _logger_lock:
        # Check if logger is already configured
        if name in _configured_loggers:
            return logging.getLogger(name)
        
        logger = logging.getLogger(name)
        
        # Clear any existing handlers to avoid duplicates
        logger.handlers.clear()
        logger.setLevel(level)
        
        # Default format
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        formatter = logging.Formatter(format_string)
        
        # Console handler (always add)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_file:
            try:
                # Ensure log directory exists
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Use simple FileHandler for Streamlit to avoid permission issues
                if use_rotation and not _is_streamlit_context():
                    # Only use rotation outside of Streamlit
                    file_handler = logging.handlers.RotatingFileHandler(
                        log_file,
                        maxBytes=max_bytes,
                        backupCount=backup_count,
                        encoding='utf-8'
                    )
                else:
                    # Simple file handler for Streamlit context
                    file_handler = logging.FileHandler(
                        log_file,
                        mode='a',
                        encoding='utf-8'
                    )
                
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
            except (OSError, PermissionError) as e:
                # If file logging fails, continue with console only
                logger.warning(f"Could not set up file logging to {log_file}: {e}")
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        # Mark as configured
        _configured_loggers.add(name)
        
        return logger

def _is_streamlit_context() -> bool:
    """Check if we're running in a Streamlit context."""
    try:
        import streamlit as st
        # Check if Streamlit session state exists
        return hasattr(st, 'session_state')
    except ImportError:
        return False

def get_logger(name: str = __name__) -> logging.Logger:
    """Get an existing logger or create a basic one."""
    return logging.getLogger(name)

def configure_dashboard_logging(
    log_dir: Path = Path("logs"),
    level: int = logging.INFO,
    enable_file_logging: bool = True
) -> logging.Logger:
    """
    Configure logging specifically for the dashboard application.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
        enable_file_logging: Whether to enable file logging
    
    Returns:
        Main dashboard logger
    """
    # Create unique log file for this process
    process_id = os.getpid()
    log_file = None
    
    if enable_file_logging:
        log_file = log_dir / f"dashboard_{process_id}.log"
    
    return setup_logger(
        name="dashboard_main",
        log_file=str(log_file) if log_file else None,
        level=level,
        use_rotation=False  # Disable rotation for Streamlit
    )

# Dashboard-specific logging functions
def get_dashboard_logger(
    module_name: str,
    enable_file_logging: bool = True,
    log_level: int = logging.INFO
) -> logging.Logger:
    """
    Get a standardized logger for dashboard pages.
    
    Args:
        module_name: Name of the module (e.g., 'data_dashboard', 'realtime_dashboard')
        enable_file_logging: Whether to enable file logging
        log_level: Logging level
    
    Returns:
        Configured logger instance
    """
    # Extract clean module name from __name__ if needed
    if '.' in module_name:
        clean_name = module_name.split('.')[-1]
    else:
        clean_name = module_name
    
    # Remove common prefixes/suffixes for cleaner names
    clean_name = clean_name.replace('dashboard_pages.', '').replace('_dashboard', '')
    
    # Create process-specific logger name
    process_id = os.getpid()
    logger_name = f"{clean_name}_{process_id}"
    
    # Setup log file path if enabled
    log_file = None
    if enable_file_logging:
        log_file = f"logs/{logger_name}.log"
    
    return setup_logger(
        name=logger_name,
        log_file=log_file,
        level=log_level,
        use_rotation=False  # Disable rotation for Streamlit
    )

def get_page_logger(page_file: str) -> logging.Logger:
    """
    Convenience function to get logger based on page filename.
    
    Args:
        page_file: The __file__ variable from the calling module
    
    Returns:
        Configured logger
    """
    page_name = Path(page_file).stem
    return get_dashboard_logger(page_name)

def get_streamlit_logger(
    name: Optional[str] = None,
    enable_file_logging: bool = False
) -> logging.Logger:
    """
    Get a logger optimized for Streamlit applications.
    
    Args:
        name: Logger name (auto-generated if None)
        enable_file_logging: Whether to enable file logging (disabled by default for Streamlit)
    
    Returns:
        Streamlit-optimized logger
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'streamlit_app')
    
    return get_dashboard_logger(
        module_name=name,
        enable_file_logging=enable_file_logging,
        log_level=logging.INFO
    )

def disable_file_logging_globally():
    """
    Disable file logging for all future loggers.
    Useful for deployment environments where file logging causes issues.
    """
    global _default_enable_file_logging
    _default_enable_file_logging = False

def enable_file_logging_globally():
    """
    Enable file logging for all future loggers.
    """
    global _default_enable_file_logging
    _default_enable_file_logging = True

# Global setting for file logging (can be overridden per logger)
_default_enable_file_logging = True