"""
Standardized logging configuration for dashboard pages.
"""
import logging
import os
from pathlib import Path
from utils.logger import setup_logger

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