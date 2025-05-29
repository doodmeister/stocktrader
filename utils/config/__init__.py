"""
Configuration utilities for the StockTrader dashboard.

This module provides configuration functions and project path utilities.
"""

from pathlib import Path


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path: The project root directory path
    """
    # Get the directory containing this config module
    config_dir = Path(__file__).parent  # utils/config/
    utils_dir = config_dir.parent       # utils/
    project_root = utils_dir.parent     # project root
    
    return project_root


def get_data_directory() -> Path:
    """
    Get the data directory path.
    
    Returns:
        Path: The data directory path
    """
    return get_project_root() / "data"


def get_logs_directory() -> Path:
    """
    Get the logs directory path.
    
    Returns:
        Path: The logs directory path
    """
    return get_project_root() / "logs"


def get_models_directory() -> Path:
    """
    Get the models directory path.
    
    Returns:
        Path: The models directory path
    """
    return get_project_root() / "models"


def get_config_path(filename: str) -> Path:
    """
    Get path to a configuration file in the project root.
    
    Args:
        filename: Name of the configuration file
        
    Returns:
        Path: Full path to the configuration file
    """
    return get_project_root() / filename