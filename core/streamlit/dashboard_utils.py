"""
Shared utilities for dashboard functionality.

Enhanced version with improved performance, error handling, and features
while maintaining full backward compatibility.
"""

import streamlit as st
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import logging
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import time
import functools
import hashlib
from datetime import datetime
from contextlib import contextmanager
import traceback
import json

import psutil

# Check if psutil is available
try:
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Import security utilities
from security.utils import sanitize_user_input as _sanitize_user_input, validate_file_path as _validate_file_path
from security.encryption import generate_secure_token as _generate_secure_token

logger = logging.getLogger(__name__)

# Performance monitoring decorator
def monitor_performance(operation_name: str = ""):
    """Decorator to monitor performance of dashboard operations."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            memory_before = _get_memory_usage() if HAS_PSUTIL else 0
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                memory_after = _get_memory_usage() if HAS_PSUTIL else 0
                
                # Log performance if debug mode is enabled
                if st.session_state.get('enable_debug_mode', False):
                    memory_delta = memory_after - memory_before if HAS_PSUTIL else 0
                    logger.debug(
                        f"Performance [{operation_name or func.__name__}]: "
                        f"{duration:.3f}s, Memory: {memory_delta:+.1f}MB"
                    )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.warning(f"Performance [{operation_name or func.__name__}]: {duration:.3f}s, Error: {e}")
                raise
        return wrapper
    return decorator

def _get_memory_usage() -> float:
    """Get current memory usage in MB."""
    if not HAS_PSUTIL:
        return 0.0
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0

@monitor_performance("page_setup")
def setup_page(
    title: str,
    logger_name: str = __name__,
    initialize_session: bool = True,
    sidebar_title: Optional[str] = None,
    enable_debug: bool = False
):
    """
    Enhanced page setup for all dashboard pages.
    
    Args:
        title: Page title to display
        logger_name: Logger name (usually __name__)
        initialize_session: Whether to initialize session state
        sidebar_title: Optional sidebar title
        enable_debug: Enable debug features
    
    Returns:
        Logger instance
    """
    from utils.logger import setup_logger
    
    # Check if we're running within the modular dashboard system
    is_modular_mode = st.session_state.get('dashboard_initialized', False)
    
    # Enhanced page title with debug info
    title_display = title
    if enable_debug and HAS_PSUTIL:
        memory_usage = _get_memory_usage()
        title_display += f" (Memory: {memory_usage:.1f}MB)"
    
    # Only set page title if NOT in modular mode (to avoid overriding main dashboard header)
    if not is_modular_mode:
        st.title(title_display)
    else:
        # In modular mode, just add a subheader for the specific page
        st.subheader(title_display)
    
    # Setup logger
    logger = setup_logger(logger_name)
    
    # Initialize session state if requested
    if initialize_session:
        initialize_dashboard_session_state()
        # Add debug mode to session state
        if 'enable_debug_mode' not in st.session_state:
            st.session_state['enable_debug_mode'] = enable_debug
    
    # Setup sidebar if title provided
    if sidebar_title:
        st.sidebar.header(sidebar_title)
        
        # Add debug controls if enabled
        if enable_debug:
            _render_debug_controls()
    
    return logger

def _render_debug_controls():
    """Render debug controls in sidebar."""
    with st.sidebar.expander("🔧 Debug Controls", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Cache", help="Clear Streamlit cache"):
                st.cache_data.clear()
                if hasattr(st, 'cache_resource'):
                    st.cache_resource.clear()
                st.success("Cache cleared")
        
        with col2:
            if st.button("Session Info", help="Show session state info"):
                st.write(f"Session keys: {len(st.session_state)}")
                if HAS_PSUTIL:
                    st.write(f"Memory: {_get_memory_usage():.1f}MB")
        
        # Performance metrics
        if HAS_PSUTIL:
            memory_usage = _get_memory_usage()
            st.metric("Memory Usage", f"{memory_usage:.1f} MB")

def safe_streamlit_metric(label: str, value: str, delta: Optional[str] = None) -> None:
    """
    Safely display a Streamlit metric with error handling.
    
    Args:
        label: Metric label
        value: Metric value
        delta: Optional delta value
    """
    try:
        if delta:
            st.metric(label, value, delta)
        else:
            st.metric(label, value)
    except Exception as e:
        logger.error(f"Error displaying metric {label}: {e}")
        st.text(f"{label}: {value}")

def enhanced_handle_streamlit_error(
    error: Exception, 
    context: str = "",
    show_traceback: bool = False,
    show_recovery: bool = True
) -> None:
    """
    Enhanced error handling with better context and recovery options.
    
    Args:
        error: Exception that occurred
        context: Additional context about where error occurred
        show_traceback: Whether to show full traceback in debug mode
        show_recovery: Whether to show recovery suggestions
    """
    error_id = hashlib.md5(f"{str(error)}{context}{time.time()}".encode()).hexdigest()[:8]
    error_msg = f"Error [{error_id}] in {context}: {str(error)}" if context else f"Error [{error_id}]: {str(error)}"
    logger.error(error_msg)
    
    st.error(f"🚨 {error_msg}")
    
    # Show technical details in debug mode
    if show_traceback and st.session_state.get('enable_debug_mode', False):
        with st.expander("🔍 Technical Details", expanded=False):
            st.code(traceback.format_exc())
    
    # Recovery suggestions
    if show_recovery:
        with st.expander("💡 Recovery Suggestions", expanded=False):
            recovery_suggestions = _get_recovery_suggestions(error, context)
            for suggestion in recovery_suggestions:
                st.info(suggestion)

def _get_recovery_suggestions(error: Exception, context: str) -> List[str]:
    """Get recovery suggestions based on error type and context."""
    suggestions = []
    
    error_type = type(error).__name__
    error_msg = str(error).lower()
    
    if "connection" in error_msg or "timeout" in error_msg:
        suggestions.append("🌐 Check your internet connection and try again")
        suggestions.append("⏱️ The service might be temporarily unavailable")
    
    elif "memory" in error_msg or error_type == "MemoryError":
        suggestions.append("💾 Try reducing the amount of data being processed")
        suggestions.append("🔄 Clear cache and restart the dashboard")
    
    elif "file" in error_msg or "permission" in error_msg:
        suggestions.append("📁 Check if the file exists and you have permission to access it")
        suggestions.append("🔒 Ensure the file is not open in another application")
    
    elif "key" in error_msg and context == "chart":
        suggestions.append("📊 Check that your data has the required columns (open, high, low, close)")
        suggestions.append("🔄 Try refreshing the data")
    
    else:
        suggestions.append("🔄 Try refreshing the page")
        suggestions.append("📋 Copy the error ID and report it if the problem persists")
    
    return suggestions

def handle_streamlit_error(error: Exception, context: str = "") -> None:
    """Original error handler for backward compatibility."""
    enhanced_handle_streamlit_error(error, context)

def enhanced_cache_key_builder(*args, include_session: bool = False, **kwargs) -> str:
    """Enhanced cache key builder with session state support."""
    key_parts = [str(arg) for arg in args] + [f"{k}={v}" for k, v in sorted(kwargs.items())]
    
    if include_session:
        # Include relevant session state for cache invalidation
        session_keys = ['user_id', 'session_id', 'data_version']
        session_parts = [
            f"session_{k}={st.session_state.get(k, 'none')}"
            for k in session_keys
            if k in st.session_state
        ]
        key_parts.extend(session_parts)
    
    key_string = "|".join(key_parts)
    return hashlib.sha256(key_string.encode()).hexdigest()[:16]

def cache_key_builder(*args, **kwargs) -> str:
    """Original cache key builder for backward compatibility."""
    return enhanced_cache_key_builder(*args, **kwargs)

def initialize_dashboard_session_state():
    """Initialize enhanced Streamlit session state variables."""
    defaults = {
        'dashboard_initialized': True,
        'error_count': 0,
        'last_update': datetime.now(),
        'user_preferences': {},
        'cache': {},
        'notifications': [],
        'current_page': None,
        'performance_metrics': [],
        'enable_debug_mode': False,
        'session_id': hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8],
        'memory_warnings': [],
        'last_memory_check': time.time(),
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

class DashboardStateManager:
    """
    Manages the dashboard's session state.
    Provides a consistent interface for accessing and modifying session data.
    """
    def __init__(self):
        # logger.debug("DashboardStateManager initialized.") # Assuming logger is available if uncommented
        pass

    def initialize_session_state(self):
        """
        Initializes the core session state variables for the dashboard.
        Leverages the existing global function.
        """
        initialize_dashboard_session_state() # Call the existing function in this file

    def get_state(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a value from the session state.

        Args:
            key: The key for the session state variable.
            default: The default value to return if the key is not found.

        Returns:
            The value from session state or the default.
        """
        return st.session_state.get(key, default)

    def set_state(self, key: str, value: Any) -> None:
        """
        Sets a value in the session state.

        Args:
            key: The key for the session state variable.
            value: The value to set.
        """
        st.session_state[key] = value

@contextmanager
def session_state_backup():
    """Context manager to backup and restore session state on errors."""
    backup = dict(st.session_state)
    try:
        yield
    except Exception:
        # Restore session state on error
        st.session_state.clear()
        st.session_state.update(backup)
        raise

def validate_session_state() -> bool:
    """Validate session state integrity and memory usage."""
    required_keys = ['dashboard_initialized', 'session_id']
    
    for key in required_keys:
        if key not in st.session_state:
            logger.warning(f"Missing required session state key: {key}")
            return False
    
    # Check memory usage if psutil is available
    if HAS_PSUTIL:
        memory_usage = _get_memory_usage()
        memory_limit = 500.0  # Default 500MB limit
        
        if memory_usage > memory_limit:
            logger.warning(f"Memory usage ({memory_usage:.1f}MB) exceeds limit ({memory_limit}MB)")
            st.session_state.setdefault('memory_warnings', []).append({
                'timestamp': datetime.now(),
                'usage': memory_usage,
                'limit': memory_limit
            })
            return False
    
    return True

# =============================================================================
# Security and Input Validation Functions
# =============================================================================

def sanitize_user_input(input_text: str, max_length: int = 1000, allow_html: bool = False) -> str:
    """
    Sanitize user input to prevent XSS and other security issues.
    
    .. deprecated:: 
        Use security.utils.sanitize_user_input instead.
    
    Args:
        input_text: The input text to sanitize
        max_length: Maximum allowed length
        allow_html: Whether to allow HTML tags
    
    Returns:
        Sanitized input text
    """
    import warnings
    warnings.warn(
        "dashboard_utils.sanitize_user_input is deprecated. Use security.utils.sanitize_user_input instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _sanitize_user_input(input_text, max_length, allow_html)


def validate_file_path(file_path: Union[str, Path], allowed_extensions: Optional[List[str]] = None) -> bool:
    """
    Validate if a file path is safe and allowed.
    
    .. deprecated::
        Use security.utils.validate_file_path instead.
    
    Args:
        file_path: Path to validate
        allowed_extensions: List of allowed file extensions (e.g., ['.csv', '.json'])
    
    Returns:
        True if path is valid and safe
    """
    import warnings
    warnings.warn(
        "dashboard_utils.validate_file_path is deprecated. Use security.utils.validate_file_path instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _validate_file_path(file_path, allowed_extensions)


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token.
    
    .. deprecated::
        Use security.encryption.generate_secure_token instead.
    """
    import warnings
    warnings.warn(
        "dashboard_utils.generate_secure_token is deprecated. Use security.encryption.generate_secure_token instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _generate_secure_token(length)


# =============================================================================
# Advanced File Operations
# =============================================================================

def safe_json_load(file_path: Union[str, Path], default: Any = None) -> Any:
    """
    Safely load JSON data from a file with error handling.
    
    Args:
        file_path: Path to JSON file
        default: Default value to return on error
    
    Returns:
        Loaded JSON data or default value
    """
    try:
        if not validate_file_path(file_path, ['.json']):
            logger.warning(f"Invalid file path for JSON load: {file_path}")
            return default
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return default


def safe_json_save(data: Any, file_path: Union[str, Path], indent: int = 2) -> bool:
    """
    Safely save data to JSON file with error handling.
    
    Args:
        data: Data to save
        file_path: Path to save JSON file
        indent: JSON indentation
    
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        
        logger.info(f"Successfully saved JSON to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        return False


def safe_csv_operations(df: pd.DataFrame, file_path: Union[str, Path], operation: str = 'save') -> Union[pd.DataFrame, bool]:
    """
    Safely perform CSV operations with enhanced error handling.
    
    Args:
        df: DataFrame to save (for save operation) or None (for load operation)
        file_path: Path to CSV file
        operation: 'save' or 'load'
    
    Returns:
        DataFrame (for load) or success boolean (for save)
    """
    try:
        path = Path(file_path)
        
        if operation == 'save':
            if df is None or df.empty:
                logger.warning("Cannot save empty DataFrame")
                return False
            
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            logger.info(f"Successfully saved CSV to {file_path}")
            return True
            
        elif operation == 'load':
            if not validate_file_path(file_path, ['.csv']):
                logger.warning(f"Invalid CSV file path: {file_path}")
                return pd.DataFrame()
            
            df = pd.read_csv(path)
            logger.info(f"Successfully loaded CSV from {file_path}")
            return df
        else:
            logger.warning(f"Invalid operation: {operation}")
            return False if operation == 'save' else pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error in CSV {operation} operation for {file_path}: {e}")
        if operation == 'save':
            return False
        else:  # load operation
            return pd.DataFrame()


# =============================================================================
# Enhanced Notification System
# =============================================================================

def create_advanced_notification(
    message: str,
    notification_type: str = "info",
    dismissible: bool = True,
    auto_close: Optional[int] = None,
    actions: Optional[List[Dict[str, Any]]] = None
) -> None:
    """
    Create an advanced notification with enhanced features.
    
    Args:
        message: Notification message
        notification_type: Type of notification (info, success, warning, error)
        dismissible: Whether notification can be dismissed
        auto_close: Auto-close timeout in seconds
        actions: List of action buttons with callbacks
    """
    # Create unique notification ID
    notification_id = f"notification_{generate_secure_token(8)}"
    
    # Store notification in session state
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []
    
    notification_data = {
        'id': notification_id,
        'message': sanitize_user_input(message),
        'type': notification_type,
        'dismissible': dismissible,
        'auto_close': auto_close,
        'actions': actions or [],
        'timestamp': datetime.now(),
        'dismissed': False
    }
    
    st.session_state.notifications.append(notification_data)
    
    # Display notification
    _render_notification(notification_data)


def _render_notification(notification: Dict[str, Any]) -> None:
    """Render a notification with appropriate styling."""
    if notification.get('dismissed', False):
        return
    
    notification_type = notification.get('type', 'info')
    message = notification.get('message', '')
    
    # Choose appropriate Streamlit method based on type
    if notification_type == 'success':
        st.success(message)
    elif notification_type == 'warning':
        st.warning(message)
    elif notification_type == 'error':
        st.error(message)
    else:
        st.info(message)
    
    # Add action buttons if specified
    actions = notification.get('actions', [])
    if actions:
        cols = st.columns(len(actions))
        for i, action in enumerate(actions):
            with cols[i]:
                if st.button(action.get('label', 'Action'), key=f"{notification['id']}_action_{i}"):
                    callback = action.get('callback')
                    if callback and callable(callback):
                        callback()


def dismiss_notification(notification_id: str) -> None:
    """Dismiss a specific notification."""
    if 'notifications' in st.session_state:
        for notification in st.session_state.notifications:
            if notification['id'] == notification_id:
                notification['dismissed'] = True
                break


def clear_all_notifications() -> None:
    """Clear all notifications from session state."""
    if 'notifications' in st.session_state:
        st.session_state.notifications = []


# =============================================================================
# Enhanced Caching System
# =============================================================================

def create_tiered_cache_key(*args, cache_level: str = "user", **kwargs) -> str:
    """
    Create a hierarchical cache key for multi-level caching.
    
    Args:
        *args: Positional arguments for cache key
        cache_level: Cache level (global, user, session)
        **kwargs: Keyword arguments for cache key
    
    Returns:
        Hierarchical cache key
    """
    base_key = cache_key_builder(*args, **kwargs)
    
    if cache_level == "global":
        return f"global_{base_key}"
    elif cache_level == "user":
        user_id = st.session_state.get('user_id', 'anonymous')
        return f"user_{user_id}_{base_key}"
    elif cache_level == "session":
        session_id = st.session_state.get('session_id', generate_secure_token(8))
        if 'session_id' not in st.session_state:
            st.session_state.session_id = session_id
        return f"session_{session_id}_{base_key}"
    else:
        return base_key


@contextmanager
def cache_context(cache_key: str, ttl: Optional[int] = None):
    """
    Context manager for caching operations with automatic cleanup.
    
    Args:
        cache_key: Cache key
        ttl: Time to live in seconds
    """
    start_time = time.time()
    try:
        yield cache_key
    finally:
        elapsed_time = time.time() - start_time
        logger.debug(f"Cache operation completed in {elapsed_time:.3f}s for key: {cache_key}")

class EnhancedDashboardBase:
    """
    A base class for enhanced Streamlit dashboards, providing common
    functionalities like page configuration and logging.
    """
    def __init__(self, page_title: str, page_icon: str = "⚙️"):
        """
        Initializes the EnhancedDashboardBase.

        Args:
            page_title (str): The title of the dashboard page.
            page_icon (str, optional): The icon for the dashboard page. Defaults to "⚙️".
        """
        self.page_title = page_title
        self.page_icon = page_icon
        self._configure_page()
        logger.info(f"EnhancedDashboardBase initialized for page: '{self.page_title}'")

    def _configure_page(self):
        """
        Configures the Streamlit page with the title and icon.
        This method is typically called during initialization.
        """
        try:
            st.set_page_config(page_title=self.page_title, page_icon=self.page_icon, layout="wide")
            logger.debug(f"Page configured for '{self.page_title}' with icon '{self.page_icon}'.")
        except Exception as e:
            # This can happen if called after the first st.command, though ideally init is early.
            logger.warning(f"Could not set page config for '{self.page_title}': {e}")

    def display_header(self):
        """
        Displays the main header for the dashboard page.
        """
        st.title(f"{self.page_icon} {self.page_title}")
        st.sidebar.success(f"Navigated to {self.page_title}")
        logger.debug(f"Header displayed for '{self.page_title}'.")

    def show_error(self, message: str, exception: Optional[Exception] = None):
        """
        Displays an error message on the Streamlit page and logs it.

        Args:
            message (str): The error message to display.
            exception (Optional[Exception], optional): The exception object, if any. Defaults to None.
        """
        st.error(message)
        if exception:
            logger.error(f"{message} - Exception: {exception}", exc_info=True)
        else:
            logger.error(message)

    def show_warning(self, message: str):
        """
        Displays a warning message on the Streamlit page and logs it.

        Args:
            message (str): The warning message to display.
        """
        st.warning(message)
        logger.warning(message)

    def show_info(self, message: str):
        """
        Displays an informational message on the Streamlit page and logs it.

        Args:
            message (str): The informational message to display.
        """
        st.info(message)
        logger.info(message)

    def show_success(self, message: str):
        """
        Displays a success message on the Streamlit page and logs it.

        Args:
            message (str): The success message to display.
        """
        st.success(message)
        logger.info(message)