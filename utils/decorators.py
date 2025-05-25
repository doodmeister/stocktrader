# utils/decorators.py
import streamlit as st
import functools
import logging

logger = logging.getLogger(__name__)

def handle_exceptions(func):
    """Decorator to handle exceptions in Streamlit applications."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            st.error(f"An error occurred in {func.__name__}: {e}")
            return None
    return wrapper

def handle_dashboard_exceptions(error_context: str = "operation"):
    """Decorator factory for dashboard-specific exception handling."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {error_context} ({func.__name__}): {e}")
                st.error(f"An error occurred during {error_context}: {e}")
                return None
        return wrapper
    return decorator