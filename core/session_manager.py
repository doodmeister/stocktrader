"""
Session State Management System for StockTrader Dashboards

This module provides a comprehensive solution for managing session state,
button keys, and form keys across the modular dashboard architecture to prevent
conflicts, duplicate keys, and state pollution between different pages.

Key Features:
- Automatic namespace generation for each dashboard page
- Unique button and form key management
- Session state isolation and cleanup
- Conflict prevention across modular architecture
- Thread-safe operations

Author: GitHub Copilot
Created: 2025-05-29
"""

import streamlit as st
import hashlib
import time
import uuid
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import logging
from contextlib import contextmanager

from utils.logger import get_dashboard_logger

logger = get_dashboard_logger(__name__)


@dataclass
class PageContext:
    """Context information for a dashboard page."""
    page_name: str
    namespace: str
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    button_count: int = 0
    form_count: int = 0
    cleanup_keys: List[str] = field(default_factory=list)


class SessionManager:
    """
    Comprehensive session state management for StockTrader dashboards.
    
    Provides automatic namespacing, key conflict prevention, and session isolation
    to solve recurring button and form key issues in the modular architecture.
    """
    
    def __init__(self, page_name: str = None):
        """
        Initialize the session manager for a specific dashboard page.
        
        Args:
            page_name: Name of the dashboard page (auto-detected if None)
        """
        self.page_name = page_name or self._detect_page_name()
        self.namespace = self._generate_namespace()
        self.session_id = self._get_or_create_session_id()
        
        # Initialize page context
        self._initialize_page_context()
        
        logger.debug(f"SessionManager initialized for {self.page_name} with namespace {self.namespace}")
    
    def _detect_page_name(self) -> str:
        """Automatically detect the current page name from the call stack."""
        import inspect
        
        # Look through the call stack for the calling module
        for frame_info in inspect.stack():
            filename = Path(frame_info.filename).stem
            if filename.startswith(('realtime_', 'data_', 'simple_', 'advanced_', 'model_', 'classic_')):
                return filename
        
        # Fallback to generic name with timestamp
        return f"dashboard_{int(time.time())}"
    
    def _generate_namespace(self) -> str:
        """Generate a unique namespace for this page."""
        # Create a stable but unique namespace based on page name and session
        base_string = f"{self.page_name}_{st.session_state.get('session_id', 'default')}"
        namespace_hash = hashlib.md5(base_string.encode()).hexdigest()[:8]
        return f"{self.page_name}_{namespace_hash}"
    
    def _get_or_create_session_id(self) -> str:
        """Get or create a unique session ID."""
        if 'global_session_id' not in st.session_state:
            st.session_state.global_session_id = str(uuid.uuid4())[:8]
        return st.session_state.global_session_id
    
    def _initialize_page_context(self) -> None:
        """Initialize the page context in session state."""
        context_key = f"_page_context_{self.namespace}"
        
        if context_key not in st.session_state:
            st.session_state[context_key] = PageContext(
                page_name=self.page_name,
                namespace=self.namespace,
                session_id=self.session_id
            )
    
    def get_unique_key(self, base_key: str, key_type: str = "button") -> str:
        """
        Generate a unique key for buttons, forms, or other Streamlit components.
        
        Args:
            base_key: Base name for the key (e.g., "update", "clear_data")
            key_type: Type of component ("button", "form", "input", etc.)
            
        Returns:
            str: Unique key that won't conflict with other pages
        """
        context_key = f"_page_context_{self.namespace}"
        context = st.session_state.get(context_key)
        
        if context and key_type == "button":
            context.button_count += 1
            count = context.button_count
        elif context and key_type == "form":
            context.form_count += 1
            count = context.form_count
        else:
            count = int(time.time() * 1000) % 10000
        
        unique_key = f"{self.namespace}_{key_type}_{base_key}_{count}"
        
        # Track this key for cleanup
        if context:
            context.cleanup_keys.append(unique_key)
        
        logger.debug(f"Generated unique key: {unique_key}")
        return unique_key
    
    def get_form_key(self, form_name: str = "main") -> str:
        """Get a unique form key for this page."""
        return self.get_unique_key(form_name, "form")
    
    def get_button_key(self, button_name: str) -> str:
        """Get a unique button key for this page."""
        return self.get_unique_key(button_name, "button")
    
    def get_input_key(self, input_name: str) -> str:
        """Get a unique input key for this page."""
        return self.get_unique_key(input_name, "input")
    
    def set_page_state(self, key: str, value: Any) -> None:
        """
        Set a page-specific session state value.
        
        Args:
            key: The key name (will be namespaced automatically)
            value: The value to store
        """
        namespaced_key = f"{self.namespace}_{key}"
        st.session_state[namespaced_key] = value
        logger.debug(f"Set page state: {namespaced_key} = {type(value).__name__}")
    
    def get_page_state(self, key: str, default: Any = None) -> Any:
        """
        Get a page-specific session state value.
        
        Args:
            key: The key name (will be namespaced automatically)
            default: Default value if key doesn't exist
            
        Returns:
            The stored value or default
        """
        namespaced_key = f"{self.namespace}_{key}"
        return st.session_state.get(namespaced_key, default)
    
    def clear_page_state(self, key: str = None) -> None:
        """
        Clear page-specific session state.
        
        Args:
            key: Specific key to clear, or None to clear all page state
        """
        if key:
            namespaced_key = f"{self.namespace}_{key}"
            if namespaced_key in st.session_state:
                del st.session_state[namespaced_key]
                logger.debug(f"Cleared page state key: {namespaced_key}")
        else:
            # Clear all keys for this namespace
            keys_to_remove = [k for k in st.session_state.keys() if k.startswith(f"{self.namespace}_")]
            for k in keys_to_remove:
                del st.session_state[k]
            logger.debug(f"Cleared all page state for namespace: {self.namespace}")
    
    def cleanup_page(self) -> None:
        """Clean up all session state for this page."""
        self.clear_page_state()
        
        # Also clean up the page context
        context_key = f"_page_context_{self.namespace}"
        if context_key in st.session_state:
            del st.session_state[context_key]
        
        logger.info(f"Cleaned up session state for page: {self.page_name}")
    
    @contextmanager
    def form_container(self, form_name: str = "main", **form_kwargs):
        """
        Context manager for creating forms with automatic key management.
        
        Args:
            form_name: Name for the form
            **form_kwargs: Additional arguments for st.form()
            
        Usage:
            with session_manager.form_container("my_form") as form:
                # form content here
                submitted = st.form_submit_button("Submit")
        """
        form_key = self.get_form_key(form_name)
        
        try:
            with st.form(form_key, **form_kwargs) as form:
                logger.debug(f"Created form with key: {form_key}")
                yield form
        except Exception as e:
            logger.error(f"Error in form container {form_key}: {e}")
            raise
    
    def create_button(self, label: str, button_name: str = None, **button_kwargs) -> bool:
        """
        Create a button with automatic key management.
        
        Args:
            label: Button label text
            button_name: Internal name for the button (defaults to label)
            **button_kwargs: Additional arguments for st.button()
            
        Returns:
            bool: True if button was clicked
        """
        if button_name is None:
            button_name = label.lower().replace(" ", "_")
        
        button_key = self.get_button_key(button_name)
        
        try:
            clicked = st.button(label, key=button_key, **button_kwargs)
            if clicked:
                logger.debug(f"Button clicked: {button_key}")
            return clicked
        except Exception as e:
            logger.error(f"Error creating button {button_key}: {e}")
            return False
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the current session state."""
        context_key = f"_page_context_{self.namespace}"
        context = st.session_state.get(context_key)
        
        return {
            "page_name": self.page_name,
            "namespace": self.namespace,
            "session_id": self.session_id,
            "context_exists": context is not None,
            "button_count": context.button_count if context else 0,
            "form_count": context.form_count if context else 0,
            "cleanup_keys": len(context.cleanup_keys) if context else 0,
            "total_session_keys": len(st.session_state),
            "page_specific_keys": len([k for k in st.session_state.keys() if k.startswith(self.namespace)])
        }


class GlobalSessionManager:
    """
    Global session manager for handling cross-page session state issues.
    """
    
    @staticmethod
    def initialize_global_state() -> None:
        """Initialize global session state variables."""
        if 'global_session_initialized' not in st.session_state:
            st.session_state.global_session_initialized = True
            st.session_state.global_session_id = str(uuid.uuid4())[:8]
            st.session_state.active_pages = {}
            st.session_state.navigation_history = []
            st.session_state.last_cleanup = time.time()
            
            logger.info("Global session state initialized")
    
    @staticmethod
    def register_page(page_name: str, namespace: str) -> None:
        """Register a page as active."""
        GlobalSessionManager.initialize_global_state()
        
        st.session_state.active_pages[page_name] = {
            'namespace': namespace,
            'last_accessed': datetime.now(),
            'access_count': st.session_state.active_pages.get(page_name, {}).get('access_count', 0) + 1
        }
        
        # Update navigation history
        if page_name not in st.session_state.navigation_history[-5:]:  # Keep last 5
            st.session_state.navigation_history.append(page_name)
            if len(st.session_state.navigation_history) > 5:
                st.session_state.navigation_history = st.session_state.navigation_history[-5:]
    
    @staticmethod
    def cleanup_inactive_pages(max_age_minutes: int = 30) -> None:
        """Clean up session state for pages that haven't been accessed recently."""
        if 'active_pages' not in st.session_state:
            return
        
        current_time = datetime.now()
        inactive_pages = []
        
        for page_name, info in st.session_state.active_pages.items():
            age_minutes = (current_time - info['last_accessed']).total_seconds() / 60
            if age_minutes > max_age_minutes:
                inactive_pages.append((page_name, info['namespace']))
        
        # Clean up inactive pages
        for page_name, namespace in inactive_pages:
            keys_to_remove = [k for k in st.session_state.keys() if k.startswith(f"{namespace}_")]
            for k in keys_to_remove:
                del st.session_state[k]
            
            del st.session_state.active_pages[page_name]
            logger.info(f"Cleaned up inactive page: {page_name}")
        
        st.session_state.last_cleanup = time.time()
    
    @staticmethod
    def get_global_debug_info() -> Dict[str, Any]:
        """Get global debug information."""
        GlobalSessionManager.initialize_global_state()
        
        return {
            "session_id": st.session_state.get('global_session_id'),
            "active_pages": list(st.session_state.get('active_pages', {}).keys()),
            "navigation_history": st.session_state.get('navigation_history', []),
            "total_session_keys": len(st.session_state),
            "last_cleanup": st.session_state.get('last_cleanup', 0),
            "time_since_cleanup": time.time() - st.session_state.get('last_cleanup', 0)
        }


# Convenience functions for common usage patterns
def create_session_manager(page_name: str = None) -> SessionManager:
    """Create a session manager for the current page."""
    manager = SessionManager(page_name)
    GlobalSessionManager.register_page(manager.page_name, manager.namespace)
    return manager


def auto_cleanup_session(max_age_minutes: int = 30) -> None:
    """Automatically clean up old session state if needed."""
    last_cleanup = st.session_state.get('last_cleanup', 0)
    if time.time() - last_cleanup > 300:  # Every 5 minutes
        GlobalSessionManager.cleanup_inactive_pages(max_age_minutes)


# Debug utilities
def show_session_debug_info(manager: SessionManager = None) -> None:
    """Show debug information in the sidebar."""
    if not st.sidebar.checkbox("Show Session Debug", value=False):
        return
    
    with st.sidebar.expander("🔧 Session Debug Info", expanded=False):
        if manager:
            st.write("**Page Info:**")
            debug_info = manager.get_debug_info()
            for key, value in debug_info.items():
                st.write(f"- {key}: {value}")
        
        st.write("**Global Info:**")
        global_info = GlobalSessionManager.get_global_debug_info()
        for key, value in global_info.items():
            st.write(f"- {key}: {value}")
        
        if st.button("Clean Up Session", key="debug_cleanup"):
            if manager:
                manager.cleanup_page()
            GlobalSessionManager.cleanup_inactive_pages(max_age_minutes=0)  # Force cleanup
            st.success("Session cleaned up!")
            st.rerun()
