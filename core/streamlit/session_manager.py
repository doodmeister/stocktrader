# filepath: c:\\dev\\stocktrader\\core\\session_manager.py
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
"""

import streamlit as st
import time
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager

from utils.logger import get_dashboard_logger

logger = get_dashboard_logger(__name__)

# Key for st.session_state to track the last active SessionManager namespace
_SM_LAST_ACTIVE_NAMESPACE_KEY = "_sm_last_active_page_namespace"


@dataclass
class PageContext:
    """Context information for a dashboard page."""
    page_name: str  # This will be derived from namespace_prefix
    namespace: str
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    button_count: int = 0
    form_count: int = 0
    cleanup_keys: List[str] = field(default_factory=list)


class SessionManager:
    """
    Comprehensive session state management for StockTrader dashboards.
    Now supports optional tab context for widget key namespacing and
    detects navigation to a new page to signal state clearing.
    """

    def __init__(self, namespace_prefix: str, tab: Optional[str] = None, debug_mode: bool = False):
        """Initialize the SessionManager with a unique namespace for the page.

        Args:
            namespace_prefix: A unique identifier for the page (e.g., 'data_analysis_v2').
            tab: Optional tab identifier for namespacing keys within a tab.
            debug_mode: Enable verbose logging for debugging.
        """
        self.namespace_prefix = namespace_prefix
        self.tab_context = f"_{tab}" if tab else ""
        # Ensure '_stable' suffix is applied correctly
        if namespace_prefix.endswith('_stable'):
            self.namespace = f"{namespace_prefix}{self.tab_context}"
        else:
            self.namespace = f"{namespace_prefix}{self.tab_context}_stable"

        self.debug_mode = debug_mode
        self.logger = get_dashboard_logger(f"session_manager_{self.namespace}")

        if self.debug_mode:
            self.logger.info(f"SessionManager initialized for namespace: {self.namespace}")
            self.logger.info(f"Current _SM_LAST_ACTIVE_NAMESPACE_KEY before init logic: {st.session_state.get(_SM_LAST_ACTIVE_NAMESPACE_KEY)}")

        # Initialize session state for this namespace if it doesn't exist
        if self.namespace not in st.session_state:
            st.session_state[self.namespace] = {}
            if self.debug_mode:
                self.logger.info(f"Initialized new session state for {self.namespace}")

        # Initialize a set to keep track of keys managed by this SM instance
        self._managed_keys_key = f"_sm_managed_keys_{self.namespace}"
        if self._managed_keys_key not in st.session_state:
            st.session_state[self._managed_keys_key] = set()
            if self.debug_mode:
                self.logger.info(f"Initialized managed keys set for {self.namespace}")
        
        self._initialize_page_context() # Ensure page context is initialized

        # Log the current value of _SM_LAST_ACTIVE_NAMESPACE_KEY at the end of init
        self.logger.info(f"SessionManager __init__ for '{self.namespace}'. _SM_LAST_ACTIVE_NAMESPACE_KEY is currently: '{st.session_state.get(_SM_LAST_ACTIVE_NAMESPACE_KEY)}'")

    def _log_debug(self, message: str):
        """Log a debug message if debug mode is enabled."""
        if self.debug_mode:
            self.logger.debug(message)

    def _clear_all_sm_managed_state_for_namespace(self):
        """
        Clears all state managed by SessionManager for the current namespace.
        This includes keys set by set_page_state(), widget keys, and the PageContext itself.
        """
        logger.info(f"SessionManager: Clearing all SM-managed state for namespace '{self.namespace}'.")
        keys_to_delete = []
        # Collect keys associated with this namespace (page state and widget keys)
        for key in st.session_state.keys():
            if isinstance(key, str) and key.startswith(self.namespace + "_"):
                keys_to_delete.append(key)
        
        # Also collect the PageContext key
        context_key = f"_page_context_{self.namespace}"
        if context_key in st.session_state:
            keys_to_delete.append(context_key)

        for key in keys_to_delete:
            if key in st.session_state: # Double check before deleting
                del st.session_state[key]
                logger.debug(f"SessionManager: Deleted SM-managed key '{key}' for namespace '{self.namespace}'.")

    def has_navigated_to_page(self) -> bool:
        """
        Detects if navigation to a new page has occurred.

        This should be called once at thebeginning of each page's script.
        It updates the global last active namespace tracker.

        Returns:
            bool: True if navigation to this page is new or if it's a different page,
                  False otherwise.
        """
        last_active_namespace = st.session_state.get(_SM_LAST_ACTIVE_NAMESPACE_KEY)
        self.logger.info(f"has_navigated_to_page called for '{self.namespace}'. Previous last_active_namespace: '{last_active_namespace}'")

        if last_active_namespace != self.namespace:
            self.logger.info(f"SessionManager: New page entry or navigation detected for namespace '{self.namespace}'. Last active was '{last_active_namespace}'.")
            st.session_state[_SM_LAST_ACTIVE_NAMESPACE_KEY] = self.namespace
            self.logger.info(f"Updated _SM_LAST_ACTIVE_NAMESPACE_KEY to '{self.namespace}'")
            # Clear all SM-managed state for this namespace upon new entry/navigation
            self._clear_all_sm_managed_state_for_namespace() # Corrected method call
            return True
        
        self.logger.info(f"has_navigated_to_page: No navigation detected for '{self.namespace}'. _SM_LAST_ACTIVE_NAMESPACE_KEY remains '{st.session_state.get(_SM_LAST_ACTIVE_NAMESPACE_KEY)}'.")
        return False
    
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
        """Generate a stable namespace for this page."""
        # Create a stable namespace based only on page name for consistency
        # This ensures buttons maintain their keys across page refreshes
        # Use namespace_prefix as the base for the page name part of the namespace
        return f"{self.namespace_prefix}_stable"
    
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
                page_name=self.namespace_prefix, # Use namespace_prefix for page_name
                namespace=self.namespace,
                session_id=self._get_or_create_session_id() # Use method to get session_id
            )
    
    def _tab_prefix(self) -> str:
        """Return tab prefix for key if tab context is set."""
        # self.tab is not an attribute, use self.tab_context which is derived from tab
        return self.tab_context + "_" if self.tab_context else ""

    def get_unique_key(self, base_key: str, key_type: str = "button") -> str:
        """
        Generate a stable unique key for buttons, forms, or other Streamlit components.
        
        Args:
            base_key: Base name for the key (e.g., "update", "clear_data")
            key_type: Type of component ("button", "form", "input", etc.)
            
        Returns:
            str: Stable unique key that won't conflict with other pages or tabs
        """
        tab_prefix = self._tab_prefix()
        unique_key = f"{self.namespace}_{tab_prefix}{key_type}_{base_key}"
        
        # Track this key for cleanup
        context_key = f"_page_context_{self.namespace}"
        context = st.session_state.get(context_key)
        if context and unique_key not in context.cleanup_keys:
            context.cleanup_keys.append(unique_key)
        
        logger.debug(f"Generated stable key: {unique_key}")
        return unique_key
    
    def get_form_key(self, form_name: str = "main") -> str:
        """Get a unique form key for this page."""
        return self.get_unique_key(form_name, "form")
    
    def get_button_key(self, button_name: str) -> str:
        return self.get_widget_key("button", button_name)

    def get_input_key(self, input_name: str) -> str:
        return self.get_widget_key("input", input_name)

    def get_widget_key(self, widget_type: str, name: str) -> str:
        """
        Generic method to generate a stable unique key for any Streamlit widget.

        Args:
            widget_type: The type of widget (e.g. 'button', 'form', 'input', 'text_area')
            name: Base name for the key (e.g. 'submit', 'file_upload')
        Returns:
            str: Namespaced, stable widget key.
        """
        # Ensure widget_type is a string and doesn't contain problematic characters
        safe_widget_type = "".join(c if c.isalnum() or c == '_' else '' for c in str(widget_type).lower())
        safe_name = "".join(c if c.isalnum() or c == '_' else '' for c in str(name).lower())

        tab_prefix = self._tab_prefix()
        unique_key = f"{self.namespace}_{tab_prefix}{safe_widget_type}_{safe_name}"
        
        # Track this key for cleanup
        context_key = f"_page_context_{self.namespace}"
        context = st.session_state.get(context_key)
        if context and unique_key not in context.cleanup_keys:
            context.cleanup_keys.append(unique_key)
        
        logger.debug(f"Generated stable widget key: {unique_key}")
        return unique_key

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
    
    def clear_page_state(self, key: Optional[str] = None) -> None:
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
            keys_to_remove = [k for k in st.session_state.keys() if isinstance(k, str) and k.startswith(f"{self.namespace}_")]
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
        
        logger.info(f"Cleaned up session state for namespace: {self.namespace}") # Log with namespace
    
    @contextmanager
    def form_container(self, form_name: str = "main", location: Optional[str] = None, **form_kwargs):
        """
        Context manager for creating forms with automatic key management and location support.
        
        Args:
            form_name: Name for the form
            location: Where to place the form ("sidebar" or None for main area)
            **form_kwargs: Additional arguments for st.form() (excluding location)
            
        Usage:
            with session_manager.form_container("my_form", location="sidebar") as form:
                # form content here
                submitted = st.form_submit_button("Submit")
        """
        form_key = self.get_form_key(form_name)
        
        try:
            # Handle location parameter - sidebar vs main area
            if location == "sidebar":
                with st.sidebar.form(key=form_key, **form_kwargs) as form:
                    logger.debug(f"Created sidebar form with key: {form_key}")
                    yield form
            else:
                with st.form(key=form_key, **form_kwargs) as form:
                    logger.debug(f"Created main area form with key: {form_key}")
                    yield form
        except Exception as e:
            logger.error(f"Error in form container {form_key}: {e}")
            raise
    
    def create_button(self, label: str, button_name: Optional[str] = None, **button_kwargs) -> bool:
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
    
    def create_checkbox(self, label: str, checkbox_name: Optional[str] = None, **checkbox_kwargs) -> bool:
        """
        Create a checkbox with automatic key management.
        
        Args:
            label: Checkbox label text
            checkbox_name: Internal name for the checkbox (defaults to label)
            **checkbox_kwargs: Additional arguments for st.checkbox()
            
        Returns:
            bool: True if checkbox is checked
        """
        if checkbox_name is None:
            checkbox_name = label.lower().replace(" ", "_")
        
        checkbox_key = self.get_button_key(checkbox_name)  # Reuse button key logic for consistency
        
        try:
            checked = st.checkbox(label, key=checkbox_key, **checkbox_kwargs)
            logger.debug(f"Checkbox {checkbox_key}: {checked}")
            return checked
        except Exception as e:
            logger.error(f"Error creating checkbox {checkbox_key}: {e}")
            return False
    
    def create_selectbox(self, label: str, options, selectbox_name: Optional[str] = None, **selectbox_kwargs):
        """
        Create a selectbox with automatic key management.
        
        Args:
            label: Selectbox label text
            options: List of options for the selectbox
            selectbox_name: Internal name for the selectbox (defaults to label)
            **selectbox_kwargs: Additional arguments for st.selectbox()
            
        Returns:
            Selected option
        """
        if selectbox_name is None:
            selectbox_name = label.lower().replace(" ", "_")
        
        selectbox_key = self.get_button_key(selectbox_name)  # Reuse button key logic for consistency
        
        try:
            selected = st.selectbox(label, options, key=selectbox_key, **selectbox_kwargs)
            logger.debug(f"Selectbox {selectbox_key}: {selected}")
            return selected
        except Exception as e:
            logger.error(f"Error creating selectbox {selectbox_key}: {e}")
            return options[0] if options else None

    def create_multiselect(self, label: str, options, multiselect_name: Optional[str] = None, **multiselect_kwargs):
        """
        Create a multiselect with automatic key management.
        
        Args:
            label: Multiselect label text
            options: List of options for the multiselect
            multiselect_name: Internal name for the multiselect (defaults to label)
            **multiselect_kwargs: Additional arguments for st.multiselect()
            
        Returns:
            List of selected options
        """
        if multiselect_name is None:
            multiselect_name = label.lower().replace(" ", "_")
        
        multiselect_key = self.get_unique_key(multiselect_name, "multiselect")
        
        try:
            selected = st.multiselect(label, options, key=multiselect_key, **multiselect_kwargs)
            logger.debug(f"Multiselect {multiselect_key}: {selected}")
            return selected
        except Exception as e:
            logger.error(f"Error creating multiselect {multiselect_key}: {e}")
            return []

    def create_slider(self, label: str, min_value, max_value, value=None, slider_name: Optional[str] = None, **slider_kwargs):
        """
        Create a slider with automatic key management.
        
        Args:
            label: Slider label text
            min_value: Minimum value
            max_value: Maximum value
            value: Initial value
            slider_name: Internal name for the slider (defaults to label)
            **slider_kwargs: Additional arguments for st.slider()
            
        Returns:
            Selected value
        """
        if slider_name is None:
            slider_name = label.lower().replace(" ", "_")
        
        slider_key = self.get_unique_key(slider_name, "slider")
        
        try:
            selected = st.slider(label, min_value, max_value, value, key=slider_key, **slider_kwargs)
            logger.debug(f"Slider {slider_key}: {selected}")
            return selected
        except Exception as e:
            logger.error(f"Error creating slider {slider_key}: {e}")
            return value if value is not None else min_value

    def create_text_input(self, label: str, value="", text_input_name: Optional[str] = None, **text_input_kwargs):
        """
        Create a text input with automatic key management.
        
        Args:
            label: Text input label text
            value: Initial value
            text_input_name: Internal name for the text input (defaults to label)
            **text_input_kwargs: Additional arguments for st.text_input()
            
        Returns:
            Input text value
        """
        if text_input_name is None:
            text_input_name = label.lower().replace(" ", "_")
        
        text_input_key = self.get_unique_key(text_input_name, "text_input")
        
        try:
            text_value = st.text_input(label, value, key=text_input_key, **text_input_kwargs)
            logger.debug(f"Text input {text_input_key}: {text_value}")
            return text_value
        except Exception as e:
            logger.error(f"Error creating text input {text_input_key}: {e}")
            return value

    def create_number_input(self, label: str, min_value=None, max_value=None, value=None, number_input_name: Optional[str] = None, **number_input_kwargs):
        """
        Create a number input with automatic key management.
        
        Args:
            label: Number input label text
            min_value: Minimum value
            max_value: Maximum value
            value: Initial value
            number_input_name: Internal name for the number input (defaults to label)
            **number_input_kwargs: Additional arguments for st.number_input()
        
        Returns:
            Input number value
        """
        if number_input_name is None:
            number_input_name = label.lower().replace(" ", "_")
        number_input_key = self.get_unique_key(number_input_name, "number_input")
        try:
            number_value = st.number_input(label, min_value, max_value, value, key=number_input_key, **number_input_kwargs)
            logger.debug(f"Number input {number_input_key}: {number_value}")
            return number_value
        except Exception as e:
            logger.error(f"Error creating number input {number_input_key}: {e}")
            return value if value is not None else 0

    def create_date_input(self, label: str, value=None, date_input_name: Optional[str] = None, **date_input_kwargs):
        """
        Create a date input with automatic key management.
        
        Args:
            label: Date input label text
            value: Initial date value
            date_input_name: Internal name for the date input (defaults to label)
            **date_input_kwargs: Additional arguments for st.date_input()
            
        Returns:
            Selected date
        """
        if date_input_name is None:
            date_input_name = label.lower().replace(" ", "_")
        
        date_input_key = self.get_unique_key(date_input_name, "date_input")
        
        try:
            date_value = st.date_input(label, value, key=date_input_key, **date_input_kwargs)
            logger.debug(f"Date input {date_input_key}: {date_value}")
            return date_value
        except Exception as e:
            logger.error(f"Error creating date input {date_input_key}: {e}")
            return value

    def create_file_uploader(self, label: str, type=None, file_uploader_name: Optional[str] = None, **file_uploader_kwargs):
        """
        Create a file uploader with automatic key management.
        
        Args:
            label: File uploader label text
            type: Accepted file types
            file_uploader_name: Internal name for the file uploader (defaults to label)
            **file_uploader_kwargs: Additional arguments for st.file_uploader()
            
        Returns:
            Uploaded file object
        """
        if file_uploader_name is None:
            file_uploader_name = label.lower().replace(" ", "_")
        
        file_uploader_key = self.get_unique_key(file_uploader_name, "file_uploader")
        
        try:
            uploaded_file = st.file_uploader(label, type, key=file_uploader_key, **file_uploader_kwargs)
            logger.debug(f"File uploader {file_uploader_key}: {uploaded_file.name if uploaded_file else None}")
            return uploaded_file
        except Exception as e:
            logger.error(f"Error creating file uploader {file_uploader_key}: {e}")
            return None

    def create_radio(self, label: str, options, radio_name: Optional[str] = None, **radio_kwargs):
        """
        Create a radio button group with automatic key management.
        
        Args:
            label: Radio button label text
            options: List of options for the radio buttons
            radio_name: Internal name for the radio buttons (defaults to label)
            **radio_kwargs: Additional arguments for st.radio()
            
        Returns:
            Selected option
        """
        if radio_name is None:
            radio_name = label.lower().replace(" ", "_")
        
        radio_key = self.get_unique_key(radio_name, "radio")
        
        try:
            selected = st.radio(label, options, key=radio_key, **radio_kwargs)
            logger.debug(f"Radio {radio_key}: {selected}")
            return selected
        except Exception as e:
            logger.error(f"Error creating radio {radio_key}: {e}")
            return options[0] if options else None

    def create_time_input(self, label: str, value=None, time_input_name: Optional[str] = None, **time_input_kwargs):
        """
        Create a time input with automatic key management.
        
        Args:
            label: Time input label text
            value: Initial time value
            time_input_name: Internal name for the time input (defaults to label)
            **time_input_kwargs: Additional arguments for st.time_input()
            
        Returns:
            Selected time
        """
        if time_input_name is None:
            time_input_name = label.lower().replace(" ", "_")
        
        time_input_key = self.get_unique_key(time_input_name, "time_input")
        
        try:
            time_value = st.time_input(label, value, key=time_input_key, **time_input_kwargs)
            logger.debug(f"Time input {time_input_key}: {time_value}")
            return time_value
        except Exception as e:
            logger.error(f"Error creating time input {time_input_key}: {e}")
            return value

    def create_color_picker(self, label: str, value="#000000", color_picker_name: Optional[str] = None, **color_picker_kwargs):
        """
        Create a color picker with automatic key management.
        
        Args:
            label: Color picker label text
            value: Initial color value
            color_picker_name: Internal name for the color picker (defaults to label)
            **color_picker_kwargs: Additional arguments for st.color_picker()
            
        Returns:
            Selected color
        """
        if color_picker_name is None:
            color_picker_name = label.lower().replace(" ", "_")
        
        color_picker_key = self.get_unique_key(color_picker_name, "color_picker")
        
        try:
            color_value = st.color_picker(label, value, key=color_picker_key, **color_picker_kwargs)
            logger.debug(f"Color picker {color_picker_key}: {color_value}")
            return color_value
        except Exception as e:
            logger.error(f"Error creating color picker {color_picker_key}: {e}")
            return value
            
    def debug_session_state(self) -> None:
        """
        Displays detailed session state information for debugging purposes,
        focusing on the current namespace.
        """
        self.logger.info(f"Displaying session debug info for namespace: {self.namespace}")
        st.subheader(f"Session State Debug: Namespace '{self.namespace}'")

        st.write("Page Context:")
        context_key = f"_page_context_{self.namespace}"
        if context_key in st.session_state:
            # Safely convert to dict if it's a dataclass instance
            context_data = st.session_state[context_key]
            if hasattr(context_data, '__dict__'):
                st.json(context_data.__dict__)
            elif isinstance(context_data, dict):
                 st.json(context_data)
            else:
                st.write(str(context_data))
        else:
            st.warning("No PageContext found for this namespace.")

        st.write(f"Managed Keys for '{self.namespace}':")
        managed_keys_key = f"_sm_managed_keys_{self.namespace}"
        if managed_keys_key in st.session_state and st.session_state[managed_keys_key]:
            st.json(list(st.session_state[managed_keys_key]))
        else:
            st.info(f"No keys explicitly managed by this SessionManager instance for this namespace.")

        st.write(f"All Session State Keys Starting with '{self.namespace}_':")
        namespaced_keys = {
            k: str(type(v))
            for k, v in st.session_state.items()
            if isinstance(k, str) and k.startswith(self.namespace + "_")
        }
        if namespaced_keys:
            st.json(namespaced_keys)
        else:
            st.info(f"No session state keys found starting with '{self.namespace}_'.")
        
        st.write(f"Raw st.session_state['{self.namespace}'] (if exists):")
        if self.namespace in st.session_state:
            st.json(st.session_state[self.namespace])
        else:
            st.info(f"No direct entry for st.session_state['{self.namespace}'].");

        st.write("Global Session Manager Info:")
        st.text(f"_SM_LAST_ACTIVE_NAMESPACE_KEY: {st.session_state.get(_SM_LAST_ACTIVE_NAMESPACE_KEY)}")
        if 'global_session_id' in st.session_state:
            st.text(f"Global Session ID: {st.session_state.global_session_id}")

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the current session state."""
        context_key = f"_page_context_{self.namespace}"
        context = st.session_state.get(context_key)
        
        return {
            "page_name": self.namespace_prefix, # Use namespace_prefix
            "namespace": self.namespace,
            "session_id": self._get_or_create_session_id(), # Use method
            "context_exists": context is not None,
            "button_count": context.button_count if context else 0,
            "form_count": context.form_count if context else 0,
            "cleanup_keys": len(context.cleanup_keys) if context else 0,
            "total_session_keys": len(st.session_state),
            "page_specific_keys": len([k for k in st.session_state.keys() if isinstance(k, str) and k.startswith(self.namespace)])
        }

    def reset_on_first_load(self):
        """
        Reset all page-specific session state ONLY on the first load of the page (not on rerun).
        This uses a page-specific flag in session state to track first load.
        """
        first_load_key = f"{self.namespace}_first_load_done"
        if not st.session_state.get(first_load_key, False):
            self.clear_page_state() # This clears keys starting with self.namespace + "_"
            # Additionally, clear the specific namespace dict if it exists (for non-widget state)
            if self.namespace in st.session_state and isinstance(st.session_state[self.namespace], dict):
                 st.session_state[self.namespace] = {}

            st.session_state[first_load_key] = True
            logger.info(f"SessionManager: Reset page state for namespace {self.namespace} on first load.")


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
            keys_to_remove = [k for k in st.session_state.keys() if isinstance(k, str) and k.startswith(f"{namespace}_")]
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
def create_session_manager(page_name: Optional[str] = None) -> SessionManager:
    """Create a session manager for the current page."""
    # Use a default page_name if None is provided, e.g., "main_dashboard"
    actual_page_name = page_name if page_name is not None else "main_dashboard_page"
    manager = SessionManager(actual_page_name) # Pass actual_page_name as namespace_prefix
    # Register with namespace_prefix, as page_name in PageContext is derived from it
    GlobalSessionManager.register_page(manager.namespace_prefix, manager.namespace) 
    return manager


def auto_cleanup_session(max_age_minutes: int = 30) -> None:
    """Automatically clean up old session state if needed."""
    last_cleanup = st.session_state.get('last_cleanup', 0)
    if time.time() - last_cleanup > 300:  # Every 5 minutes
        GlobalSessionManager.cleanup_inactive_pages(max_age_minutes)


# Debug utilities
def show_session_debug_info(manager: Optional[SessionManager] = None) -> None:
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
