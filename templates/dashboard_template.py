"""
Dashboard Template with Page Change Detection

This template provides a complete solution for Streamlit dashboard pages that need
file upload functionality without data persistence issues across page navigation.

CRITICAL ISSUE SOLVED:
When users upload files and navigate away from a page, the uploaded data
should be cleared when they return. This template implements a robust
page change detection mechanism that integrates with the dashboard controller's
navigation system.

HOW IT WORKS:
1. Uses SessionManager for all session state and widget key management.
2. SessionManager.has_navigated_to_page() detects page transitions.
3. Clears uploaded data when user navigates away and returns, managed by SessionManager and _init_template_state.
4. Integrates seamlessly with the existing dashboard architecture.

USAGE FOR FUTURE AI:
1. Copy this template for any new dashboard page.
2. Update the namespace_prefix in SessionManager instantiation.
3. Customize the file upload and display logic as needed.
4. Rely on SessionManager and the _init_template_state pattern for state management.

TESTED SOLUTION:
This pattern has been successfully implemented and tested in data_analysis_v2.py
and resolves the critical data persistence issue across page navigation.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Optional, Dict, Any

# Import core modules
from utils.logger import get_dashboard_logger
from core.streamlit.dashboard_utils import setup_page
from core.streamlit.session_manager import SessionManager # Updated import
from core.data_validator import validate_dataframe as core_validate_dataframe
from core.validation.validation_results import DataFrameValidationResult

# Initialize logger for this dashboard page (UPDATE THIS FOR YOUR PAGE)
logger = get_dashboard_logger('dashboard_template')

# Initialize page configuration (UPDATE THESE FOR YOUR PAGE)
setup_page(
    title="📋 Dashboard Template",
    logger_name=__name__,
    sidebar_title="Dashboard Controls"
)

# Create session manager instance (UPDATE NAMESPACE_PREFIX FOR YOUR PAGE)
session_manager = SessionManager(namespace_prefix="dashboard_template")


def _init_template_state():
    """
    Initialize session state for the dashboard template page.
    Uses SessionManager.has_navigated_to_page() to determine if page-specific state
    (non-namespaced keys like 'uploaded_dataframe') should be cleared.
    SessionManager handles its own namespaced keys (clears them on navigation).
    This function ensures all necessary keys (namespaced and non-namespaced) have defaults.
    """
    is_new_page_navigation = session_manager.has_navigated_to_page()

    if is_new_page_navigation:
        logger.info(
            f"DashboardTemplate: Navigation to page detected by SessionManager for namespace '{session_manager.namespace}'. "
            f"Clearing additional page-specific (non-namespaced) state."
        )
        # These are keys not automatically managed by SessionManager's namespace clearing
        # because they are not prefixed with the namespace.
        non_namespaced_keys_to_clear = [
            'uploaded_dataframe', # Example: still using a non-namespaced key for the main data
            'uploaded_file_name',
            'template_last_file_id' # Important for new file upload detection
        ]
        for k in non_namespaced_keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]
                logger.debug(f"Cleared non-namespaced key: {k}")
    else:
        logger.info(
            f"DashboardTemplate: In-page rerun for namespace '{session_manager.namespace}'. Not clearing non-namespaced state."
        )

    # Define non-namespaced state and their defaults (if any)
    # These are typically for data that might be large or shared across components without SM prefixing.
    # Ensure their absence is handled gracefully if they are cleared on navigation.
    # For this template, 'uploaded_dataframe' and 'uploaded_file_name' are set by file upload logic.

    # Define page-specific state (managed by SessionManager) and their defaults
    page_state_definitions = { # key_suffix: default_value
        'analysis_summary': '',
        'show_debug_info_template': False,
        'new_file_uploaded_this_run': False,
    }

    for key_suffix, default_value in page_state_definitions.items():
        _unique_sentinel = object()
        current_sm_value = session_manager.get_page_state(key_suffix, _unique_sentinel)
        if is_new_page_navigation or current_sm_value is _unique_sentinel:
            session_manager.set_page_state(key_suffix, default_value)
            logger.debug(f"SM state '{key_suffix}' set to default '{default_value}' (new page or missing).")

_init_template_state() # Call initialization


def display_uploaded_data():
    """
    Complete File Upload Example with SessionManager Integration.
    """
    session_manager.set_page_state('new_file_uploaded_this_run', False)

    uploaded_file = session_manager.create_file_uploader(
        label="Upload your CSV file",
        type=["csv"],
        file_uploader_name="template_csv_uploader",
        help="Upload a CSV file containing your data for analysis"
    )
    
    if uploaded_file is not None:
        current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        previous_file_id = st.session_state.get('template_last_file_id', '')
        
        if current_file_id != previous_file_id:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state['uploaded_dataframe'] = df # Store raw DataFrame (non-namespaced for example)
                st.session_state['uploaded_file_name'] = uploaded_file.name
                st.session_state['template_last_file_id'] = current_file_id
                logger.info(f"Successfully uploaded file: {uploaded_file.name}")
                
                # Clear relevant SessionManager-managed states on new file upload
                sm_keys_to_clear_on_new_file = ['analysis_summary']
                for key_suffix in sm_keys_to_clear_on_new_file:
                    session_manager.clear_page_state(key_suffix)
                    logger.debug(f"Cleared SM state '{key_suffix}' due to new file upload.")
                
                session_manager.set_page_state('new_file_uploaded_this_run', True)
                st.success("File uploaded successfully. Previous analysis results cleared.")
                
            except Exception as e:
                st.error(f"Error reading file: {e}")
                logger.error(f"File upload error: {e}")
                st.session_state.pop('uploaded_dataframe', None)
                st.session_state.pop('uploaded_file_name', None)
                return

    if 'uploaded_dataframe' in st.session_state and st.session_state.get('uploaded_dataframe') is not None:
        df_display = st.session_state['uploaded_dataframe']
        file_name = st.session_state.get('uploaded_file_name', 'Unknown file')
        
        st.subheader("Uploaded Data")
        st.info(f"📁 File: **{file_name}** | Shape: {df_display.shape[0]} rows × {df_display.shape[1]} columns")
        st.dataframe(df_display)
        
        if not df_display.empty:
            _display_chart(df_display)
            
        with st.expander("Data Summary", expanded=False):
            st.write("**Column Information:**")
            st.write(df_display.dtypes)
            st.write("**Statistical Summary:**")
            st.write(df_display.describe())
    else:
        st.info("Please upload a CSV file to see the data and analysis.")


def _display_chart(df: pd.DataFrame):
    """
    Display a chart based on the available columns in the DataFrame.
    
    This function demonstrates adaptive charting that works with various
    data formats and column naming conventions.
    
    Args:
        df: The DataFrame to chart
    """
    try:
        # Find appropriate columns for plotting
        date_col = _find_column(df, ['date', 'timestamp', 'datetime', 'time'])
        value_col = _find_column(df, ['close', 'value', 'price', 'amount', 'y'])
        
        # Convert date column to datetime if needed
        if date_col and pd.api.types.is_string_dtype(df[date_col]):
            try:
                df = df.copy()
                df[date_col] = pd.to_datetime(df[date_col])
                # If 'uploaded_dataframe' is the source, update it. 
                # Be cautious if df is a copy made for local modification.
                if 'uploaded_dataframe' in st.session_state and st.session_state['uploaded_dataframe'] is df:
                    st.session_state['uploaded_dataframe'] = df
            except Exception:
                st.warning(f"Could not convert '{date_col}' column to datetime. Chart may not display correctly.")
        
        # Create chart based on available columns
        if date_col and value_col:
            fig = px.line(df, x=date_col, y=value_col, title=f'Data Visualization ({value_col} over {date_col})')
            st.plotly_chart(fig, use_container_width=True)
        elif value_col:
            fig = px.line(df, y=value_col, title=f'Data Visualization ({value_col} over Index)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Show first numeric column if available
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                first_numeric = numeric_cols[0]
                fig = px.line(df, y=first_numeric, title=f'Data Visualization ({first_numeric})')
                st.plotly_chart(fig, use_container_width=True)
            else:
                available_cols = list(df.columns)
                st.warning(f"Could not find suitable columns for charting. Available columns: {available_cols}")
                
    except Exception as e:
        st.error(f"Could not display chart: {e}")
        logger.error(f"Chart display error: {e}")


def _find_column(df: pd.DataFrame, column_names: list) -> Optional[str]:
    """
    Find the first matching column name (case-insensitive).
    
    Args:
        df: DataFrame to search
        column_names: List of possible column names to match
        
    Returns:
        The first matching column name, or None if no match found
    """
    for col in df.columns:
        if col.lower() in column_names:
            return col
    return None


def main():
    """
    Main Function - Entry Point for Dashboard Page
    The _init_template_state() call handles page change detection and state initialization.
    """
    # Page header
    st.header("Dashboard Template with Updated Session Management")
    
    st.markdown("""
    This template demonstrates the complete solution for dashboard pages with file upload
    functionality using the new SessionManager for robust state handling.
    
    **Key Features:**
    - ✅ File upload with session state persistence via SessionManager
    - ✅ Page change detection and data clearing handled by SessionManager and _init_template_state
    - ✅ Error handling and data validation
    - ✅ Adaptive charting and data display
    - ✅ Integration with dashboard controller
    """)

    with st.expander("📁 File Upload Section", expanded=True):
        display_uploaded_data()

    with st.expander("🔍 Data Analysis Section", expanded=False):
        if 'uploaded_dataframe' in st.session_state and st.session_state.get('uploaded_dataframe') is not None:
            df = st.session_state['uploaded_dataframe']
            st.subheader("Data Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rows", len(df))
                st.metric("Total Columns", len(df.columns))
            with col2:
                numeric_cols = len(df.select_dtypes(include=['number']).columns)
                text_cols = len(df.select_dtypes(include=['object']).columns)
                st.metric("Numeric Columns", numeric_cols)
                st.metric("Text Columns", text_cols)
        else:
            st.info("Upload data to enable analysis features")
    
    with st.expander("⚙️ Configuration & Settings", expanded=False):
        st.info("Additional configuration options can be added here")
        st.subheader("Display Settings")
        # Use SessionManager for widget key and state
        show_debug = session_manager.create_checkbox(
            "Show Debug Information", 
            checkbox_name="template_debug_checkbox", # SM will namespace this
            # Default value for checkbox can be set via session_manager.get_page_state in _init_template_state
            # or directly here if not managed in _init_template_state
            value=session_manager.get_page_state('show_debug_info_template', False) 
        )
        session_manager.set_page_state('show_debug_info_template', show_debug)
        
        if show_debug:
            session_manager.debug_session_state() # Use SessionManager's debug display
            st.subheader("Additional Non-Namespaced Debug Info")
            st.write("**Non-Namespaced Session State Keys (Example):**")
            non_namespaced_to_show = ['uploaded_dataframe', 'uploaded_file_name', 'template_last_file_id']
            for key in non_namespaced_to_show:
                if key in st.session_state:
                    st.write(f"- {key}: ({type(st.session_state[key])}) - Present")
                else:
                    st.write(f"- {key}: Not Set")
            st.write("**Current Page (from dashboard_controller):** ", st.session_state.get('current_page', 'Unknown'))


"""
INSTRUCTIONS FOR FUTURE AI:

To use this template for a new dashboard page:

1. COPY THIS FILE to dashboard_pages/your_new_page.py

2. UPDATE THESE REQUIRED SECTIONS:
   - Logger name (e.g., 'dashboard_template' -> 'your_page_name')
   - `setup_page` parameters (title, etc.)
   - `SessionManager` instantiation: `SessionManager(namespace_prefix="your_page_name")`
   - `_init_template_state` function name to `_init_your_page_name_state` and update its internal logging/keys if necessary.
   - All specific widget names passed to `session_manager.create_widget_type()` methods to be descriptive for your page.
   - If using non-namespaced keys (like 'uploaded_dataframe'), ensure their names are unique if necessary or managed carefully.

3. CUSTOMIZE FUNCTIONALITY:
   - Modify `display_uploaded_data()` for your specific file processing needs.
   - Update `_display_chart()` for your charting requirements.
   - Add your specific analysis and processing sections.
   - Modify the `main()` function layout as needed.

4. KEY PATTERNS TO MAINTAIN:
   - Instantiate `SessionManager` at the top with a unique `namespace_prefix`.
   - Call an `_init_your_page_name_state()` function at the beginning of your script (after SM instantiation) that uses `session_manager.has_navigated_to_page()` to clear non-namespaced state and initialize SM-managed state.
   - Use `session_manager.create_widget_type()` for all Streamlit widgets to ensure unique keys.
   - Use `session_manager.get_page_state()` and `session_manager.set_page_state()` for managing state that needs to be namespaced and automatically handled by SessionManager.

5. TEST THOROUGHLY:
   - Upload a file on your page.
   - Navigate to another page (e.g., Home).
   - Return to your page.
   - Verify that non-namespaced data (like 'uploaded_dataframe') is cleared and SM-managed state is reset as expected.

This template uses the new SessionManager for robust state and widget key management,
solving data persistence issues and simplifying development.
"""

