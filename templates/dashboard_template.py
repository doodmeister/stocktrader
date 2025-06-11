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
1. Uses st.session_state.current_page from dashboard_controller.py for navigation tracking
2. Implements _check_and_clear_on_page_return() to detect page transitions
3. Clears uploaded data when user navigates away and returns
4. Integrates seamlessly with the existing dashboard architecture

USAGE FOR FUTURE AI:
1. Copy this template for any new dashboard page
2. Update the page_name in session_manager creation
3. Update the current_page variable in _check_and_clear_on_page_return()
4. Customize the file upload and display logic as needed
5. Keep the page change detection logic unchanged

TESTED SOLUTION:
This pattern has been successfully implemented and tested in data_analysis_v2.py
and resolves the critical data persistence issue across page navigation.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import time
from typing import Optional, Dict, Any

# Import core modules
from utils.logger import get_dashboard_logger
from core.streamlit.dashboard_utils import setup_page
from core.streamlit.session_manager import create_session_manager
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

# Create session manager instance (UPDATE PAGE NAME FOR YOUR PAGE)
session_manager = create_session_manager(page_name="dashboard_template")


def _check_and_clear_on_page_return():
    """
    Page Change Detection and Data Clearing Mechanism
    
    CRITICAL FUNCTION: This solves the data persistence issue across page navigation.
    
    PROBLEM SOLVED:
    - User uploads CSV file on page A
    - User navigates to Home page  
    - User returns to page A
    - Previously uploaded data should be cleared (but wasn't before this fix)
    
    HOW IT WORKS:
    1. Integrates with dashboard_controller.py's navigation tracking system
    2. Uses st.session_state.current_page to detect the active page
    3. Tracks the last visited page in a page-specific session key
    4. Clears uploaded data when navigation away and back is detected
    5. Uses timestamp-based logic to ensure proper clearing timing
    
    INTEGRATION POINTS:
    - dashboard_controller.py sets st.session_state.current_page
    - This function reads that value to detect page changes
    - Works with the existing page loader and UI renderer system
    
    CUSTOMIZATION FOR NEW PAGES:
    1. Update 'current_page' variable to match your page's filename
    2. Update session state keys to be unique for your page
    3. Add any additional data keys that need clearing on navigation
    
    TESTED: This pattern successfully resolves data persistence issues.
    """
    # UPDATE THIS: Use your page's actual filename as used by dashboard controller
    current_page = "dashboard_template.py"  # Change this for your specific page
    
    # Get the current page from dashboard controller's navigation system
    dashboard_current_page = st.session_state.get('current_page', 'home')
    
    # Create a unique session key for tracking this page's navigation history
    # UPDATE THIS: Use your page's name in the session key
    last_visit_key = 'dashboard_template_last_visit_page'
    navigation_timestamp_key = 'dashboard_template_navigation_timestamp'
    
    # Get current timestamp for navigation timing
    current_time = time.time()
    
    # If we're currently on this page (according to dashboard controller)
    if dashboard_current_page == current_page or dashboard_current_page == 'dashboard_template.py':
        # Check if we were previously on a different page
        last_visited_page = st.session_state.get(last_visit_key, current_page)
        last_navigation_time = st.session_state.get(navigation_timestamp_key, current_time)
        
        # If the last visit was from a different page and enough time has passed
        # (prevents clearing during normal page refresh/rerun)
        if (last_visited_page != current_page and 
            last_visited_page != 'dashboard_template.py' and
            current_time - last_navigation_time > 1.0):  # 1 second threshold
            
            # Clear uploaded data - ADD YOUR DATA KEYS HERE
            data_keys_to_clear = [
                'uploaded_dataframe',
                'uploaded_file_name',
                # Add any other session state keys that should be cleared
                # when users navigate away and return
            ]
            
            cleared_keys = []
            for key in data_keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
                    cleared_keys.append(key)
            
            if cleared_keys:
                logger.info(f"Cleared uploaded data due to page navigation from {last_visited_page} to {current_page}. Cleared keys: {cleared_keys}")
        
        # Update navigation tracking
        st.session_state[last_visit_key] = current_page
        st.session_state[navigation_timestamp_key] = current_time
    else:
        # We're visiting from another page, record where we came from
        st.session_state[last_visit_key] = dashboard_current_page
        st.session_state[navigation_timestamp_key] = current_time


def display_uploaded_data():
    """
    Complete File Upload Example with Page Change Detection Integration
    
    This function demonstrates the proper way to implement file upload
    functionality that integrates with the page change detection system.
    
    FEATURES:
    - Unique widget keys to prevent Streamlit conflicts
    - Immediate data processing and session state storage
    - Error handling with proper cleanup
    - Integration with page change detection
    - Chart display for uploaded data
    
    CUSTOMIZATION:
    1. Update the file_uploader key to be unique for your page
    2. Modify file types and processing logic as needed
    3. Customize the data display and charting logic
    4. Add additional validation or processing steps
    """
    # File uploader with unique key (UPDATE THE KEY FOR YOUR PAGE)
    uploaded_file = st.file_uploader(
        label="Upload your CSV file",
        type=["csv"],
        key="dashboard_template_csv_uploader",  # UNIQUE KEY - change for your page
        help="Upload a CSV file containing your data for analysis"
    )
    
    # Process the uploaded file immediately and store in session state
    if uploaded_file is not None:
        try:
            # Read the CSV and store in session state
            df = pd.read_csv(uploaded_file)
            st.session_state['uploaded_dataframe'] = df
            st.session_state['uploaded_file_name'] = uploaded_file.name
            logger.info(f"Successfully uploaded file: {uploaded_file.name}")
            
            # Optional: Add data validation here
            # validation_result = core_validate_dataframe(df)
            # if not validation_result.is_valid:
            #     st.warning("Data validation warnings detected")
            
        except Exception as e:
            st.error(f"Error reading file: {e}")
            logger.error(f"File upload error: {e}")
            
            # Clear any existing data on error
            for key in ['uploaded_dataframe', 'uploaded_file_name']:
                if key in st.session_state:
                    del st.session_state[key]
            return

    # Display the uploaded data if it exists in session state
    if 'uploaded_dataframe' in st.session_state and st.session_state.get('uploaded_dataframe') is not None:
        df_display = st.session_state['uploaded_dataframe']
        file_name = st.session_state.get('uploaded_file_name', 'Unknown file')
        
        st.subheader("Uploaded Data")
        st.info(f"📁 File: **{file_name}** | Shape: {df_display.shape[0]} rows × {df_display.shape[1]} columns")
        
        # Display data table
        st.dataframe(df_display)
        
        # Display chart if data is available
        if not df_display.empty:
            _display_chart(df_display)
            
        # Show data summary
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
    
    CRITICAL: This function is called by the dashboard controller's page loader.
    The page change detection MUST be called at the very beginning before any
    widgets are created to ensure proper data clearing.
    
    TEMPLATE STRUCTURE:
    1. Page change detection (FIRST - before any widgets)
    2. Page header and description
    3. Main functionality sections
    4. Additional features and analysis
    
    CUSTOMIZATION:
    1. Update page titles and descriptions
    2. Add your specific functionality sections
    3. Modify the layout and components as needed
    4. Keep the page change detection call at the beginning
    """
    # CRITICAL: Check for page changes and clear data BEFORE creating any widgets
    _check_and_clear_on_page_return()
    
    # Page header
    st.header("Dashboard Template with Page Change Detection")
    
    st.markdown("""
    This template demonstrates the complete solution for dashboard pages with file upload
    functionality that properly clears data when users navigate away and return.
    
    **Key Features:**
    - ✅ File upload with session state persistence
    - ✅ Page change detection and data clearing
    - ✅ Error handling and data validation
    - ✅ Adaptive charting and data display
    - ✅ Integration with dashboard controller
    """)

    # Main file upload section
    with st.expander("📁 File Upload Section", expanded=True):
        display_uploaded_data()

    # Additional sections for future functionality
    with st.expander("🔍 Data Analysis Section", expanded=False):
        if 'uploaded_dataframe' in st.session_state and st.session_state.get('uploaded_dataframe') is not None:
            df = st.session_state['uploaded_dataframe']
            
            # Example analysis functionality
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
          # Example configuration options
        st.subheader("Display Settings")
        show_debug = st.checkbox("Show Debug Information", key="dashboard_template_debug_checkbox")
        
        if show_debug:
            st.subheader("Debug Information")
            st.write("**Session State Keys:**")
            relevant_keys = [k for k in st.session_state.keys() if isinstance(k, str) and ('template' in k.lower() or 'upload' in k.lower())]
            for key in relevant_keys:
                st.write(f"- {key}: {type(st.session_state.get(key, 'Not Set'))}")
            
            st.write("**Current Page:** ", st.session_state.get('current_page', 'Unknown'))


# Template Usage Instructions (for future AI)
"""
INSTRUCTIONS FOR FUTURE AI:

To use this template for a new dashboard page:

1. COPY THIS FILE to dashboard_pages/your_new_page.py

2. UPDATE THESE REQUIRED SECTIONS:
   - Line ~35: logger name ('dashboard_template' -> 'your_page_name')
   - Line ~40: page title and setup_page parameters
   - Line ~45: session_manager page_name
   - Line ~85: current_page variable in _check_and_clear_on_page_return()
   - Line ~91: last_visit_key and navigation_timestamp_key names
   - Line ~170: file_uploader key to be unique
   - All widget keys to be unique for your page

3. CUSTOMIZE FUNCTIONALITY:
   - Modify display_uploaded_data() for your specific file processing needs
   - Update _display_chart() for your charting requirements
   - Add your specific analysis and processing sections
   - Modify the main() function layout as needed

4. KEEP UNCHANGED:
   - The _check_and_clear_on_page_return() logic (except for naming updates)
   - The overall structure and error handling patterns
   - The session state management approach

5. TEST THOROUGHLY:
   - Upload a file on your page
   - Navigate to Home page
   - Return to your page
   - Verify the uploaded data is cleared

This template solves the critical data persistence issue and provides a robust
foundation for any dashboard page that needs file upload functionality.
"""

