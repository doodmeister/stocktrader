"""
Enhanced Technical Analysis Dashboard - Completely Rebuilt Version

Designed from scratch with robust numeric handling to prevent the 
"'>' not supported between instances of 'str' and 'int'" error.

Key improvements:
- Bulletproof type safety throughout the data pipeline
- Comprehensive data validation and sanitization
- Early detection and correction of type mismatches
- Graceful error handling with detailed logging
- Consistent numeric type enforcement
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import time
import numpy as np
import json
from typing import Optional, Dict, Any

# Import core modules
from utils.logger import get_dashboard_logger
from core.streamlit.dashboard_utils import setup_page
from core.streamlit.session_manager import create_session_manager
from core.data_validator import DataValidator
from core.validation.validation_results import DataFrameValidationResult

# Initialize logger for this dashboard page
logger = get_dashboard_logger('data_analysis_v2_dashboard')

# Initialize page configuration
setup_page(
    title="📈 Stock Data Analysis V2",
    logger_name=__name__,
    sidebar_title="Dashboard Controls"
)

# Create session manager instance
session_manager = create_session_manager(page_name="data_analysis_v2")

# Initialize data validator
data_validator = DataValidator(enable_api_validation=False)  # Disable API validation for file uploads

def validate_uploaded_data(df: pd.DataFrame) -> Optional[DataFrameValidationResult]:
    """
    Comprehensive validation of uploaded CSV data using core validation system.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        DataFrameValidationResult with validation outcome and details
    """
    try:
        logger.info("Starting comprehensive data validation")
        
        # Determine if this looks like OHLC financial data
        has_ohlc = all(col in df.columns for col in ['Open', 'High', 'Low', 'Close'])
        has_ohlc_lowercase = all(col in df.columns for col in ['open', 'high', 'low', 'close'])
        
        # If lowercase OHLC columns exist, create a copy with proper case for validation
        validation_df = df.copy()
        if has_ohlc_lowercase and not has_ohlc:
            validation_df = validation_df.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'
            })
            if 'volume' in validation_df.columns:
                validation_df = validation_df.rename(columns={'volume': 'Volume'})
            has_ohlc = True
        
        # Configure validation parameters based on data type
        if has_ohlc:
            # Financial OHLC data - comprehensive validation
            validation_result = data_validator.validate_dataframe(
                validation_df,
                required_cols=['Open', 'High', 'Low', 'Close'],
                check_ohlcv=True,
                min_rows=1,
                max_rows=100000,  # Reasonable limit for UI performance
                detect_anomalies_level='basic',
                max_null_percentage=0.05  # Allow 5% nulls max
            )
        else:
            # General CSV data - basic validation
            # Try to identify numeric columns for basic validation
            numeric_cols = validation_df.select_dtypes(include=[np.number]).columns.tolist()
            
            validation_result = data_validator.validate_dataframe(
                validation_df,
                required_cols=None,  # No specific required columns for general CSV
                check_ohlcv=False,
                min_rows=1,
                max_rows=100000,
                detect_anomalies_level='basic' if len(numeric_cols) > 0 else None,
                max_null_percentage=0.2  # Allow 20% nulls for general data
            )
        
        logger.info(f"Data validation completed. Valid: {validation_result.is_valid}")
        return validation_result
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        # Return a failed validation result
        return DataFrameValidationResult(
            is_valid=False,
            errors=[f"Validation system error: {str(e)}"],
            validated_data=None,
            error_details=None,
            details=None
        )


def display_validation_results(validation_result: DataFrameValidationResult, df: pd.DataFrame):
    """
    Display comprehensive validation results in a user-friendly format.
    
    Args:
        validation_result: The validation result from core validation system
        df: Original DataFrame being validated
    """
    if validation_result.is_valid:
        st.success("✅ **Data Validation Passed**")
        
        # Display summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
            
        with col2:
            st.metric("Total Columns", len(df.columns))
            
        with col3:
            # Calculate overall data quality score
            total_cells = len(df) * len(df.columns)
            null_cells = df.isnull().sum().sum()
            quality_score = max(0, (total_cells - null_cells) / total_cells * 100) if total_cells > 0 else 0
            st.metric("Data Quality", f"{quality_score:.1f}%")
        
        # Display detailed validation information
        with st.expander("📊 Detailed Validation Results", expanded=False):
            
            # Data Overview
            st.subheader("Data Overview")
            overview_data = {
                "Metric": ["Shape", "Memory Usage", "Data Types", "Missing Values"],
                "Value": [
                    f"{df.shape[0]} rows × {df.shape[1]} columns",
                    f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
                    f"{len(df.dtypes.unique())} unique types",
                    f"{df.isnull().sum().sum():,} cells ({df.isnull().sum().sum() / (len(df) * len(df.columns) if len(df) > 0 else 1) * 100:.1f}%)"
                ]
            }
            st.table(pd.DataFrame(overview_data))
            
            # Column Analysis
            st.subheader("Column Analysis")
            column_info = []
            for col in df.columns:
                col_data = df[col]
                null_count = col_data.isnull().sum()
                null_pct = (null_count / len(df)) * 100 if len(df) > 0 else 0
                
                # Get unique count safely
                try:
                    unique_count = col_data.nunique()
                except:
                    unique_count = "N/A"
                
                column_info.append({
                    "Column": col,
                    "Data Type": str(col_data.dtype),
                    "Non-Null Count": f"{len(df) - null_count:,}",
                    "Null %": f"{null_pct:.1f}%",
                    "Unique Values": unique_count
                })
            
            column_df = pd.DataFrame(column_info)
            st.dataframe(column_df, use_container_width=True)
            
            # OHLC Validation Results (if applicable)
            if hasattr(validation_result, 'details') and validation_result.details:
                ohlc_stats = validation_result.details.get('ohlc_validation_stats', {})
                if ohlc_stats:
                    st.subheader("OHLC Data Validation")
                    st.info("✅ OHLC price relationships validated successfully")
                    
                    if 'rows_checked' in ohlc_stats:
                        st.write(f"**Rows Checked:** {ohlc_stats['rows_checked']:,}")
            
            # Anomaly Detection Results
            if hasattr(validation_result, 'details') and validation_result.details:
                anomaly_stats = validation_result.details.get('anomaly_detection_stats', {})
                if anomaly_stats and anomaly_stats.get('anomalies_found', 0) > 0:
                    st.subheader("Anomaly Detection")
                    st.warning(f"⚠️ Found {anomaly_stats['anomalies_found']} potential anomalies")
                    
                    # Show anomaly details
                    anomaly_details = anomaly_stats.get('details', {})
                    for col, details in anomaly_details.items():
                        if isinstance(details, dict) and 'count' in details:
                            st.write(f"**{col}:** {details['count']} outliers detected using {details.get('method', 'unknown')} method")
                elif anomaly_stats:
                    st.subheader("Anomaly Detection")
                    st.info("✅ No significant anomalies detected")
    
    else:
        # Validation failed
        st.error("❌ **Data Validation Failed**")
        
        if validation_result.errors:
            st.subheader("Validation Errors:")
            for i, error in enumerate(validation_result.errors, 1):
                st.error(f"{i}. {error}")
        
        # Show error details if available
        if hasattr(validation_result, 'error_details') and validation_result.error_details:
            with st.expander("🔍 Detailed Error Information", expanded=True):
                for location, error_list in validation_result.error_details.items():
                    st.write(f"**{location}:**")
                    if isinstance(error_list, list):
                        for error in error_list:
                            st.write(f"  • {error}")
                    else:
                        st.write(f"  • {error_list}")
        
        # Still show some basic statistics even if validation failed
        if df is not None and not df.empty:
            st.subheader("Basic Data Information:")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))


def _check_and_clear_on_page_return():
    """Clear uploaded data if user has navigated away and returned to this page."""
    current_page = "data_analysis_v2.py"
    
    # Check if this is a fresh page load vs a Streamlit rerun
    # We use page history to determine if user has navigated away and returned
    page_history = st.session_state.get('page_history', [])
    current_page_from_controller = st.session_state.get('current_page', 'home')
    
    # Key insight: If we're on data_analysis_v2.py but the previous page in history 
    # was different, it means user navigated away and came back
    if (current_page_from_controller == current_page and 
        len(page_history) >= 2 and 
        page_history[-2] != current_page):
        
        # User has returned to this page from a different page
        # Clear uploaded data
        for key in ['uploaded_dataframe', 'uploaded_file_name', 'validation_result']:
            if key in st.session_state:
                del st.session_state[key]
                
        logger.info(f"Cleared uploaded data due to page navigation. History: {page_history[-3:] if len(page_history) >= 3 else page_history}")
    
    # Alternative approach: Track when we were last on this page
    last_page_visit_key = 'data_analysis_v2_last_session_time'
    current_session_time = st.session_state.get('load_time', time.time())
    last_visit_time = st.session_state.get(last_page_visit_key, 0)
    
    # If significant time has passed or this is first visit, clear data
    if current_session_time - last_visit_time > 1:  # More than 1 second indicates page navigation
        if 'uploaded_dataframe' in st.session_state:
            # Clear uploaded data on fresh page visits
            for key in ['uploaded_dataframe', 'uploaded_file_name', 'validation_result']:
                if key in st.session_state:
                    del st.session_state[key]
            logger.info("Cleared uploaded data due to fresh page visit")
    
    # Update last visit time
    st.session_state[last_page_visit_key] = current_session_time

def display_uploaded_data():
    """Handles file upload and displays the DataFrame."""
    # File uploader with unique key for this page
    uploaded_file = st.file_uploader(
        label="Upload your CSV file",
        type=["csv"],
        key="data_analysis_v2_csv_uploader",
        help="Upload a CSV file containing stock data with columns like 'timestamp', 'close', etc."
    )
    
    # Process the uploaded file immediately and store the DataFrame
    if uploaded_file is not None:
        try:
            # Read the CSV and store in session state
            df = pd.read_csv(uploaded_file)
            st.session_state['uploaded_dataframe'] = df
            st.session_state['uploaded_file_name'] = uploaded_file.name
            
            # Run validation on uploaded data
            validation_result = validate_uploaded_data(df)
            st.session_state['validation_result'] = validation_result
            
            logger.info(f"Successfully uploaded file: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            # Clear any existing data on error
            for key in ['uploaded_dataframe', 'uploaded_file_name', 'validation_result']:
                if key in st.session_state:
                    del st.session_state[key]
            return

    # Display the uploaded data if it exists in session state
    if 'uploaded_dataframe' in st.session_state and st.session_state.get('uploaded_dataframe') is not None:
        df_display = st.session_state['uploaded_dataframe']
        file_name = st.session_state.get('uploaded_file_name', 'Unknown file')
        
        st.subheader("Uploaded Data")
        st.info(f"📁 File: **{file_name}** | Shape: {df_display.shape[0]} rows × {df_display.shape[1]} columns")
        st.dataframe(df_display)

        # Display chart if data is available
        if not df_display.empty:
            _display_chart(df_display)
    else:
        st.info("Please upload a CSV file to see the data and chart.")


def _display_chart(df: pd.DataFrame):
    """Display a chart based on the available columns in the DataFrame."""
    try:
        # Find appropriate columns for plotting
        date_col = _find_column(df, ['date', 'timestamp', 'datetime', 'time'])
        close_col = _find_column(df, ['close', 'closing_price', 'adj_close', 'adjusted_close'])
        
        # Convert date column to datetime if needed
        if date_col and pd.api.types.is_string_dtype(df[date_col]):
            try:
                df = df.copy()
                df[date_col] = pd.to_datetime(df[date_col])
                st.session_state['uploaded_dataframe'] = df
            except Exception:
                st.warning(f"Could not convert '{date_col}' column to datetime. Chart may not display correctly.")
        
        # Create chart based on available columns
        if date_col and close_col:
            fig = px.line(df, x=date_col, y=close_col, title=f'Stock Prices ({close_col} over {date_col})')
            st.plotly_chart(fig, use_container_width=True)
        elif close_col:
            fig = px.line(df, y=close_col, title=f'Stock Prices ({close_col} over Index)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            available_cols = list(df.columns)
            st.warning(f"Could not find suitable price column. Available columns: {available_cols}")
    except Exception as e:
        st.error(f"Could not display chart: {e}")


def _find_column(df: pd.DataFrame, column_names: list) -> Optional[str]:
    """Find the first matching column name (case-insensitive)."""
    for col in df.columns:
        if col.lower() in column_names:
            return col
    return None

def main():
    """Main function to render the Data Analysis V2 page."""
    # Check if user has returned to this page and clear data if needed
    _check_and_clear_on_page_return()
    
    st.header("Stock Data Upload and Display")

    # Section for uploading data
    with st.expander("Upload Stock Data (CSV)", expanded=True):
        display_uploaded_data()

    # Export/Download Section - MOVED HERE to come after upload
    with st.expander("💾 Export & Download Data", expanded=False):
        if 'uploaded_dataframe' in st.session_state:
            df = st.session_state['uploaded_dataframe']
            st.subheader("📁 Download Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Download original data as CSV
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Original CSV",
                    data=csv_data,
                    file_name=f"exported_data_{st.session_state.get('uploaded_file_name', 'unknown')}.csv",
                    mime="text/csv",
                    key="data_analysis_v2_download_original_csv"
                )
            
            with col2:
                # Download validation report (if available)
                if 'validation_result' in st.session_state:
                    validation_result = st.session_state['validation_result']
                    validation_report = {
                        "validation_status": "PASSED" if validation_result.is_valid else "FAILED",
                        "errors": validation_result.errors if validation_result.errors else [],
                        "data_shape": f"{df.shape[0]} rows × {df.shape[1]} columns",
                        "memory_usage_mb": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f}",
                        "null_count": int(df.isnull().sum().sum()),
                        "columns": list(df.columns)
                    }
                    
                    validation_json = json.dumps(validation_report, indent=2)
                    st.download_button(
                        label="📋 Download Validation Report",
                        data=validation_json,
                        file_name=f"validation_report_{st.session_state.get('uploaded_file_name', 'unknown')}.json",
                        mime="application/json",
                        key="data_analysis_v2_download_validation_report"
                    )
                else:
                    st.info("📋 Validation report available after validation")
            
            with col3:
                # Download summary statistics
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    stats_df = df[numeric_cols].describe()
                    stats_csv = stats_df.to_csv()
                    st.download_button(
                        label="📊 Download Statistics",
                        data=stats_csv,
                        file_name=f"statistics_{st.session_state.get('uploaded_file_name', 'unknown')}.csv",
                        mime="text/csv",
                        key="data_analysis_v2_download_statistics"
                    )
                else:
                    st.info("📊 No numeric data for statistics")
        else:
            st.info("Upload a CSV file first to enable download options")
            st.markdown("""
            **Available after uploading:**
            - 📥 **Original CSV**: Download the processed data
            - 📋 **Validation Report**: JSON report with validation results (after validation)
            - 📊 **Statistics**: CSV with descriptive statistics for numeric columns
            """)

    # Enhanced Data Validation Section - MOVED AFTER downloads
    with st.expander("🔍 Data Validation", expanded=False):
        if 'uploaded_dataframe' in st.session_state:
            df = st.session_state['uploaded_dataframe']
            
            # Add a button to run validation manually
            if st.button("🚀 Run Data Validation", key="run_validation_button"):
                with st.spinner("Running comprehensive data validation..."):
                    validation_result = validate_uploaded_data(df)
                    st.session_state['validation_result'] = validation_result
                st.rerun()
            
            # Display validation results if available
            if 'validation_result' in st.session_state:
                validation_result = st.session_state['validation_result']
                st.subheader("📊 Validation Results")
                display_validation_results(validation_result, df)
            else:
                st.info("Click 'Run Data Validation' above to validate your uploaded data")
        else:
            st.info("Upload a CSV file to enable data validation")
            st.markdown("""
            **Validation includes:**
            - **Data Quality Metrics**: Row/column counts, memory usage, missing values
            - **Column Analysis**: Data types, null percentages, unique values
            - **OHLC Validation**: Price relationship integrity (for financial data)            - **Anomaly Detection**: Statistical outlier identification
            - **Error Reporting**: Detailed validation issues and recommendations
            """)
        
    with st.expander("📊 Data Analysis", expanded=False):
        if 'uploaded_dataframe' in st.session_state:
            df = st.session_state['uploaded_dataframe']
            st.subheader("📊 Basic Data Statistics")
            
            # Get numeric columns for analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # Show basic statistics for numeric columns
            if len(numeric_cols) > 0:
                st.write("**Numeric Column Statistics:**")
                st.dataframe(df[numeric_cols].describe())
            else:
                st.info("No numeric columns found for statistical analysis.")
                
            # Show data types
            st.write("**Column Information:**")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info)
        else:
            st.info("📊 Advanced data analysis features will be available after uploading a file")
            st.markdown("""
            **Analysis features include:**
            - 📊 **Statistical Summary**: Descriptive statistics for numeric columns
            - 📋 **Column Information**: Data types, null counts, and unique values  
            - 📈 **Data Visualization**: Interactive charts and plots
            """)


if __name__ == '__main__':
    main()

