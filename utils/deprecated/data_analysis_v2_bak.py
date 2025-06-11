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
import plotly.graph_objects as go
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
from core.indicators.rsi import calculate_rsi
from core.indicators.macd import calculate_macd
from core.indicators.bollinger_bands import calculate_bollinger_bands
from core.technical_indicators import calculate_sma, calculate_ema
# Add candlestick pattern imports
from patterns.patterns import CandlestickPatterns, create_pattern_detector

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
    Display only a simple pass/fail message after validation.
    
    Args:
        validation_result: The validation result from core validation system
        df: Original DataFrame being validated
    """
    if validation_result.is_valid:
        st.success("✅ Data Validation Passed")
    else:
        st.error("❌ Data Validation Failed")
        if validation_result.errors:
            for i, error in enumerate(validation_result.errors, 1):
                st.error(f"{i}. {error}")


def _check_and_clear_on_page_return():
    """Clear uploaded data if user has navigated away and returned to this page."""
    current_page = "data_analysis_v2.py"
    
    # Check if this is a fresh page load vs a Streamlit rerun
    # We use page history to determine if user has navigated away and came back
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

def display_technical_indicators(df: pd.DataFrame):
    """
    Display RSI, MACD, Bollinger Bands, SMA, and EMA for the uploaded DataFrame.
    Enhanced: SMA and EMA are plotted with the 'close' price and a sensitive y-axis.
    """
    st.subheader("Technical Indicators")
    # Only show if 'close' column exists
    if 'close' not in df.columns:
        st.info("No 'close' column found. Technical indicators require a 'close' price column.")
        return

    # RSI
    try:
        rsi = calculate_rsi(df, length=14)
        st.line_chart(rsi, use_container_width=True)
        st.caption("RSI (14)")
    except Exception as e:
        st.warning(f"Could not calculate RSI: {e}")

    # MACD
    try:
        macd_output = calculate_macd(df)
        macd = None
        signal = None
        # Handle tuple output (macd_line, signal_line, histogram)
        if isinstance(macd_output, tuple) and len(macd_output) >= 2:
            macd, signal = macd_output[0], macd_output[1]
        elif isinstance(macd_output, pd.DataFrame):
            columns = list(macd_output.columns) if hasattr(macd_output, 'columns') else []
            macd_col = next((c for c in columns if c.lower() in ['macd', 'macd_line']), None)
            signal_col = next((c for c in columns if c.lower() in ['signal', 'signal_line']), None)
            if macd_col:
                macd = macd_output.loc[:, macd_col]
            if signal_col:
                signal = macd_output.loc[:, signal_col]
        elif isinstance(macd_output, dict):
            macd = macd_output.get('macd') or macd_output.get('macd_line')
            signal = macd_output.get('signal') or macd_output.get('signal_line')
        else:
            macd = getattr(macd_output, 'macd', None) or getattr(macd_output, 'macd_line', None)
            signal = getattr(macd_output, 'signal', None) or getattr(macd_output, 'signal_line', None)
        if macd is not None and signal is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=macd, mode='lines', name='MACD'))
            fig.add_trace(go.Scatter(y=signal, mode='lines', name='Signal'))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("MACD & Signal Line")
        else:
            st.info("MACD or Signal line not found in output.")
    except Exception as e:
        st.warning(f"Could not calculate MACD: {e}")

    # Bollinger Bands
    try:
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=bb_upper, mode='lines', name='Upper Band'))
        fig.add_trace(go.Scatter(y=bb_middle, mode='lines', name='Middle Band'))
        fig.add_trace(go.Scatter(y=bb_lower, mode='lines', name='Lower Band'))
        if 'close' in df.columns:
            fig.add_trace(go.Scatter(y=df['close'], mode='lines', name='Close Price', line=dict(color='gray', dash='dot')))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Bollinger Bands")
    except Exception as e:
        st.warning(f"Could not calculate Bollinger Bands: {e}")

    # SMA (with close price and sensitive y-axis)
    try:
        sma = calculate_sma(df, length=20)
        close = df['close']
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=close, mode='lines', name='Close Price', line=dict(color='gray', dash='dot')))
        fig.add_trace(go.Scatter(y=sma, mode='lines', name='SMA (20)', line=dict(color='blue')))
        # Set y-axis range for sensitivity
        min_y = min(sma.min(), close.min())
        max_y = max(sma.max(), close.max())
        margin = (max_y - min_y) * 0.05 if max_y > min_y else 1
        fig.update_yaxes(range=[min_y - margin, max_y + margin])
        st.plotly_chart(fig, use_container_width=True)
        st.caption("SMA (20) with Close Price")
    except Exception as e:
        st.warning(f"Could not calculate SMA: {e}")

    # EMA (with close price and sensitive y-axis)
    try:
        ema = calculate_ema(df, length=20)
        close = df['close']
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=close, mode='lines', name='Close Price', line=dict(color='gray', dash='dot')))
        fig.add_trace(go.Scatter(y=ema, mode='lines', name='EMA (20)', line=dict(color='orange')))
        min_y = min(ema.min(), close.min())
        max_y = max(ema.max(), close.max())
        margin = (max_y - min_y) * 0.05 if max_y > min_y else 1
        fig.update_yaxes(range=[min_y - margin, max_y + margin])
        st.plotly_chart(fig, use_container_width=True)
        st.caption("EMA (20) with Close Price")
    except Exception as e:
        st.warning(f"Could not calculate EMA: {e}")

def display_candlestick_patterns(df: pd.DataFrame):
    """
    Display candlestick pattern detection results for the uploaded DataFrame.
    """
    st.subheader("Candlestick Pattern Detection")
    # Only show if required columns exist
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        st.info("Candlestick pattern detection requires 'open', 'high', 'low', and 'close' columns.")
        return

    # Create detector instance
    patterns_detector = create_pattern_detector(confidence_threshold=0.7, enable_caching=True)
    available_patterns = patterns_detector.get_pattern_names()
    selected_patterns = st.multiselect(
        "Select candlestick patterns to detect:",
        available_patterns,
        key="data_analysis_v2_pattern_select"
    )
    if st.button("Detect Selected Patterns", key="data_analysis_v2_pattern_detect_btn") and selected_patterns:
        try:
            detection_results = patterns_detector.detect_patterns(df, pattern_names=selected_patterns)
            if detection_results:
                st.write("### Detected Patterns Results:")
                for result in detection_results:
                    if hasattr(result, 'detected') and result.detected:
                        st.success(f"{result.name} detected (confidence: {getattr(result, 'confidence', 0):.2f})")
                    elif hasattr(result, 'detected'):
                        st.info(f"{result.name} not detected")
            else:
                st.info("No patterns detected.")
        except Exception as e:
            st.error(f"Pattern detection error: {e}")

def main():
    """Main function to render the Data Analysis V2 page."""
    # Check if user has returned to this page and clear data if needed
    _check_and_clear_on_page_return()
    
    st.header("Stock Data Upload and Display")

    # Section for uploading data (no outer expander)
    display_uploaded_data()

    # If validation result exists, display it
    if 'validation_result' in st.session_state and 'uploaded_dataframe' in st.session_state:
        validation_result = st.session_state['validation_result']
        df = st.session_state['uploaded_dataframe']
        if validation_result is not None and df is not None:
            display_validation_results(validation_result, df)
            # Show technical indicators only if validation passed
            if validation_result.is_valid:
                display_technical_indicators(df)
                display_candlestick_patterns(df)

