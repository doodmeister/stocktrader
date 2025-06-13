"""
Enhanced Technical Analysis Dashboard - Cleaned & Hardened
"""

# --- Imports ---
import streamlit as st
import pandas as pd
from typing import Optional
from core.streamlit.session_manager import SessionManager
# import utils.chatgpt
from utils.chatgpt import get_chatgpt_insight
import plotly.express as px
import plotly.graph_objects as go
import time
import numpy as np
import json
import hashlib

# Existing imports (preserved per your requirements)
from utils.logger import get_dashboard_logger
from core.streamlit.dashboard_utils import setup_page
from core.streamlit.session_manager import SessionManager
from core.data_validator import DataValidator
from core.validation.validation_results import DataFrameValidationResult
from core.indicators.rsi import calculate_rsi
from core.indicators.macd import calculate_macd
from core.indicators.bollinger_bands import calculate_bollinger_bands
from core.technical_indicators import calculate_sma, calculate_ema
from patterns.patterns import create_pattern_detector, PatternResult
from patterns.pattern_utils import get_pattern_info


logger = get_dashboard_logger('data_analysis_v2_dashboard')

setup_page(
    title="📈 Stock Data Analysis V2",
    logger_name=__name__,
    sidebar_title="Dashboard Controls"
)

session_manager = SessionManager(namespace_prefix="data_analysis_v2")

data_validator = DataValidator(enable_api_validation=False)

# --- Session State Initialization ---
def _init_data_analysis_v2_state():
    """
    Initialize session state for the data analysis v2 page.
    
    Uses SessionManager.has_navigated_to_page() to determine if page-specific state
    (non-namespaced keys like 'uploaded_dataframe') should be cleared.
    SessionManager handles its own namespaced keys (clears them on navigation).
    This function ensures all necessary keys (namespaced and non-namespaced) have defaults.
    State is preserved during in-page reruns unless explicitly cleared by navigation or specific logic.
    """
    is_new_page_navigation = session_manager.has_navigated_to_page()

    if is_new_page_navigation:
        logger.info(
            f"Data Analysis V2: Navigation to page detected by SessionManager for namespace '{session_manager.namespace}'. "
            f"Clearing additional page-specific (non-namespaced) state."
        )
        # These are keys not automatically managed by SessionManager's namespace clearing
        # because they are not prefixed with the namespace.
        non_namespaced_keys_to_clear = [
            'uploaded_dataframe',
            'uploaded_file_name',
            'validation_result',
            'data_analysis_v2_last_file_id'  # Important for new file upload detection
        ]
        
        # One-time cleanup for potentially old non-namespaced keys that are now namespaced.
        # This helps transition if old versions of these keys exist without SM prefix.
        legacy_keys_to_check_and_clear = [
            'data_analysis_v2_detected_patterns',
            'data_analysis_v2_summary',
            'data_analysis_v2_gpt_response',
            'data_analysis_v2_pattern_detection_attempted',
            'data_analysis_v2_send_to_gpt_requested',
            'data_analysis_v2_summary_for_gpt',
        ]
        non_namespaced_keys_to_clear.extend(legacy_keys_to_check_and_clear)

        for k in non_namespaced_keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]
                logger.debug(f"Cleared non-namespaced/legacy key: {k}")
    else:
        logger.info(
            f"Data Analysis V2: In-page rerun for namespace '{session_manager.namespace}'. Not clearing non-namespaced state."
        )

    # Define non-namespaced state and their defaults
    non_namespaced_defaults = {
        'validation_result': None,
        # 'uploaded_dataframe', 'uploaded_file_name', 'data_analysis_v2_last_file_id' are set by file upload logic
        # or cleared on navigation. No explicit default needed here if their absence is handled.
    }
    for k, v in non_namespaced_defaults.items():
        if k not in st.session_state: # Initialize if missing (e.g. first ever run)
            st.session_state[k] = v
            logger.debug(f"Defaulted non-namespaced key '{k}' to {v}")

    # Define page-specific state (managed by SessionManager) and their defaults
    # SessionManager already clears these on new page navigation.
    # This loop ensures they are set to a default if they become unset for any other reason,
    # or on the very first run for that key after SM has processed the navigation.
    page_state_definitions = { # key_suffix: default_value
        'detected_patterns': [],
        'summary': '',
        'gpt_response': '',
        'pattern_detection_attempted': False,
        'new_file_uploaded_this_run': False, 
        'send_to_gpt_requested': False,
        'summary_for_gpt': '',
    }

    for key_suffix, default_value in page_state_definitions.items():
        _unique_sentinel = object() # To differentiate between a stored None and a missing key
        current_sm_value = session_manager.get_page_state(key_suffix, _unique_sentinel)

        # Special handling for 'send_to_gpt_requested'
        if key_suffix == 'send_to_gpt_requested':
            if is_new_page_navigation: # SM cleared it, must set to default False
                session_manager.set_page_state(key_suffix, False)
                logger.info(f"SM state '{key_suffix}' reset to False on new page entry.")
            elif current_sm_value is _unique_sentinel: # In-page rerun, but key is missing
                session_manager.set_page_state(key_suffix, False)
                logger.info(f"SM state '{key_suffix}' initialized to False (was missing on in-page rerun).")
            # Else: preserve existing value during in-page rerun. Managed by other logic flows.
        
        # Special handling for 'new_file_uploaded_this_run'
        elif key_suffix == 'new_file_uploaded_this_run':
            # This is a transient flag, primarily managed by display_uploaded_data().
            # Ensure it's False if new page or missing.
            if is_new_page_navigation or current_sm_value is _unique_sentinel:
                session_manager.set_page_state(key_suffix, False)
                logger.debug(f"SM state '{key_suffix}' initialized/reset to False.")
        
        # For other general SM-managed state keys
        else:
            # If new page (SM cleared it) OR key is missing on in-page rerun, set to default.
            if is_new_page_navigation or current_sm_value is _unique_sentinel:
                session_manager.set_page_state(key_suffix, default_value)
                logger.debug(f"SM state '{key_suffix}' set to default '{default_value}' (new page or missing).")

# --- Session state initialization: optimized for page-only execution from main.py ---
_init_data_analysis_v2_state()

# --- DataFrame Normalization & Validation ---
def normalize_and_validate_ohlc(df: pd.DataFrame) -> Optional[DataFrameValidationResult]:
    """
    Normalize OHLCV columns to lowercase, enforce float types,
    and run validation. Returns a DataFrameValidationResult.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return None
    try:
        df.columns = [c.lower() for c in df.columns]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # Drop rows with missing OHLC
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        # Update in session
        st.session_state['uploaded_dataframe'] = df
        # Run core validation
        validation_result = data_validator.validate_dataframe(
            df,
            required_cols=['open', 'high', 'low', 'close'],
            check_ohlcv=True,
            min_rows=1,
            max_rows=100_000,
            detect_anomalies_level='basic',
            max_null_percentage=0.05
        )
        st.session_state['validation_result'] = validation_result
        return validation_result
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return DataFrameValidationResult(
            is_valid=False,
            errors=[f"Validation error: {str(e)}"],
            details=None,
            error_details=None,
            validated_data=None
        )

# --- File Upload/Display Handler ---
def display_uploaded_data():
    """Handle file upload, normalization, and display."""
    # Default to False, set to True only if a new file is successfully processed in this call
    # This key is now managed by SessionManager
    session_manager.set_page_state('new_file_uploaded_this_run', False)

    uploaded_file = session_manager.create_file_uploader(
        "Upload Stock Data CSV",
        file_uploader_name="data_analysis_v2_file_uploader" # SM will namespace this key
    )
    
    if uploaded_file is not None:
        current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        # 'data_analysis_v2_last_file_id' is a non-namespaced key, managed directly
        previous_file_id = st.session_state.get('data_analysis_v2_last_file_id', '')
        
        if current_file_id != previous_file_id:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state['uploaded_dataframe'] = df # Store raw DataFrame (non-namespaced)
                
                # New file uploaded. Clear non-namespaced validation result.
                st.session_state.pop('validation_result', None)
                
                # Clear all SM-managed page-specific analysis states
                # This is more robust than manually listing keys.
                # SessionManager's _clear_all_sm_managed_state_for_namespace is too broad as it clears context too.
                # We need a way to clear specific page state keys registered with SM.
                # For now, let's list them, assuming they are all under the current SM's namespace.
                sm_keys_to_clear_on_new_file = [
                    'detected_patterns',
                    'summary',
                    'gpt_response',
                    'send_to_gpt_requested',
                    'summary_for_gpt',
                    'pattern_detection_attempted'
                ]
                for key_suffix in sm_keys_to_clear_on_new_file:
                    session_manager.clear_page_state(key_suffix) # Use clear_page_state
                    logger.debug(f"Cleared SM state '{key_suffix}' due to new file upload.")

                # Signal that a new file was uploaded and processed in THIS run cycle
                session_manager.set_page_state('new_file_uploaded_this_run', True)
                
                # Store the filename and file ID for tracking (non-namespaced)
                st.session_state['uploaded_file_name'] = uploaded_file.name
                st.session_state['data_analysis_v2_last_file_id'] = current_file_id
                st.success("File uploaded successfully. Previous analysis results cleared.")
            except Exception as e:
                st.error(f"Failed to read uploaded file: {e}")
                st.session_state.pop('uploaded_dataframe', None) # Clear potentially bad df
                # new_file_uploaded_this_run remains False (set at start of function)
    
    # Display logic (uses non-namespaced 'uploaded_dataframe')
    if 'uploaded_dataframe' in st.session_state and st.session_state.get('uploaded_dataframe') is not None:
        df_display = st.session_state['uploaded_dataframe']
        file_name = st.session_state.get('uploaded_file_name', 'Unknown file')
        st.subheader("Uploaded Data")
        st.info(f"File: **{file_name}** | Shape: {df_display.shape[0]} rows × {df_display.shape[1]} columns")
        st.dataframe(df_display)
        if not df_display.empty:
            _display_chart(df_display)
    else:
        st.info("Please upload a CSV file to see the data and chart.")

def _display_chart(df: pd.DataFrame):
    """Display price line chart."""
    try:
        date_col = _find_column(df, ['date', 'timestamp', 'datetime', 'time'])
        close_col = _find_column(df, ['close', 'closing_price', 'adj_close', 'adjusted_close'])
        if date_col:
            df = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            st.session_state['uploaded_dataframe'] = df
        if date_col and close_col:
            fig = px.line(df, x=date_col, y=close_col, title=f'Stock Prices ({close_col} over {date_col})')
            st.plotly_chart(fig, use_container_width=True)
        elif close_col:
            fig = px.line(df, y=close_col, title=f'Stock Prices ({close_col} over Index)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Could not find suitable price column. Columns: {list(df.columns)}")
    except Exception as e:
        st.error(f"Could not display chart: {e}")

def _find_column(df: pd.DataFrame, column_names: list) -> Optional[str]:
    for col in df.columns:
        if col.lower() in column_names:
            return col
    return None

def display_validation_results(validation_result: DataFrameValidationResult, df: pd.DataFrame):
    """Display validation pass/fail message."""
    if validation_result.is_valid:
        st.success("✅ Data Validation Passed")
    else:
        st.error("❌ Data Validation Failed")
        if validation_result.errors:
            for i, error in enumerate(validation_result.errors, 1):
                st.error(f"{i}. {error}")

# --- Technical Indicators Section ---
def get_technical_indicators_summary(df: pd.DataFrame) -> str:
    """Generate comprehensive summary of all technical indicators."""
    summary_parts = []
    if 'close' in df.columns and not df.empty:
        summary_parts.append("Technical Indicators Summary:")
        
        # RSI
        try:
            rsi = calculate_rsi(df, length=14)
            latest_rsi = rsi.iloc[-1]
            summary_parts.append(f"- RSI(14): {latest_rsi:.2f}")
            if latest_rsi > 70:
                summary_parts.append("  → Overbought condition")
            elif latest_rsi < 30:
                summary_parts.append("  → Oversold condition")
        except Exception:
            summary_parts.append("- RSI(14): Error calculating")
        
        # MACD
        try:
            macd_out = calculate_macd(df)
            if isinstance(macd_out, tuple) and len(macd_out) >= 2:
                macd, signal = macd_out[0], macd_out[1]
                latest_macd = macd.iloc[-1]
                latest_signal = signal.iloc[-1]
                summary_parts.append(f"- MACD: {latest_macd:.4f}, Signal: {latest_signal:.4f}")
                if latest_macd > latest_signal:
                    summary_parts.append("  → MACD above signal (bullish)")
                else:
                    summary_parts.append("  → MACD below signal (bearish)")
        except Exception:
            summary_parts.append("- MACD: Error calculating")
        
        # Bollinger Bands
        try:
            bb_up, bb_mid, bb_low = calculate_bollinger_bands(df)
            latest_close = df['close'].iloc[-1]
            latest_upper = bb_up.iloc[-1]
            latest_middle = bb_mid.iloc[-1]
            latest_lower = bb_low.iloc[-1]
            summary_parts.append(f"- Bollinger Bands: Upper: {latest_upper:.2f}, Middle: {latest_middle:.2f}, Lower: {latest_lower:.2f}")
            summary_parts.append(f"- Current Price: {latest_close:.2f}")
            if latest_close > latest_upper:
                summary_parts.append("  → Price above upper band (potentially overbought)")
            elif latest_close < latest_lower:
                summary_parts.append("  → Price below lower band (potentially oversold)")
        except Exception:
            summary_parts.append("- Bollinger Bands: Error calculating")
        
        # SMA
        try:
            sma = calculate_sma(df, length=20)
            latest_sma = sma.iloc[-1]
            latest_close = df['close'].iloc[-1]
            summary_parts.append(f"- SMA(20): {latest_sma:.2f}")
            if latest_close > latest_sma:
                summary_parts.append("  → Price above SMA (bullish trend)")
            else:
                summary_parts.append("  → Price below SMA (bearish trend)")
        except Exception:
            summary_parts.append("- SMA(20): Error calculating")
        
        # EMA
        try:
            ema = calculate_ema(df, length=20)
            latest_ema = ema.iloc[-1]
            latest_close = df['close'].iloc[-1]
            summary_parts.append(f"- EMA(20): {latest_ema:.2f}")
            if latest_close > latest_ema:
                summary_parts.append("  → Price above EMA (bullish trend)")
            else:
                summary_parts.append("  → Price below EMA (bearish trend)")
        except Exception:
            summary_parts.append("- EMA(20): Error calculating")
            
    else:
        summary_parts.append("No 'close' column or data is empty.")
    
    return "\n".join(summary_parts)

# --- Technical Indicators Section ---
def display_technical_indicators(df: pd.DataFrame):
    st.subheader("Technical Indicators")
    if 'close' not in df.columns:
        st.info("No 'close' column found. Technical indicators require a 'close' price column.")
        return
    try:
        # RSI
        rsi = calculate_rsi(df, length=14)
        st.line_chart(rsi, use_container_width=True)
        st.caption("RSI (14)")
    except Exception as e:
        st.warning(f"Could not calculate RSI: {e}")
    try:
        # MACD
        macd_output = calculate_macd(df)
        macd, signal = None, None
        if isinstance(macd_output, tuple) and len(macd_output) >= 2:
            macd, signal = macd_output[0], macd_output[1]
        elif isinstance(macd_output, pd.DataFrame):
            columns = list(macd_output.columns)
            macd_col = next((c for c in columns if c.lower() in ['macd', 'macd_line']), None)
            signal_col = next((c for c in columns if c.lower() in ['signal', 'signal_line']), None)
            if macd_col:
                macd = macd_output[macd_col]
            if signal_col:
                signal = macd_output[signal_col]
        elif isinstance(macd_output, dict):
            macd = macd_output.get('macd') or macd_output.get('macd_line')
            signal = macd_output.get('signal') or macd_output.get('signal_line')
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
    try:
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=bb_upper, mode='lines', name='Upper Band'))
        fig.add_trace(go.Scatter(y=bb_middle, mode='lines', name='Middle Band'))
        fig.add_trace(go.Scatter(y=bb_lower, mode='lines', name='Lower Band'))
        fig.add_trace(go.Scatter(y=df['close'], mode='lines', name='Close Price', line=dict(color='gray', dash='dot')))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Bollinger Bands")
    except Exception as e:
        st.warning(f"Could not calculate Bollinger Bands: {e}")
    try:
        # SMA
        sma = calculate_sma(df, length=20)
        close = df['close']
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=close, mode='lines', name='Close Price', line=dict(color='gray', dash='dot')))
        fig.add_trace(go.Scatter(y=sma, mode='lines', name='SMA (20)', line=dict(color='blue')))
        min_y = min(sma.min(), close.min())
        max_y = max(sma.max(), close.max())
        margin = (max_y - min_y) * 0.05 if max_y > min_y else 1
        fig.update_yaxes(range=[min_y - margin, max_y + margin])
        st.plotly_chart(fig, use_container_width=True)
        st.caption("SMA (20) with Close Price")
    except Exception as e:
        st.warning(f"Could not calculate SMA: {e}")
    try:
        # EMA
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

# --- Candlestick Pattern Detection Section ---
def get_patterns_summary(detected_patterns: list) -> str:
    """Generate summary of detected candlestick patterns."""
    if detected_patterns:
        pattern_lines = [
            f"- Index {dp['index']}: {dp['name']} (Confidence: {dp.get('confidence', 0):.2f})"
            for dp in detected_patterns[-20:]
        ]
        return "Detected Candlestick Patterns:\n" + "\n".join(pattern_lines)
    else:
        return "Detected Candlestick Patterns:\nNo patterns detected."

def display_candlestick_patterns(df: pd.DataFrame):
    st.subheader("Candlestick Pattern Detection")
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        st.info("Candlestick pattern detection requires 'open', 'high', 'low', and 'close' columns.")
        return
    patterns_detector = create_pattern_detector(confidence_threshold=0.7, enable_caching=True)
    available_patterns = patterns_detector.get_pattern_names()
    selected_patterns = st.multiselect(
        "Select candlestick patterns to detect:",
        available_patterns,
        key=session_manager.get_widget_key("multiselect", "data_analysis_v2_pattern_select")
    )
    if st.button("Detect Selected Patterns", key=session_manager.get_widget_key("button", "data_analysis_v2_pattern_detect_btn")) and selected_patterns:
        try:
            df_reset = df.reset_index(drop=True)
            current_detected_patterns = []
            for pattern_name in selected_patterns:
                min_rows = get_pattern_info(pattern_name).get('min_rows', 1)
                for i in range(min_rows - 1, len(df_reset)):
                    window = df_reset.iloc[i - min_rows + 1:i + 1]
                    detection_results = patterns_detector.detect_patterns(window, pattern_names=[pattern_name])
                    # Robust handling for detection_results (as you already have):
                    if isinstance(detection_results, PatternResult):
                        detection_results = [detection_results]
                    elif isinstance(detection_results, list):
                        flat_results = []
                        for item in detection_results:
                            if isinstance(item, list):
                                st.error(f"Pattern detection returned nested list: {item}")
                                flat_results.extend(item)
                            else:
                                flat_results.append(item)
                        detection_results = flat_results
                    else:
                        st.error(f"Pattern detection returned unexpected type: {type(detection_results)}")
                        detection_results = []
                    for result in detection_results:
                        if hasattr(result, 'detected') and result.detected:
                            result_dict = result.__dict__.copy()
                            result_dict['index'] = i
                            current_detected_patterns.append(result_dict)
            
            session_manager.set_page_state('detected_patterns', current_detected_patterns)
            session_manager.set_page_state('pattern_detection_attempted', True)
            
            # Generate comprehensive summary: technical indicators + patterns
            tech_summary = get_technical_indicators_summary(df)
            patterns_summary = get_patterns_summary(current_detected_patterns)
            full_summary = f"{tech_summary}\\n\\n{patterns_summary}"
            session_manager.set_page_state('summary', full_summary)
            st.success(f"Detected {len(current_detected_patterns)} instances of selected patterns.")
        except Exception as e:
            st.error(f"Pattern detection error: {e}")
            session_manager.set_page_state('pattern_detection_attempted', True)

    # Retrieve from session state for display after button press or if already populated
    detected_patterns_for_display = session_manager.get_page_state('detected_patterns', [])
    
    # Chart & Table
    if detected_patterns_for_display:
        st.write("### Candlestick Pattern Chart")
        fig = go.Figure()
        # Main price candlestick trace
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='green',
            decreasing_line_color='red',
            opacity=0.4
        ))
        pattern_colors = {}
        color_palette = ["red", "blue", "green", "orange", "purple", "brown", "pink", "cyan", "magenta", "yellow"]
        for idx, pattern in enumerate(set(dp['name'] for dp in detected_patterns_for_display)):
            pattern_colors[pattern] = color_palette[idx % len(color_palette)]
        for dp in detected_patterns_for_display:
            i = dp['index']
            pattern = dp['name']
            color = pattern_colors[pattern]
            # Overlay detected pattern as a single-candle trace
            fig.add_trace(go.Candlestick(
                x=[df.index[i]],
                open=[df['open'].iloc[i]],
                high=[df['high'].iloc[i]],
                low=[df['low'].iloc[i]],
                close=[df['close'].iloc[i]],
                name=pattern,
                increasing_line_color=color,
                decreasing_line_color=color,
                showlegend=False,
                opacity=1.0,
                whiskerwidth=0.8
            ))
            fig.add_annotation(
                x=df.index[i],
                y=df['high'].iloc[i] * 1.01,
                text=pattern,
                showarrow=False,
                font=dict(color=color, size=12, family="Arial Black"),
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor=color,
                borderwidth=1
            )
        if pd.api.types.is_datetime64_any_dtype(df.index):
            fig.update_xaxes(type='date', tickmode='auto')
        else:
            fig.update_xaxes(type='category', tickmode='linear')
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        # Table of detected patterns
        st.write("#### Detected Patterns Table")
        detected_df = pd.DataFrame(detected_patterns_for_display)
        if not detected_df.empty:
            if 'timestamp' in df.columns:
                detected_df['timestamp'] = detected_df['index'].apply(lambda i: df['timestamp'].iloc[i])
            elif 'date' in df.columns:
                detected_df['date'] = detected_df['index'].apply(lambda i: df['date'].iloc[i])
            grouped = detected_df.groupby('index').agg({
                'name': lambda x: ', '.join(x),
                'confidence': list,
                **({'timestamp': 'first'} if 'timestamp' in detected_df.columns else {}),
                **({'date': 'first'} if 'date' in detected_df.columns else {})
            }).reset_index()
            cols = ['index']
            if 'timestamp' in grouped.columns: cols.append('timestamp')
            if 'date' in grouped.columns: cols.append('date')
            cols += ['name', 'confidence']
            st.dataframe(grouped[cols], use_container_width=True)
        else:
            st.info("No patterns detected.")

# --- Summary and ChatGPT Review ---
def display_summary_and_gpt(df: pd.DataFrame):
    """
    Display summary and ChatGPT analysis section.
    Now available immediately after technical indicators are displayed,
    regardless of whether pattern detection has been attempted.
    """
    st.subheader("Analysis Summary & ChatGPT Review")
      # Generate technical indicators summary immediately
    tech_summary = get_technical_indicators_summary(df)
      # Get pattern summary if patterns have been detected
    detected_patterns = session_manager.get_page_state('detected_patterns', [])
    patterns_summary = get_patterns_summary(detected_patterns)
    
    # Check if we have patterns or if detection was attempted
    has_patterns = len(detected_patterns) > 0
    pattern_detection_attempted = session_manager.get_page_state('pattern_detection_attempted', False)
    
    # Debug information
    logger.info(f"Summary generation - has_patterns: {has_patterns}, detection_attempted: {pattern_detection_attempted}, patterns_count: {len(detected_patterns)}")
      # Combine summaries based on whether we have actual patterns
    if has_patterns or pattern_detection_attempted:
        # Include patterns if we have them or detection was attempted
        full_summary = f"{tech_summary}\n\n{patterns_summary}"
        logger.info("Including patterns in summary")
    else:
        # Only technical indicators if no pattern detection yet
        full_summary = f"{tech_summary}\n\nCandlestick Patterns:\nNo pattern detection attempted yet. Use the pattern detection section above to analyze candlestick patterns."
        logger.info("No patterns detected, using default message")
    
    # Store the current summary
    session_manager.set_page_state('summary', full_summary)
    summary = full_summary # Use local variable for clarity in this function
    
    # Clear previous GPT response if summary has changed significantly
    previous_summary_for_gpt = session_manager.get_page_state('summary_for_gpt', '')
    if previous_summary_for_gpt and previous_summary_for_gpt != summary:
        session_manager.set_page_state('gpt_response', '')
        logger.info("Cleared GPT response due to summary change.")
    
    send_btn = session_manager.create_button(
        "Send Summary to ChatGPT for Review",
        button_name="data_analysis_v2_send_to_gpt_btn" # SM will namespace this
    )
    if send_btn:
        logger.info("ChatGPT button clicked - setting request flag")
        # Flag value logging for send_to_gpt_requested
        flag_key = 'send_to_gpt_requested'
        logger.info(f"Flag '{flag_key}' value before setting: {session_manager.get_page_state(flag_key, 'NOT_SET')}")
        session_manager.set_page_state(flag_key, True)
        logger.info(f"Flag '{flag_key}' value after setting: {session_manager.get_page_state(flag_key, 'NOT_SET')}")
        st.rerun()  # Force immediate processing of the request
    # Note: ChatGPT processing is now handled in the main run() method 
    # to prevent stuck states when validation fails

    # Display the current summary for reference
    st.subheader("Current Analysis Summary")
    st.text_area("Summary that will be sent to ChatGPT:", value=summary, height=200, disabled=True)

    # Display ChatGPT response here
    _render_chatgpt_response_section()

# --- Main Dashboard Run ---
# (Removed _display_gpt_response method from class and created _render_chatgpt_response_section below)

# Module-level function to render ChatGPT response
def _render_chatgpt_response_section():
    """Display ChatGPT response section if available."""
    # Explicitly log what's retrieved from session state via SessionManager
    gpt_response_key = 'gpt_response'
    raw_gpt_response_from_session = session_manager.get_page_state(gpt_response_key)
    logger.info(f"_render_chatgpt_response_section: Raw response from SM for key '{gpt_response_key}': '{raw_gpt_response_from_session}' (Type: {type(raw_gpt_response_from_session)})")

    gpt_resp = session_manager.get_page_state(gpt_response_key, '') # Default to empty string
    logger.info(f"_render_chatgpt_response_section: Value of gpt_resp for conditional: '{gpt_resp}' (Type: {type(gpt_resp)})")

    if gpt_resp:
        logger.info("_render_chatgpt_response_section: Condition 'if gpt_resp' is TRUE. Displaying response.")
        st.subheader("ChatGPT Review Response")
        col1, col2 = st.columns([4, 1])
        with col2:
            if session_manager.create_button("Clear Response", button_name="data_analysis_v2_clear_gpt_btn"):
                session_manager.set_page_state(gpt_response_key, '')
                logger.info(f"Cleared SM state for '{gpt_response_key}' via button.")
                st.rerun()
        with col1:
            st.write(gpt_resp)
    else:
        logger.info("_render_chatgpt_response_section: Condition 'if gpt_resp' is FALSE. Not displaying response.")

class DataAnalysisV2Dashboard:
    def run(self):
        # Handle any pending ChatGPT requests FIRST, before any session state initialization
        self._handle_pending_gpt_requests()
        
        # REMOVED: self._display_gpt_response() call from here
        
        st.header("Stock Data Upload and Display")
        display_uploaded_data()
        
        df_from_session = st.session_state.get('uploaded_dataframe')
        if not isinstance(df_from_session, pd.DataFrame):
            return
        df: pd.DataFrame = df_from_session
        validation_result = st.session_state.get('validation_result')
        # Use SessionManager to get this state variable
        new_file_just_uploaded = session_manager.get_page_state('new_file_uploaded_this_run', False)
        if new_file_just_uploaded or not isinstance(validation_result, DataFrameValidationResult):
            normalize_and_validate_ohlc(df)
            df_after_norm_val = st.session_state.get('uploaded_dataframe')
            if isinstance(df_after_norm_val, pd.DataFrame):
                df = df_after_norm_val
            else:
                st.error("Data normalization failed. Cannot proceed with analysis.")
                return
            validation_result = st.session_state.get('validation_result')
        
        if validation_result is not None and getattr(validation_result, "is_valid", False):
            # ONLY if valid!
            display_validation_results(validation_result, df)
            display_technical_indicators(df)
            display_candlestick_patterns(df)
            display_summary_and_gpt(df)
        elif validation_result is not None:
            # Present but failed
            display_validation_results(validation_result, df)
            st.warning("Data validation failed. Cannot display technical indicators, candlestick patterns, or summary.")
        else:
            # Not yet validated
            st.info("Data is available but has not been validated yet. Please ensure the data is processed.")
    
    def _handle_pending_gpt_requests(self):
        """
        Handle any pending ChatGPT requests to prevent stuck states.
        This ensures the flag gets cleared even if validation fails.
        Uses SessionManager for state access.
        """
        send_requested_key = 'send_to_gpt_requested'
        summary_key = 'summary'
        gpt_response_key = 'gpt_response'
        summary_for_gpt_key = 'summary_for_gpt'

        logger.info(f"Checking for pending ChatGPT requests. Flag '{send_requested_key}' status: {session_manager.get_page_state(send_requested_key, False)}")
        
        if session_manager.get_page_state(send_requested_key, False):
            logger.info("Found pending ChatGPT request - processing now")
            # Clear the flag immediately to prevent infinite loops
            session_manager.set_page_state(send_requested_key, False)
            
            # Try to get the summary from session state via SessionManager
            summary_text = session_manager.get_page_state(summary_key, '')
            
            if not summary_text:
                st.error("No analysis summary available to send to ChatGPT. Please ensure data is uploaded and analyzed first.")
                return
            
            logger.info("Processing ChatGPT request for technical analysis")
            st.info("Processing ChatGPT request...")
            
            # Create a placeholder for the spinner and result
            with st.spinner("Contacting ChatGPT API... This may take up to 30 seconds."):
                try:
                    logger.info("Calling get_chatgpt_insight function")
                    api_response = get_chatgpt_insight(summary_text)
                    
                    logger.info(f"ChatGPT API response received: {len(api_response) if api_response else 0} characters")
                    
                    response_to_store = ""
                    if api_response and isinstance(api_response, str) and api_response.strip():
                        if api_response.startswith("Error:"):
                            response_to_store = api_response
                            st.error(f"❌ {api_response}")
                        else:
                            response_to_store = api_response
                            st.success("✅ ChatGPT response received successfully!")
                    elif api_response == "":
                        response_to_store = "ChatGPT returned an empty response."
                        st.warning("⚠️ ChatGPT returned an empty response.")
                    else:
                        response_to_store = "ChatGPT returned an unexpected response."
                        st.warning("⚠️ ChatGPT returned an unexpected response.")
                    
                    session_manager.set_page_state(gpt_response_key, response_to_store)
                    session_manager.set_page_state(summary_for_gpt_key, summary_text)
                    logger.info(f"ChatGPT response stored in SM state for keys '{gpt_response_key}' and '{summary_for_gpt_key}'.")
                    
                except Exception as e:
                    error_msg = f"Error: {e}"
                    session_manager.set_page_state(gpt_response_key, error_msg)
                    st.error(f"❌ ChatGPT error: {e}")
                    logger.error(f"ChatGPT API error: {e}")


DataAnalysisV2Dashboard().run()
