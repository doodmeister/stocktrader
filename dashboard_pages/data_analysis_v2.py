"""
Enhanced Technical Analysis Dashboard - Cleaned & Hardened

- Consistent OHLCV column normalization
- Bulletproof type enforcement for numeric columns
- Streamlined session state management
- Comprehensive error feedback
"""

import streamlit as st
from utils.chatgpt import get_chatgpt_insight  # ChatGPT helper
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import numpy as np
import json
import hashlib
from typing import Optional, Dict, Any

# Existing imports (preserved per your requirements)
from utils.logger import get_dashboard_logger
from core.streamlit.dashboard_utils import setup_page
from core.streamlit.session_manager import create_session_manager
from core.data_validator import DataValidator
from core.validation.validation_results import DataFrameValidationResult
from core.indicators.rsi import calculate_rsi
from core.indicators.macd import calculate_macd
from core.indicators.bollinger_bands import calculate_bollinger_bands
from core.technical_indicators import calculate_sma, calculate_ema
from patterns.patterns import CandlestickPatterns, create_pattern_detector

logger = get_dashboard_logger('data_analysis_v2_dashboard')

setup_page(
    title="📈 Stock Data Analysis V2",
    logger_name=__name__,
    sidebar_title="Dashboard Controls"
)

session_manager = create_session_manager(page_name="data_analysis_v2")
data_validator = DataValidator(enable_api_validation=False)

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

# --- Session State Clear on Page Return ---
def _check_and_clear_on_page_return():
    current_page = "data_analysis_v2.py"
    page_history = st.session_state.get('page_history', [])
    current_page_from_controller = st.session_state.get('current_page', 'home')
    if (current_page_from_controller == current_page and
        len(page_history) >= 2 and
        page_history[-2] != current_page):
        keys_to_clear_on_return = [
            'uploaded_dataframe', 'uploaded_file_name', 'validation_result',
            'data_analysis_v2_detected_patterns', 'data_analysis_v2_summary',
            'data_analysis_v2_gpt_response', 'data_analysis_v2_show_gpt_response'
        ]
        for key in keys_to_clear_on_return:
            st.session_state.pop(key, None)

_check_and_clear_on_page_return()

# --- File Upload/Display Handler ---
def display_uploaded_data():
    """Handle file upload, normalization, and display."""
    for key in [
        'data_analysis_v2_csv_uploader',
        'data_analysis_v2_csv_uploader_file_uploader',
        'data_analysis_v2_csv_upload',
    ]:
        st.session_state.pop(key, None)
    # Use a unique file_uploader_name for this page
    uploaded_file = session_manager.create_file_uploader(
        label="Upload your CSV file",
        type=["csv"],
        file_uploader_name="data_analysis_v2_csv_upload",
        help="Upload a CSV with columns like 'timestamp', 'close', etc."
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['uploaded_file_name'] = uploaded_file.name
            # Normalize, enforce types, and validate
            validation_result = normalize_and_validate_ohlc(df)
            # Remove any previous summaries/gpt results if new file uploaded
            for key in ['data_analysis_v2_summary', 'data_analysis_v2_gpt_response']:
                st.session_state.pop(key, None)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            for key in ['uploaded_dataframe', 'uploaded_file_name', 'validation_result']:
                st.session_state.pop(key, None)
            return
    # Display
    if 'uploaded_dataframe' in st.session_state and st.session_state.get('uploaded_dataframe') is not None:
        df_display = st.session_state['uploaded_dataframe']
        file_name = st.session_state.get('uploaded_file_name', 'Unknown file')
        st.subheader("Uploaded Data")
        st.info(f"📁 File: **{file_name}** | Shape: {df_display.shape[0]} rows × {df_display.shape[1]} columns")
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
        key="data_analysis_v2_pattern_select"
    )
    detected_patterns = st.session_state.get('data_analysis_v2_detected_patterns', [])
    if st.button("Detect Selected Patterns", key="data_analysis_v2_pattern_detect_btn") and selected_patterns:
        try:
            df = df.reset_index(drop=True)
            window_size = max([patterns_detector.get_pattern_info(p)['min_rows'] for p in selected_patterns])
            detected_patterns = []
            for i in range(window_size - 1, len(df)):
                window = df.iloc[i - window_size + 1:i + 1]
                detection_results = patterns_detector.detect_patterns(window, pattern_names=selected_patterns)
                for result in detection_results:
                    if hasattr(result, 'detected') and result.detected:
                        detected_patterns.append({
                            'index': i,
                            'pattern': result.name,
                            'confidence': getattr(result, 'confidence', 0.0)
                        })
            st.session_state['data_analysis_v2_detected_patterns'] = detected_patterns
            st.session_state.pop('data_analysis_v2_summary', None)
        except Exception as e:
            st.error(f"Pattern detection error: {e}")
            detected_patterns = []
            st.session_state['data_analysis_v2_detected_patterns'] = detected_patterns
    detected_patterns = st.session_state.get('data_analysis_v2_detected_patterns', [])
    # Chart & Table
    if detected_patterns:
        st.write("### Candlestick Pattern Chart")
        fig = go.Figure()
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
        for idx, pattern in enumerate(set(dp['pattern'] for dp in detected_patterns)):
            pattern_colors[pattern] = color_palette[idx % len(color_palette)]
        for dp in detected_patterns:
            i = dp['index']
            pattern = dp['pattern']
            color = pattern_colors[pattern]
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
        detected_df = pd.DataFrame(detected_patterns)
        if not detected_df.empty:
            if 'timestamp' in df.columns:
                detected_df['timestamp'] = detected_df['index'].apply(lambda i: df['timestamp'].iloc[i])
            elif 'date' in df.columns:
                detected_df['date'] = detected_df['index'].apply(lambda i: df['date'].iloc[i])
            grouped = detected_df.groupby('index').agg({
                'pattern': lambda x: ', '.join(x),
                'confidence': list,
                **({'timestamp': 'first'} if 'timestamp' in detected_df.columns else {}),
                **({'date': 'first'} if 'date' in detected_df.columns else {})
            }).reset_index()
            cols = ['index']
            if 'timestamp' in grouped.columns: cols.append('timestamp')
            if 'date' in grouped.columns: cols.append('date')
            cols += ['pattern', 'confidence']
            st.dataframe(grouped[cols], use_container_width=True)
        else:
            st.info("No patterns detected.")

# --- Summary and ChatGPT Review ---
def generate_dashboard_summary(df, detected_patterns):
    """Generate summary of technical indicators and detected patterns."""
    summary_parts = []
    if 'close' in df.columns and not df.empty:
        summary_parts.append("Technical Indicators Summary:")
        try:
            rsi = calculate_rsi(df, length=14)
            summary_parts.append(f"- RSI(14): {rsi.iloc[-1]:.2f}")
        except Exception:
            summary_parts.append("- RSI(14): Error")
        try:
            macd_out = calculate_macd(df)
            if isinstance(macd_out, tuple) and len(macd_out) >= 2:
                macd, signal = macd_out[0], macd_out[1]
                summary_parts.append(f"- MACD: {macd.iloc[-1]:.2f}, Signal: {signal.iloc[-1]:.2f}")
        except Exception:
            summary_parts.append("- MACD: Error")
        try:
            bb_up, bb_mid, bb_low = calculate_bollinger_bands(df)
            summary_parts.append(f"- Bollinger Middle: {bb_mid.iloc[-1]:.2f}")
        except Exception:
            summary_parts.append("- Bollinger: Error")
    else:
        summary_parts.append("No 'close' column or data empty.")

    summary_parts.append("\nDetected Candlestick Patterns:")
    if detected_patterns:
        for dp in detected_patterns[-20:]:
            summary_parts.append(
                f"- Index {dp['index']}: {dp['pattern']} (Confidence: {dp.get('confidence', 0):.2f})"
            )
    else:
        summary_parts.append("No patterns detected.")
    return "\n".join(summary_parts)


def display_summary_and_gpt(df: pd.DataFrame, detected_patterns):
    """Display summary and handle ChatGPT review interactions. Always show if DataFrame is present."""
    if not (df is not None and isinstance(df, pd.DataFrame)):
        return

    # Ensure summary exists and is up to date
    summary_key = 'data_analysis_v2_summary'
    prev_summary = st.session_state.get(summary_key)
    prev_df_id = st.session_state.get('data_analysis_v2_summary_df_id')
    # Use id(df) to detect if the DataFrame has changed (new upload)
    curr_df_id = id(df)
    if prev_summary is None or prev_df_id != curr_df_id:
        st.session_state[summary_key] = generate_dashboard_summary(df, detected_patterns)
        st.session_state['data_analysis_v2_summary_df_id'] = curr_df_id
    summary = st.session_state[summary_key]

    # Show summary text area
    st.text_area(
        "Summary of Technical Indicators and Detected Patterns",
        value=summary,
        height=300,
        key="data_analysis_v2_summary_textarea"
    )

    # ChatGPT review button
    send_btn = session_manager.create_button(
        "Send Summary to ChatGPT for Review",
        "data_analysis_v2_send_to_gpt_btn"
    )
    if send_btn and not st.session_state.get('data_analysis_v2_waiting_for_gpt', False):
        st.session_state['data_analysis_v2_waiting_for_gpt'] = True
        st.session_state.pop('data_analysis_v2_gpt_response', None)
        st.rerun()

    # If waiting, call ChatGPT
    if st.session_state.get('data_analysis_v2_waiting_for_gpt', False):
        with st.spinner("Contacting ChatGPT..."):
            try:
                response = get_chatgpt_insight(st.session_state[summary_key])
            except Exception as e:
                response = f"Failed to send summary to ChatGPT: {e}"
            st.session_state['data_analysis_v2_gpt_response'] = response
            st.session_state['data_analysis_v2_waiting_for_gpt'] = False
            st.rerun()

    # Always show response if present
    gpt_resp = st.session_state.get('data_analysis_v2_gpt_response')
    if gpt_resp:
        st.subheader("ChatGPT Review Response")
        st.write(gpt_resp)

# --- Main Dashboard Run ---
class DataAnalysisV2Dashboard:
    def run(self):
        _check_and_clear_on_page_return()
        st.header("Stock Data Upload and Display")
        display_uploaded_data()
        if 'validation_result' in st.session_state and 'uploaded_dataframe' in st.session_state:
            validation_result = st.session_state['validation_result']
            df = st.session_state['uploaded_dataframe']
            if validation_result is not None and df is not None and isinstance(df, pd.DataFrame):
                display_validation_results(validation_result, df)
                if validation_result.is_valid:
                    display_technical_indicators(df)
                    display_candlestick_patterns(df)
        df = st.session_state.get('uploaded_dataframe', None)
        detected_patterns = st.session_state.get('data_analysis_v2_detected_patterns', [])
        # Always show summary and GPT UI if DataFrame is present
        if df is not None and isinstance(df, pd.DataFrame):
            display_summary_and_gpt(df, detected_patterns)

# --- Ensure dashboard runs when imported by main.py ---
DataAnalysisV2Dashboard().run()
