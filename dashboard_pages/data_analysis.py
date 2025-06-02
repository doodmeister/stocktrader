"""
Streamlined Technical Analysis Dashboard

Refactored to use the new centralized technical analysis architecture:
- core.technical_indicators: Pure calculation functions
- utils.technicals.analysis: High-level analysis classes

Optimized for conciseness while preserving all essential functionality.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Tuple, Optional
import os
from collections import defaultdict

# Core imports
from utils.logger import get_dashboard_logger
from core.dashboard_utils import (
    setup_page, handle_streamlit_error, initialize_dashboard_session_state
)
from core.data_validator import validate_dataframe
from core.session_manager import create_session_manager

# New centralized technical analysis imports
from core.technical_indicators import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands, 
    calculate_atr, IndicatorError
)
from utils.technicals.analysis import TechnicalAnalysis, compute_price_stats, compute_return_stats

# Pattern detection and utilities
from patterns.patterns import create_pattern_detector
from security.authentication import get_openai_api_key
from utils.chatgpt import get_chatgpt_insight as _get_chatgpt_insight

# Initialize the page and logger
logger = setup_page(
    title="📊 Technical Analysis Dashboard",
    logger_name=__name__,
    sidebar_title="Analysis Controls"
)

# Initialize SessionManager for conflict-free widget handling
session_manager = create_session_manager("data_analysis_v3")

@st.cache_data(show_spinner=False)
def load_df(uploaded_file) -> pd.DataFrame:
    """Load and return CSV as DataFrame with enhanced error handling and column mapping."""
    try:
        # Always reset file pointer to start
        uploaded_file.seek(0)
        
        # Log start of function
        logger.info(f"🔧 load_df called with file: {getattr(uploaded_file, 'name', 'unknown')}")
        
        # Simple, direct read - no complex parsing logic yet
        df = pd.read_csv(uploaded_file)
        
        # Log what we got immediately
        logger.info(f"📊 CSV loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"📋 Original columns: {list(df.columns)}")
        logger.info(f"📏 DataFrame shape: {df.shape}")
        
        # If we have data, process it
        if len(df) > 0:
            # Normalize column names (case-insensitive)
            original_cols = list(df.columns)
            df.columns = df.columns.str.lower().str.strip()
            logger.info(f"🔤 After normalization: {list(df.columns)}")
              # Show sample of first few rows
            logger.info(f"📝 First 2 rows:\n{df.head(2).to_string()}")
            
            # Convert numeric columns to proper data types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        logger.info(f"✅ Converted {col} to numeric")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to convert {col} to numeric: {e}")
            
            # Ensure we have the expected OHLCV columns
            expected_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in expected_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"⚠️ Missing expected columns: {missing_cols}")
                st.warning(f"⚠️ CSV is missing expected columns: {missing_cols}")
                st.info("💡 Expected columns: Open, High, Low, Close, Volume")
            else:
                logger.info(f"✅ All required OHLCV columns present")
            
            logger.info(f"🎯 Final result: {len(df)} rows, {list(df.columns)}")
            return df
        else:
            logger.error("❌ DataFrame is empty after loading")
            st.error("❌ CSV file appears to be empty")
            return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"💥 Failed to load CSV file: {e}")
        import traceback
        logger.error(f"💥 Traceback: {traceback.format_exc()}")
        st.error(f"❌ Failed to load CSV file: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def get_chatgpt_insight(summary: str) -> str:
    return _get_chatgpt_insight(summary)

@st.cache_data
def cached_compute_price_stats(df: pd.DataFrame) -> pd.DataFrame:
    return compute_price_stats(df)

@st.cache_data
def cached_compute_return_stats(df: pd.DataFrame) -> pd.DataFrame:
    return compute_return_stats(df)

@st.cache_data
def get_indicator_series(
    df: pd.DataFrame,
    rsi_period: int,
    macd_fast: int,
    macd_slow: int,
    bb_period: int,
    bb_std: int
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculate RSI, MACD lines, and Bollinger Bands using centralized indicators."""
    try:
        # Use centralized technical analysis functions with correct parameter names
        rsi = calculate_rsi(df, length=rsi_period)
        macd_line, signal_line, _ = calculate_macd(
            df, 
            fast=macd_fast, 
            slow=macd_slow, 
            signal=9
        )
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
            df, 
            length=bb_period, 
            std=bb_std
        )
        return rsi, macd_line, signal_line, bb_upper, bb_lower
    except IndicatorError as e:
        logger.error(f"Error calculating indicators: {e}")
        # Return empty series with same index as fallback
        empty_series = pd.Series(index=df.index, dtype=float)
        return empty_series, empty_series, empty_series, empty_series, empty_series

def get_pattern_results(df: pd.DataFrame, patterns: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """Detect and return selected candlestick patterns."""
    results: List[Dict[str, Any]] = []
    pattern_detector = create_pattern_detector()
    for i in range(len(df)):
        window = df.iloc[max(0, i-4):i+1]
        detected_results = pattern_detector.detect_patterns(window)
        # Extract pattern names from PatternResult objects
        detected = [result.name for result in detected_results if result.detected]
        for p in detected:
            if p in patterns:
                results.append({
                    "index": i,
                    "pattern": p,
                    "date": df.index[i] if hasattr(df.index, 'name') else i
                })
    return results

def validate_dataframe_for_analysis(df: pd.DataFrame) -> bool:
    """Ensure required columns are present using core validation with enhanced feedback."""
    if df.empty:
        st.error("⚠️ The uploaded CSV file appears to be empty or could not be parsed correctly.")
        st.info("💡 **Tips for CSV files:**")
        st.info("- Ensure your CSV has column headers: Date/Timestamp, Open, High, Low, Close, Volume")
        st.info("- Check that data is comma-separated (not semicolon or tab)")
        st.info("- Verify the file has actual data rows, not just headers")
        logger.error("Empty DataFrame provided for validation")
        return False
    
    # Show current DataFrame info for debugging
    st.info(f"📊 **Loaded Data Info:**")
    st.info(f"- **Rows:** {len(df)}")
    st.info(f"- **Columns:** {list(df.columns)}")
    
    # Check row count first
    if len(df) < 10:
        st.error(f"⚠️ Insufficient data: {len(df)} rows found, minimum 10 required for analysis.")
        st.info("💡 Please upload a CSV file with at least 10 rows of historical data.")
        logger.error(f"Insufficient data: {len(df)} rows (minimum 10 required)")
        return False
    
    required = ['open', 'high', 'low', 'close', 'volume']
    validation_result = validate_dataframe(df, required_cols=required)
    
    if not validation_result.is_valid:
        st.error("⚠️ **Data Validation Failed:**")
        for error in validation_result.errors:
            st.error(f"• {error}")
            logger.error(f"Validation error: {error}")
        
        # Provide helpful suggestions
        st.info("💡 **Expected CSV format:**")
        st.code("""
Date,Open,High,Low,Close,Volume
2024-01-01,100.00,105.00,99.00,104.00,1000000
2024-01-02,104.00,106.00,103.00,105.50,1100000
...
        """)
        return False
    
    # Success feedback
    logger.info(f"Data validation passed: {len(df)} rows, columns: {list(df.columns)}")
    return True

def plot_technical_indicators(
    rsi: pd.Series,
    macd_line: pd.Series,
    signal_line: pd.Series,
    df: pd.DataFrame,
    upper_band: pd.Series,
    lower_band: pd.Series,
    width: int = 1200,
    height: int = 400
):
    """Streamlined technical indicators plotting."""
    st.markdown("### 📈 Technical Indicator Analysis")
    
    # Create subplot for all indicators
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('RSI', 'MACD', 'Price with Bollinger Bands'),
        vertical_spacing=0.08,
        row_heights=[0.25, 0.25, 0.5]
    )
      # RSI subplot
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi, mode='lines', name='RSI', line=dict(color='purple')), row=1, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row="1", col="1")
    fig.add_hline(y=30, line_dash="dash", line_color="green", row="1", col="1")
    
    # MACD subplot  
    fig.add_trace(go.Scatter(x=macd_line.index, y=macd_line, mode='lines', name='MACD', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=signal_line.index, y=signal_line, mode='lines', name='Signal', line=dict(color='orange')), row=2, col=1)
    
    # Bollinger Bands with price
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close', line=dict(color='black')), row=3, col=1)
    fig.add_trace(go.Scatter(x=upper_band.index, y=upper_band, mode='lines', name='Upper BB', line=dict(color='red', dash='dot')), row=3, col=1)
    fig.add_trace(go.Scatter(x=lower_band.index, y=lower_band, mode='lines', name='Lower BB', line=dict(color='green', dash='dot')), row=3, col=1)
    
    fig.update_layout(height=height*2, width=width, showlegend=True, title_text="Technical Indicators Overview")
    st.plotly_chart(fig, use_container_width=True)    # Quick stats with status indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        rsi_last = float(rsi.iloc[-1]) if not rsi.isna().all() and pd.notna(rsi.iloc[-1]) else 0.0
        rsi_status = "🔴 Overbought" if rsi_last > 70 else "🟢 Oversold" if rsi_last < 30 else "🟡 Neutral"
        st.metric("RSI", f"{rsi_last:.1f}", help="70+ overbought, 30- oversold")
        st.caption(rsi_status)
    with col2:
        macd_diff = float(macd_line.iloc[-1] - signal_line.iloc[-1]) if not macd_line.isna().all() and pd.notna(macd_line.iloc[-1]) and pd.notna(signal_line.iloc[-1]) else 0.0
        macd_status = "🟢 Bullish" if macd_diff > 0 else "🔴 Bearish"
        st.metric("MACD Signal", f"{macd_diff:.3f}", help="Positive = bullish, Negative = bearish")
        st.caption(macd_status)
    with col3:
        try:
            close_val = float(df['close'].iloc[-1])
            upper_val = float(upper_band.iloc[-1])
            lower_val = float(lower_band.iloc[-1])
            bb_position = (close_val - lower_val) / (upper_val - lower_val) * 100 if not upper_band.isna().all() and upper_val != lower_val else 50.0
        except (ValueError, TypeError, IndexError):
            bb_position = 50.0
        bb_status = "🔴 Near Upper" if bb_position > 80 else "🟢 Near Lower" if bb_position < 20 else "🟡 Middle"
        st.metric("BB Position", f"{bb_position:.1f}%", help="Position within Bollinger Bands")
        st.caption(bb_status)

def plot_candlestick_with_patterns(df: pd.DataFrame, pattern_results: List[Dict[str, Any]], width: int = 1400, height: int = 800):
    """Render Plotly candlestick chart with markers and adjustable size."""
    st.markdown("### 🕯️ Candlestick Chart with Pattern Markers")

    # Use timestamp column if present, else fallback to index
    timestamp_col = None
    for col in df.columns:
        if col.lower() in ("timestamp", "date", "datetime", "time"):
            timestamp_col = col
            break

    x_vals = df[timestamp_col] if timestamp_col else df.index

    fig = go.Figure(data=[go.Candlestick(
        x=x_vals,
        open=df['open'], high=df['high'], low=df['low'], close=df['close']
    )])

    # Group patterns by index to avoid overlap
    pattern_by_idx = defaultdict(list)
    for res in pattern_results:
        pattern_by_idx[res['index']].append(res['pattern'])

    avg_range = (df['high'] - df['low']).mean()
    first = True
    for idx, patterns in pattern_by_idx.items():
        offset = 0.02 * avg_range * (len(patterns) - 1)
        y = df['high'].iloc[idx] + offset
        x = df[timestamp_col].iloc[idx] if timestamp_col else df.index[idx]
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            marker=dict(size=12, color='red'),
            text=[", ".join(patterns)],
            textposition='top center',
            textfont=dict(size=11, color='black'),
            name="Pattern" if first else None,
            showlegend=first,
            hovertemplate="Patterns: %{text}<extra></extra>"
        ))
        first = False

    fig.update_layout(width=width, height=height)
    st.plotly_chart(fig, use_container_width=False)

class TechnicalAnalysisDashboard:
    def __init__(self):
        pass
        
    def run(self):
        """Main dashboard application entry point."""
        initialize_dashboard_session_state()
        
        uploaded_file = session_manager.create_file_uploader("Upload CSV Data", type='csv', file_uploader_name="analysis_data_uploader")
        if not uploaded_file:
            st.info("Please upload a CSV file to begin.")
            return
            
        filename = uploaded_file.name
        stock_ticker = filename.split('_')[0].upper() if '_' in filename else os.path.splitext(filename)[0].upper()
        st.markdown(f"### Stock Ticker: {stock_ticker}")
          # Clear cached DataFrame if filename changes OR if cached df is invalid
        cached_filename = st.session_state.get('filename')
        cached_df = st.session_state.get('df')
        
        logger.info(f"🔍 Session state check:")
        logger.info(f"   - Current filename: {filename}")
        logger.info(f"   - Cached filename: {cached_filename}")
        logger.info(f"   - Cached df rows: {len(cached_df) if cached_df is not None else 'None'}")
        
        needs_reload = (
            cached_filename != filename or 
            cached_df is None or 
            len(cached_df) == 0 or 
            len(cached_df) < 10
        )
        
        logger.info(f"🔄 Needs reload: {needs_reload}")
        
        if needs_reload:
            logger.info(f"🔄 Reloading CSV: filename_changed={cached_filename != filename}, cached_df_rows={len(cached_df) if cached_df is not None else 0}")
            # Clear the cache to force fresh load
            if hasattr(load_df, 'clear'):
                load_df.clear()
                logger.info(f"🧹 Cleared load_df cache")
            df = load_df(uploaded_file)
            st.session_state.df = df
            st.session_state.filename = filename
            logger.info(f"💾 Stored in session: {len(df)} rows")
        else:
            df = cached_df
            logger.info(f"♻️ Using cached DataFrame: {len(df) if df is not None else 0} rows")
            
        if df is None or not validate_dataframe_for_analysis(df):
            return
            
        # Data overview - compact layout
        st.subheader("📊 Data Overview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df.head(5), height=200)
        with col2:
            # Key metrics in compact format
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Total Bars", len(df))
                price_range = f"${df['low'].min():.2f}–${df['high'].max():.2f}"
                st.metric("Price Range", price_range)
                avg_daily_range = ((df['high'] - df['low']) / df['low'] * 100).mean()
                st.metric("Avg Daily Range %", f"{avg_daily_range:.2f}")
            with metrics_col2:
                date_range = f"{df.index[0]} → {df.index[-1]}" if hasattr(df.index, 'name') else f"1 → {len(df)}"
                st.metric("Date Range", date_range)
                daily_returns = df['close'].pct_change() * 100
                annualized_vol = daily_returns.std() * (252 ** 0.5)
                st.metric("Annualized Vol", f"{annualized_vol:.2f}")
                st.metric("Avg Volume", f"{int(df['volume'].mean()):,}")

        # Price and return statistics in expandable sections
        with st.expander("📈 Detailed Statistics"):
            tab1, tab2 = st.tabs(["Price Stats", "Return Stats"])
            with tab1:
                price_stats = cached_compute_price_stats(df)
                st.table(price_stats.round(2))
            with tab2:
                return_stats = cached_compute_return_stats(df)
                st.table(return_stats.round(2))

        # Technical analysis settings and calculations
        st.markdown("---")
        st.markdown("## 🎯 Technical Analysis")
        
        # Streamlined settings in collapsible sections
        with st.expander("⚙️ Indicator Settings", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Momentum**")
                rsi_period = session_manager.create_number_input("RSI Period", min_value=2, max_value=30, value=14, number_input_name="rsi_period")
            with col2:
                st.markdown("**Trend**") 
                macd_fast = session_manager.create_number_input("MACD Fast", min_value=2, max_value=30, value=12, number_input_name="macd_fast")
                macd_slow = session_manager.create_number_input("MACD Slow", min_value=(macd_fast or 12)+1, max_value=60, value=26, number_input_name="macd_slow")
            with col3:
                st.markdown("**Volatility**")
                bb_period = session_manager.create_number_input("BB Period", min_value=2, max_value=30, value=20, number_input_name="bb_period")
                bb_std = session_manager.create_number_input("BB Std Dev", min_value=1, max_value=4, value=2, number_input_name="bb_std")
                  # Additional settings
            col1, col2 = st.columns(2)
            with col1:
                fib_lookback = session_manager.create_number_input("Fib Lookback", min_value=5, max_value=100, value=30, number_input_name="fib_lookback")
            with col2:
                fib_ext = session_manager.create_number_input("Fib Extension", min_value=0.1, max_value=2.0, value=0.618, step=0.001, number_input_name="fib_ext")

        # Type conversions for function calls
        rsi_period_int = int(rsi_period) if rsi_period is not None else 14
        macd_fast_int = int(macd_fast) if macd_fast is not None else 12
        macd_slow_int = int(macd_slow) if macd_slow is not None else 26
        bb_period_int = int(bb_period) if bb_period is not None else 20
        bb_std_int = int(bb_std) if bb_std is not None else 2
        bb_std_float = float(bb_std) if bb_std is not None else 2.0
        fib_lookback_int = int(fib_lookback) if fib_lookback is not None else 30
        fib_ext_float = float(fib_ext) if fib_ext is not None else 0.618

        # Educational content in collapsible section
        with st.expander("📚 Understanding Technical Indicators"):
            st.markdown("""
            **RSI:** Momentum oscillator (0-100). Values >70 suggest overbought, <30 oversold conditions.
            
            **MACD:** Shows relationship between moving averages. Crossovers indicate potential trend changes.
            
            **Bollinger Bands:** Price volatility bands. Price touching upper/lower bands may signal reversal points.
              **ATR:** Measures volatility without indicating direction. Higher values = more volatile price action.
            """)
        # Calculate and display indicators (single calculation)
        with st.spinner("Calculating technical indicators..."):
            rsi, macd_line, signal_line, upper_band, lower_band = get_indicator_series(
                df, rsi_period_int, macd_fast_int, macd_slow_int, bb_period_int, bb_std_int
            )
            plot_technical_indicators(rsi, macd_line, signal_line, df, upper_band, lower_band)
            
        # Combined Technical Signal
        st.markdown("### 🧮 Combined Technical Signal")
          # Use centralized TechnicalAnalysis class for high-level evaluation
        ta = TechnicalAnalysis(df)
        signal_value, rsi_score, macd_score, bb_score = ta.evaluate(
            market_data=df,
            rsi_period=rsi_period_int,
            macd_fast=macd_fast_int,
            macd_slow=macd_slow_int,
            macd_signal=9,
            bb_period=bb_period_int,
            bb_std=bb_std_float,
        )
        col1, col2 = st.columns([2, 1])
        with col1:
            st.metric("Combined Technical Signal", f"{signal_value:.3f}" if signal_value is not None else "N/A", help="Range: -1 (bearish) to 1 (bullish)")
            # Defensive float cast and logging for all technical signal values
            def safe_float(val, name):
                try:
                    fval = float(val)
                    logger.debug(f"{name} value for comparison: {fval} (type={type(fval)})")
                    return fval
                except Exception as e:
                    logger.error(f"Could not convert {name} to float: {val} ({e})")
                    return 0.0

            signal_value_float = safe_float(signal_value, 'signal_value')
            rsi_score_float = safe_float(rsi_score, 'rsi_score')
            macd_score_float = safe_float(macd_score, 'macd_score')
            bb_score_float = safe_float(bb_score, 'bb_score')

            signal_status = (
                "🟢 Bullish" if signal_value_float > 0.1 else
                "🔴 Bearish" if signal_value_float < -0.1 else
                "🟡 Neutral"
            )
            st.caption(signal_status)
        with col2:
            with st.expander("Component Scores"):
                st.metric("RSI", f"{rsi_score:.2f}" if rsi_score is not None else "N/A")
                st.metric("MACD", f"{macd_score:.2f}" if macd_score is not None else "N/A")
                st.metric("BB", f"{bb_score:.2f}" if bb_score is not None else "N/A")

        # ATR & Price Target - compact layout
        st.markdown("### 📏 Volatility & Price Targets")
        col1, col2, col3 = st.columns(3)
        
        # Calculate ATR for both display and summary
        atr_value = None
        try:
            atr_series = calculate_atr(df, length=14)
            atr_value = atr_series.iloc[-1] if len(atr_series) > 0 and not pd.isna(atr_series.iloc[-1]) else None
        except IndicatorError as e:
            logger.warning(f"ATR calculation failed: {e}")
        
        with col1:
            st.metric("ATR", f"{atr_value:.3f}" if atr_value is not None else "N/A", help="Average True Range - measures volatility")
        with col2:
            # Use centralized TechnicalAnalysis class for price target calculation
            ta = TechnicalAnalysis(df)
            pt = ta.calculate_price_target()
            st.metric("Price Target", f"{pt:.3f}" if pt is not None else "N/A", help="Based on trend, volatility, momentum")
        with col3:
            # Use centralized TechnicalAnalysis class for Fibonacci price target
            ta = TechnicalAnalysis(df)
            pt_fib = ta.calculate_price_target_fib(lookback=fib_lookback_int, extension=fib_ext_float)
            st.metric("Fib Price Target", f"{pt_fib:.3f}" if pt_fib is not None else "N/A", help="Fibonacci-based projection")

        # Pattern detection
        st.markdown("---")
        st.markdown("## 🔎 Candlestick Pattern Detection")

        # Create pattern detector instance to get pattern names
        pattern_detector = create_pattern_detector()
        pattern_names = pattern_detector.get_pattern_names()
        selected = tuple(session_manager.create_multiselect("Patterns to scan for", options=pattern_names, default=pattern_names[:6], multiselect_name="pattern_selection"))
        patterns = get_pattern_results(df, selected)

        if patterns:
            st.subheader("Detected Patterns")
            st.dataframe(pd.DataFrame(patterns))
        else:
            st.info("No selected patterns detected in this data.")

        # Candlestick chart
        plot_candlestick_with_patterns(df, patterns, width=1200, height=600)

        # Analysis Summary
        st.markdown("---")
        st.markdown("## 📋 Analysis Summary")

        # Determine time range from the first column (timestamp)
        timestamp_col = df.columns[0]
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            start_date = str(df[timestamp_col].min())
            end_date = str(df[timestamp_col].max())
        except Exception:
            start_date = str(df[timestamp_col].iloc[0])
            end_date = str(df[timestamp_col].iloc[-1])

        summary_lines: List[str] = [
            f"Technical Analysis Summary for {stock_ticker}:",
            f"CSV File: {filename}",
            f"Date Range: {start_date} to {end_date}",
            f"Latest Close: {df['close'].iloc[-1]:.2f}",
            f"RSI (period={rsi_period}): {rsi.iloc[-1]:.2f}" if not rsi.isna().all() else "RSI: N/A",
            f"MACD ({macd_fast}/{macd_slow}): {macd_line.iloc[-1]:.2f}, Signal: {signal_line.iloc[-1]:.2f}",
            f"Bollinger Bands ({bb_period}±{bb_std}): Upper={upper_band.iloc[-1]:.2f}, Lower={lower_band.iloc[-1]:.2f}",
            f"ATR: {atr_value:.3f}" if atr_value is not None else "ATR: N/A",
            f"Price Target: {pt:.3f}" if pt is not None else "Price Target: N/A",
            f"Combined Signal: {signal_value:.3f}" if signal_value is not None else "Combined Signal: N/A",
            "",
            "Detected Patterns (last 20):"
        ]
        if patterns:
            for p in patterns[-20:]:
                summary_lines.append(f"- Index {p['index']}, Date {p['date']}: {p['pattern']}")
        else:
            summary_lines.append("None detected.")
        summary_text = "\n".join(summary_lines)
        
        st.text_area("Copyable Analysis Summary", summary_text, height=300)
        
        if session_manager.create_button("Get ChatGPT Insight", "get_chatgpt_insight"):
            with st.spinner("Contacting ChatGPT..."):
                chatgpt_insight = get_chatgpt_insight(summary_text)
            st.markdown("**ChatGPT Insight:**")
            st.write(chatgpt_insight)

        # Download processed data
        st.markdown("---")
        df_out = df.copy()
        df_out['RSI'] = rsi
        df_out['MACD'] = macd_line
        df_out['MACD_Signal'] = signal_line
        df_out['BB_Upper'] = upper_band
        df_out['BB_Lower'] = lower_band
        csv = df_out.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Enhanced CSV", csv, "processed_data.csv", "text/csv")

# Execute the main function
if __name__ == "__main__":
    try:
        dashboard = TechnicalAnalysisDashboard()
        dashboard.run()
    except Exception as e:
        handle_streamlit_error(e, "Technical Analysis Dashboard")
