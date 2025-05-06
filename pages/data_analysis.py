import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
from typing import List, Dict, Any, Optional

from patterns import CandlestickPatterns
from utils.indicators import add_technical_indicators
from utils.performance_utils import PatternDetector
from utils.technical_analysis import TechnicalAnalysis

# Configure logging
logger = logging.getLogger("data_analysis")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

def load_csv(uploaded_file) -> Optional[pd.DataFrame]:
    """Safely load a CSV file into a DataFrame."""
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("Uploaded CSV is empty.")
            logger.error("Uploaded CSV is empty.")
            return None
        return df
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        logger.exception("Failed to load CSV")
        return None

def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate that required columns exist."""
    required_cols = {'open', 'high', 'low', 'close', 'volume'}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        logger.error(f"Missing required columns: {missing}")
        return False
    return True

def plot_technical_indicators(df: pd.DataFrame, ta: TechnicalAnalysis, rsi_period: int, macd_fast: int, macd_slow: int, bb_period: int, bb_std: int):
    """Plot RSI, MACD, and Bollinger Bands."""
    st.subheader("Technical Indicator Analysis")

    # RSI
    rsi = ta.rsi(period=rsi_period)
    st.line_chart(rsi, height=150, use_container_width=True)
    st.caption(f"RSI (last value: {rsi.iloc[-1]:.2f})" if not rsi.isna().all() else "RSI not available.")

    # MACD
    macd_line, signal_line = ta.macd(fast_period=macd_fast, slow_period=macd_slow)
    st.line_chart(pd.DataFrame({'MACD': macd_line, 'Signal': signal_line}), height=150, use_container_width=True)
    st.caption(f"MACD (last diff: {(macd_line.iloc[-1] - signal_line.iloc[-1]):.2f})")

    # Bollinger Bands
    upper_band, lower_band = ta.bollinger_bands(period=bb_period, std_dev=bb_std)
    bb_df = pd.DataFrame({'Close': df['close'], 'Upper Band': upper_band, 'Lower Band': lower_band})
    st.line_chart(bb_df, height=200, use_container_width=True)

    return rsi, macd_line, signal_line, upper_band, lower_band

def detect_patterns(df: pd.DataFrame, selected_patterns: List[str]) -> List[Dict[str, Any]]:
    """Detect selected candlestick patterns in the DataFrame."""
    pattern_results = []
    for i in range(len(df)):
        window = df.iloc[max(0, i-4):i+1]
        detected = CandlestickPatterns.detect_patterns(window)
        for pattern in detected:
            if pattern in selected_patterns:
                pattern_results.append({
                    "index": i,
                    "pattern": pattern,
                    "date": df.index[i] if df.index.name else i
                })
    return pattern_results

def plot_candlestick_with_patterns(df: pd.DataFrame, pattern_results: List[Dict[str, Any]]):
    """Plot a candlestick chart with pattern markers."""
    fig = go.Figure(data=[go.Candlestick(
        x=df.index if df.index.name else range(len(df)),
        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name="Price"
    )])
    for res in pattern_results:
        fig.add_trace(go.Scatter(
            x=[res["date"]], y=[df['high'].iloc[res["index"]]],
            mode="markers+text", marker=dict(size=12, color="red"),
            text=[res["pattern"]], textposition="top center"
        ))
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("📊 Technical Analysis Dashboard")

    uploaded_file = st.file_uploader("Upload CSV Data for Analysis", type="csv")
    if not uploaded_file:
        st.info("Please upload a CSV file to begin.")
        return

    df = load_csv(uploaded_file)
    if df is None or not validate_dataframe(df):
        return

    st.subheader("Data Preview")
    st.dataframe(df.head())

    st.subheader("Basic Statistics")
    st.write(df.describe())

    # User controls for indicator periods
    col1, col2, col3 = st.columns(3)
    with col1:
        rsi_period = st.number_input("RSI Period", min_value=2, max_value=30, value=14)
    with col2:
        macd_fast = st.number_input("MACD Fast Period", min_value=2, max_value=30, value=12)
        macd_slow = st.number_input("MACD Slow Period", min_value=macd_fast+1, max_value=60, value=26)
    with col3:
        bb_period = st.number_input("Bollinger Bands Period", min_value=2, max_value=30, value=20)
        bb_std = st.number_input("BB Std Dev", min_value=1, max_value=4, value=2)

    ta = TechnicalAnalysis(df)

    # Plot indicators and get series for download
    rsi, macd_line, signal_line, upper_band, lower_band = plot_technical_indicators(
        df, ta, rsi_period, macd_fast, macd_slow, bb_period, bb_std
    )

    # Combined Signal
    signal = ta.evaluate(df)
    st.metric("Combined Technical Signal", f"{signal:.3f}" if signal is not None else "N/A", help="Range: -1 (bearish) to 1 (bullish)")

    # ATR and Price Target
    atr = ta.calculate_atr(symbol=None)
    price_target = ta.calculate_price_target(symbol=None)
    st.write(f"**ATR (period=3):** {atr:.3f}" if atr is not None else "ATR not available.")
    st.write(f"**Price Target:** {price_target:.3f}" if price_target is not None else "Price target not available.")

    # Pattern detection
    pattern_names = CandlestickPatterns.get_pattern_names()
    selected_patterns = st.multiselect("Patterns to scan for", pattern_names, default=pattern_names[:3])
    pattern_results = detect_patterns(df, selected_patterns)

    if pattern_results:
        st.subheader("Detected Patterns")
        st.dataframe(pd.DataFrame(pattern_results))
    else:
        st.info("No selected patterns detected in this data.")

    # Candlestick chart with pattern markers
    plot_candlestick_with_patterns(df, pattern_results)

    # Optionally, allow download of processed data
    st.subheader("Download Processed Data")
    df['RSI'] = rsi
    df['MACD'] = macd_line
    df['MACD_Signal'] = signal_line
    df['BB_Upper'] = upper_band
    df['BB_Lower'] = lower_band
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "processed_data.csv", "text/csv")

if __name__ == "__main__":
    main()