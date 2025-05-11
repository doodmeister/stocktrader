import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.logger import setup_logger
from typing import List, Dict, Any, Tuple
import os
import openai
from patterns.patterns import CandlestickPatterns
from utils.technicals.technical_analysis import TechnicalAnalysis
from utils.dashboard_utils import initialize_dashboard_session_state
from utils.security import get_openai_api_key

# Configure logging
logger = setup_logger(__name__)

@st.cache_data(show_spinner=False)
def load_df(uploaded_file) -> pd.DataFrame:
    """Load and return CSV as DataFrame."""
    return pd.read_csv(uploaded_file)

@st.cache_data(show_spinner=False)
def get_chatgpt_insight(summary: str) -> str:
    """Send the technical summary to ChatGPT and return its analysis."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-4-turbo"
            messages=[
                {"role": "system", "content": "You are a professional financial analyst."},
                {"role": "user", "content": f"Analyze this technical summary:\n{summary}"}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

@st.cache_data
def compute_price_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for price columns."""
    return pd.DataFrame({
        "Open": [df['open'].min(), df['open'].max(), df['open'].mean(), df['open'].std()],
        "High": [df['high'].min(), df['high'].max(), df['high'].mean(), df['high'].std()],
        "Low":  [df['low'].min(),  df['low'].max(),  df['low'].mean(),  df['low'].std()],
        "Close":[df['close'].min(),df['close'].max(),df['close'].mean(),df['close'].std()]
    }, index=["Min","Max","Mean","Std"])

@st.cache_data
def compute_return_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute distribution statistics for daily returns."""
    daily_returns = df['close'].pct_change() * 100
    return pd.DataFrame({
        "Daily Returns (%)": [
            daily_returns.min(),
            daily_returns.quantile(0.25),
            daily_returns.median(),
            daily_returns.quantile(0.75),
            daily_returns.max(),
            daily_returns.mean(),
            daily_returns.std()
        ]
    }, index=["Min","25%","Median","75%","Max","Mean","Std"])

@st.cache_data
def get_indicator_series(
    df: pd.DataFrame,
    rsi_period: int,
    macd_fast: int,
    macd_slow: int,
    bb_period: int,
    bb_std: int
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculate RSI, MACD lines, and Bollinger Bands."""
    ta = TechnicalAnalysis(df)
    rsi = ta.rsi(period=rsi_period)
    macd_line, signal_line = ta.macd(fast_period=macd_fast, slow_period=macd_slow)
    upper_band, lower_band = ta.bollinger_bands(period=bb_period, std_dev=bb_std)
    return rsi, macd_line, signal_line, upper_band, lower_band

@st.cache_data
def get_pattern_results(df: pd.DataFrame, patterns: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """Detect and return selected candlestick patterns."""
    results: List[Dict[str, Any]] = []
    for i in range(len(df)):
        window = df.iloc[max(0, i-4):i+1]
        detected = CandlestickPatterns.detect_patterns(window)
        for p in detected:
            if p in patterns:
                results.append({
                    "index": i,
                    "pattern": p,
                    "date": df.index[i] if hasattr(df.index, 'name') else i
                })
    return results


def validate_dataframe(df: pd.DataFrame) -> bool:
    """Ensure required columns are present."""
    required = {'open','high','low','close','volume'}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        logger.error(f"Missing columns: {missing}")
        return False
    return True


def plot_technical_indicators(
    rsi: pd.Series,
    macd_line: pd.Series,
    signal_line: pd.Series,
    df: pd.DataFrame,
    upper_band: pd.Series,
    lower_band: pd.Series
):
    """Render indicator charts with captions."""
    st.markdown("""
    ### 📈 Technical Indicator Analysis

    **Technical indicators** are mathematical calculations based on price, volume, or open interest.
    - **RSI (>70 overbought, <30 oversold)**
    - **MACD (line vs. signal)**
    - **Bollinger Bands (volatility bands)**
    """)

    st.line_chart(rsi, height=150, use_container_width=True)
    st.caption(f"RSI (last: {rsi.iloc[-1]:.2f}) — >70 overbought, <30 oversold.")

    macd_df = pd.DataFrame({'MACD': macd_line, 'Signal': signal_line})
    st.line_chart(macd_df, height=150, use_container_width=True)
    st.caption(f"MACD diff (last): {(macd_line.iloc[-1] - signal_line.iloc[-1]):.2f}.")

    bb_df = pd.DataFrame({
        'Close': df['close'],
        'Upper Band': upper_band,
        'Lower Band': lower_band
    })
    st.line_chart(bb_df, height=200, use_container_width=True)
    st.caption("Bollinger Bands visualize volatility.")


def plot_candlestick_with_patterns(df: pd.DataFrame, pattern_results: List[Dict[str, Any]]):
    """Render Plotly candlestick chart with markers."""
    st.markdown("""
    ### 🕯️ Candlestick Chart with Pattern Markers

    Price action as candlesticks with markers for detected patterns.
    """)
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'], high=df['high'], low=df['low'], close=df['close']
    )])
    for res in pattern_results:
        fig.add_trace(go.Scatter(
            x=[res['date']], y=[df['high'].iloc[res['index']]],
            mode='markers+text', marker=dict(size=10, color='red'),
            text=[res['pattern']], textposition='top center'
        ))
    st.plotly_chart(fig, use_container_width=True)


def main():
    openai.api_key = get_openai_api_key()
    initialize_dashboard_session_state()
    st.title("📊 Technical Analysis Dashboard")

    uploaded_file = st.file_uploader("Upload CSV Data", type='csv')
    if not uploaded_file:
        st.info("Please upload a CSV file to begin.")
        return

    filename = uploaded_file.name
    stock_ticker = filename.split('_')[0].upper() if '_' in filename else os.path.splitext(filename)[0].upper()
    st.markdown(f"### Stock Ticker: `{stock_ticker}`")

    if st.session_state.get('filename') != filename:
        df = load_df(uploaded_file)
        st.session_state.df = df
        st.session_state.filename = filename
    else:
        df = st.session_state.df

    if df is None or not validate_dataframe(df):
        return

    st.subheader("Data Preview")
    st.dataframe(df.head(5), height=200)

    price_stats = compute_price_stats(df)
    st.markdown("### 📈 Price Statistics")
    st.table(price_stats.round(2))

    return_stats = compute_return_stats(df)
    st.markdown("### 🔄 Returns Analysis")
    st.table(return_stats.round(2))

    # Key metrics row 1
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Bars", len(df))
    date_range = f"{df.index[0]} → {df.index[-1]}" if hasattr(df.index, 'name') else f"1 → {len(df)}"
    col2.metric("Date Range", date_range)
    price_range = f"${df['low'].min():.2f}–${df['high'].max():.2f}"
    col3.metric("Price Range", price_range)

    # Key metrics row 2
    col1, col2, col3 = st.columns(3)
    avg_daily_range = ((df['high'] - df['low']) / df['low'] * 100).mean().round(2)
    col1.metric("Avg Daily Range %", f"{avg_daily_range}")
    avg_volume = int(df['volume'].mean())
    col2.metric("Avg Volume", f"{avg_volume:,}")
    max_volume = int(df['volume'].max())
    col3.metric("Max Volume", f"{max_volume:,}")

    # Key metrics row 3
    col1, _, _ = st.columns(3)
    daily_returns = df['close'].pct_change() * 100
    annualized_vol = (daily_returns.std() * (252 ** 0.5)).round(2)
    col1.metric("Annualized Volatility", f"{annualized_vol}")

    # Indicator settings
    st.markdown("---")
    st.markdown("## ⚙️ Indicator Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        rsi_period = st.number_input("RSI Period", 2, 30, 14)
    with col2:
        macd_fast = st.number_input("MACD Fast", 2, 30, 12)
        macd_slow = st.number_input("MACD Slow", macd_fast+1, 60, 26)
    with col3:
        bb_period = st.number_input("BB Period", 2, 30, 20)
        bb_std = st.number_input("BB Std Dev", 1, 4, 2)

    st.markdown("""
    **Indicator Settings Explained:**

    - **RSI Period**:  
      Sets the lookback window for the Relative Strength Index (RSI), a momentum oscillator that measures the speed and change of price movements.  
      - *Lower values* make RSI more sensitive (more signals, more noise).  
      - *Higher values* smooth out RSI (fewer, but stronger signals).

    - **MACD Fast / Slow**:  
      Control the short-term (fast) and long-term (slow) moving averages for the MACD (Moving Average Convergence Divergence) indicator.  
      - *Lower fast period* reacts more quickly to price changes.  
      - *Higher slow period* smooths out the MACD line.  
      - Adjusting these can help you spot trend changes earlier or filter out noise.

    - **BB Period / BB Std Dev**:  
      Set the window and width for Bollinger Bands, which measure price volatility.  
      - *BB Period* is the number of bars used for the moving average.  
      - *BB Std Dev* controls how wide the bands are (higher = wider bands, capturing more volatility).  
      - Tighter bands (lower std dev) can signal breakouts; wider bands (higher std dev) can help avoid false signals.

    *Tip: Adjust these settings to match your trading style or to experiment with different market conditions!*
    """)

    # Plot indicators
    with st.container():
        rsi, macd_line, signal_line, upper_band, lower_band = get_indicator_series(
            df, rsi_period, macd_fast, macd_slow, bb_period, bb_std
        )
        plot_technical_indicators(rsi, macd_line, signal_line, df, upper_band, lower_band)

    # Combined Technical Signal
    st.markdown("""
    ---
    ## 🧮 Combined Technical Signal

    This metric combines RSI, MACD, and Bollinger Bands into a single value between -1 (bearish) and 1 (bullish).

    - **Interpretation**:
      - Values near 1 suggest bullish conditions.
      - Values near -1 suggest bearish conditions.
    - **Note**:
      - This is a composite signal for research, not a trading recommendation.
    """)
    signal_value = TechnicalAnalysis(df).evaluate(df)
    st.metric("Combined Technical Signal", f"{signal_value:.3f}" if signal_value is not None else "N/A", help="Range: -1 (bearish) to 1 (bullish)")

    # ATR & Price Target
    st.markdown("""
    ---
    ## 📏 Volatility & Price Target

    - **ATR (Average True Range)**: Measures market volatility. Higher ATR means more price movement.
    - **Price Target**: A calculated projection based on trend, volatility, and momentum.
    """)
    atr = TechnicalAnalysis(df).calculate_atr(symbol=None)
    pt = TechnicalAnalysis(df).calculate_price_target(symbol=None)
    if atr is not None:
        st.write(f"**ATR (period=3):** {atr:.3f}")
    else:
        st.write("ATR unavailable.")
    if pt is not None:
        st.write(f"**Price Target:** {pt:.3f}")
    else:
        st.write("Price target unavailable.")

   # Pattern detection
    st.markdown("""
    ---
    ## 🔎 Candlestick Pattern Detection

    Select patterns to scan for. Detected patterns may signal trend reversals or continuations.
    """)

    pattern_names = CandlestickPatterns.get_pattern_names()
    selected = tuple(st.multiselect("Patterns to scan for", pattern_names, default=pattern_names[:6])) # default 6 patterns this can be upped to the max number in the patterns.py
    patterns = get_pattern_results(df, selected)

    if patterns:
        st.subheader("Detected Patterns")
        st.dataframe(pd.DataFrame(patterns))
        st.markdown("""
        **How to read:**  
        - Each row shows the index/date and the detected pattern.
        - Use this to spot potential trade setups or validate your strategy.
        """)
    else:
        st.info("No selected patterns detected in this data.")

    # Candlestick chart
    plot_candlestick_with_patterns(df, patterns)

    # Detailed Analysis Summary
    st.markdown("""
    ---
    ## 📋 Detailed Analysis Summary

    Copy the summary below and paste it into ChatGPT or another LLM for further insights.
    """)

    # Determine time range from the first column (timestamp)
    timestamp_col = df.columns[0]
    try:
        # Ensure the column is in datetime format
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        start_date = str(df[timestamp_col].min())
        end_date = str(df[timestamp_col].max())
    except Exception as e:
        # Fallback if conversion fails
        start_date = str(df[timestamp_col].iloc[0])
        end_date = str(df[timestamp_col].iloc[-1])

    summary_lines: List[str] = [
        f"Technical Analysis Summary for {stock_ticker}:",
        f"CSV File: {filename}",
        f"Date Range: {start_date} to {end_date}",  # <-- Now uses timestamp column
        f"Latest Close: {df['close'].iloc[-1]:.2f}",
        f"RSI (period={rsi_period}): {rsi.iloc[-1]:.2f}" if not rsi.isna().all() else "RSI: N/A",
        f"MACD (fast={macd_fast}, slow={macd_slow}): {macd_line.iloc[-1]:.2f}",
        f"MACD Signal: {signal_line.iloc[-1]:.2f}",
        f"Bollinger Bands (period={bb_period}, std={bb_std}): Upper={upper_band.iloc[-1]:.2f}, Lower={lower_band.iloc[-1]:.2f}",
        f"ATR (period=3): {atr:.3f}" if atr is not None else "ATR: N/A",
        f"Price Target: {pt:.3f}" if pt is not None else "Price Target: N/A",
        f"Combined Technical Signal: {signal_value:.3f}" if signal_value is not None else "Combined Technical Signal: N/A",
        "",
        "Detected Patterns (last 10):"
    ]
    if patterns:
        for p in patterns[-10:]:
            summary_lines.append(f"- Index {p['index']}, Date {p['date']}: {p['pattern']}")
    else:
        summary_lines.append("None detected.")
    summary_text = "\n".join(summary_lines)
    st.text_area("Copyable Analysis Summary", summary_text, height=300)

    if st.button("Get ChatGPT Insight"):
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
    st.download_button("Download CSV", csv, "processed_data.csv", "text/csv")

if __name__ == "__main__":
    main()