import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Any, Tuple
import os
import openai
from collections import defaultdict

from utils.logger import get_dashboard_logger
logger = get_dashboard_logger(__name__)

from patterns.patterns import CandlestickPatterns, create_pattern_detector
from utils.technicals.technical_analysis import TechnicalAnalysis
from utils.technicals.indicators import add_bollinger_bands, compute_price_stats, compute_return_stats
from core.dashboard_utils import (
    setup_page, 
    handle_streamlit_error, 
    safe_streamlit_metric, 
    create_candlestick_chart, 
    validate_ohlc_dataframe, 
    initialize_dashboard_session_state
)
from utils.security import get_openai_api_key
from utils.chatgpt import get_chatgpt_insight as _get_chatgpt_insight

# Initialize the page (setup_page returns a logger, but we already have one)
setup_page(
    title="📊 Technical Analysis Dashboard",
    logger_name=__name__,
    sidebar_title="Analysis Controls"
)

@st.cache_data(show_spinner=False)
def load_df(uploaded_file) -> pd.DataFrame:
    """Load and return CSV as DataFrame."""
    return pd.read_csv(uploaded_file)

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
    """Calculate RSI, MACD lines, and Bollinger Bands using indicators.py."""
    ta = TechnicalAnalysis(df)
    rsi = ta.rsi(period=rsi_period)
    macd_line, signal_line = ta.macd(fast_period=macd_fast, slow_period=macd_slow)
    # Use add_bollinger_bands directly
    bb_df = add_bollinger_bands(df, length=bb_period, std=bb_std)
    upper_band = bb_df['bb_upper']
    lower_band = bb_df['bb_lower']
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
    lower_band: pd.Series,
    width: int = 1200,
    height: int = 600
):
    st.markdown("""
    ### 📈 Technical Indicator Analysis

    **Technical indicators** are mathematical calculations based on price, volume, or open interest.
    - **RSI (>70 overbought, <30 oversold)**
    - **MACD (line vs. signal)**
    - **Bollinger Bands (volatility bands)**
    """)

    # RSI Plot
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=rsi.index, y=rsi, mode='lines', name='RSI'))
    fig_rsi.update_layout(title="RSI", width=width, height=height)
    st.plotly_chart(fig_rsi, use_container_width=False)
    st.caption(f"RSI (last: {rsi.iloc[-1]:.2f}) — >70 overbought, <30 oversold.")

    # MACD Plot
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=macd_line.index, y=macd_line, mode='lines', name='MACD'))
    fig_macd.add_trace(go.Scatter(x=signal_line.index, y=signal_line, mode='lines', name='Signal'))
    fig_macd.update_layout(title="MACD", width=width, height=height)
    st.plotly_chart(fig_macd, use_container_width=False)
    st.caption(f"MACD diff (last): {(macd_line.iloc[-1] - signal_line.iloc[-1]):.2f}.")

    # Bollinger Bands Plot
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close'))
    fig_bb.add_trace(go.Scatter(x=upper_band.index, y=upper_band, mode='lines', name='Upper Band'))
    fig_bb.add_trace(go.Scatter(x=lower_band.index, y=lower_band, mode='lines', name='Lower Band'))
    fig_bb.update_layout(title="Bollinger Bands", width=width, height=height)
    st.plotly_chart(fig_bb, use_container_width=False)
    st.caption("Bollinger Bands visualize volatility.")


def plot_candlestick_with_patterns(df: pd.DataFrame, pattern_results: List[Dict[str, Any]], width: int = 1400, height: int = 800):
    """Render Plotly candlestick chart with markers and adjustable size."""
    st.markdown("""
    ### 🕯️ Candlestick Chart with Pattern Markers

    Price action as candlesticks with markers for detected patterns.
    """)

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
        openai.api_key = get_openai_api_key()
        initialize_dashboard_session_state()


        uploaded_file = st.file_uploader("Upload CSV Data", type='csv')
        if not uploaded_file:
            st.info("Please upload a CSV file to begin.")
            return

        filename = uploaded_file.name
        stock_ticker = filename.split('_')[0].upper() if '_' in filename else os.path.splitext(filename)[0].upper()
        st.markdown(f"### Stock Ticker: {stock_ticker}")

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

        price_stats = cached_compute_price_stats(df)
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
            fib_lookback = st.number_input("Fib Lookback", 5, 100, 30)
            fib_ext      = st.number_input("Fib Extension", 0.1, 2.0, 0.618, step=0.001)
        with col3:
            bb_period = st.number_input("BB Period", 2, 30, 20)
            bb_std = st.number_input("BB Std Dev", 1, 4, 2)

        st.markdown("""
        ### 📚 Understanding Technical Indicators

        Technical indicators are mathematical tools that help traders interpret stock price charts and make sense of market trends, momentum, and volatility. Here’s what the key indicators in this dashboard mean:

        **Relative Strength Index (RSI):**  
        The RSI is a momentum oscillator that measures the speed and magnitude of recent price changes. It ranges from 0 to 100. Values above 70 typically indicate that a stock is “overbought” (it may be due for a pullback), while values below 30 suggest it’s “oversold” (it may be due for a bounce). RSI helps traders spot potential reversals or confirm trends.

        **MACD (Moving Average Convergence Divergence):**  
        MACD shows the relationship between two moving averages of price (usually 12-period and 26-period). When the MACD line crosses above its “signal line,” it’s often seen as a bullish (buy) sign; when it crosses below, it can be bearish (sell). MACD is used to spot changes in trend direction and momentum.

        **Bollinger Bands:**  
        Bollinger Bands consist of three lines plotted over the price: a simple moving average (middle band) and two bands set a certain number of standard deviations above and below (upper and lower bands). When price approaches the upper band, the stock may be considered “expensive” or overbought; when it nears the lower band, it may be “cheap” or oversold. Bollinger Bands expand during volatile periods and contract when the market is calm.

        **ATR (Average True Range):**  
        ATR measures how much a stock typically moves during a given period. Higher ATR means more volatility; lower ATR means less. ATR doesn’t indicate direction, just the degree of price movement, and is often used to set stop-losses or gauge trading risk.

        **Combined Technical Signal:**  
        This dashboard also calculates a “composite signal,” which blends the information from RSI, MACD, and Bollinger Bands into a single score from -1 (bearish) to +1 (bullish). This helps provide a quick sense of overall market conditions based on several indicators working together.

        *Note: No indicator can predict the future with certainty. They are best used as guides, not guarantees, and should always be combined with sound risk management.*
        """)

        # Plot indicators
        with st.container():
            rsi, macd_line, signal_line, upper_band, lower_band = get_indicator_series(
                df, rsi_period, macd_fast, macd_slow, bb_period, bb_std
            )
            plot_technical_indicators(rsi, macd_line, signal_line, df, upper_band, lower_band, width=1200, height=600)

        # Combined Technical Signal
        st.info("""
        ## 🧮 Combined Technical Signal

        This metric combines RSI, MACD, and Bollinger Bands into a single value between -1 (bearish) and 1 (bullish).

        - **Interpretation**:
          - Values near **1** suggest strong bullish conditions.
          - Values near **-1** suggest strong bearish conditions.
          - Values near **0** indicate neutral or trendless conditions.
          - Slightly negative (**–0.1 to 0**) → mildly bearish or indecisive.
          - Slightly positive (**0 to +0.1**) → mildly bullish or indecisive.
        - **Note**:
          - This is a composite signal for research, not a trading recommendation.
        """)
        signal_value, rsi_score, macd_score, bb_score = TechnicalAnalysis(df).evaluate(
            market_data=df,
            rsi_period=rsi_period,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=9,  # or expose as UI
            bb_period=bb_period,
            bb_std=bb_std,
        )
        st.metric("Combined Technical Signal", f"{signal_value:.3f}" if signal_value is not None else "N/A", help="Range: -1 (bearish) to 1 (bullish)")

        with st.expander("▶ Component Scores"):
            st.metric("RSI Score",   f"{rsi_score:.2f}" if rsi_score is not None else "N/A")
            st.metric("MACD Score",  f"{macd_score:.2f}" if macd_score is not None else "N/A")
            st.metric("BB Score",    f"{bb_score:.2f}" if bb_score is not None else "N/A")

        # ATR & Price Target
        st.markdown("""
        ---
        ## 📏 Volatility & Price Target

        - **ATR (Average True Range)**: Measures market volatility. Higher ATR means more price movement.
        - **Price Target**: A calculated projection based on trend, volatility, and momentum.
        """)
        atr = TechnicalAnalysis(df).calculate_atr()
        pt = TechnicalAnalysis(df).calculate_price_target()
        if atr is not None:
            st.write(f"**ATR (period=3):** {atr:.3f}")
        else:
            st.write("ATR unavailable.")
        if pt is not None:
            st.write(f"**Price Target:** {pt:.3f}")
        else:
            st.write("Price target unavailable.")

        ta = TechnicalAnalysis(df)
        pt = ta.calculate_price_target_fib(
            lookback=fib_lookback,
            extension=fib_ext
        )
        st.write(f"**Fib Price Target:** {pt:.3f}")

       # Pattern detection
        st.markdown("""
        ---
        ## 🔎 Candlestick Pattern Detection

        Select patterns to scan for. Detected patterns may signal trend reversals or continuations.
        """)

        # Create pattern detector instance to get pattern names
        pattern_detector = create_pattern_detector()
        pattern_names = pattern_detector.get_pattern_names()
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
        plot_candlestick_with_patterns(df, patterns, width=1200, height=600)

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
            "Detected Patterns (last 20):"
        ]
        if patterns:
            for p in patterns[-20:]:
                summary_lines.append(f"- Index {p['index']}, Date {p['date']}: {p['pattern']}")
        else:
            summary_lines.append("None detected.")
        summary_text = "\n".join(summary_lines)
        st.text_area("Copyable Analysis Summary", summary_text, height=400)

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

# Execute the main function
if __name__ == "__main__":
    try:
        dashboard = TechnicalAnalysisDashboard()
        dashboard.run()
    except Exception as e:
        handle_streamlit_error(e, "Technical Analysis Dashboard")

