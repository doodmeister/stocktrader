import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import ta
from patterns.patterns import CandlestickPatterns
import openai  # Make sure openai is in requirements.txt
from utils.logger import setup_logger
from utils.security import get_openai_api_key  # <-- Use this instead of the local function
from utils.chatgpt import get_chatgpt_insight

logger = setup_logger(__name__)

def flatten_column(series_or_df):
    """Ensures the returned object is always a 1D Series, never a DataFrame or ndarray."""
    if isinstance(series_or_df, pd.DataFrame):
        return series_or_df.iloc[:, 0]
    elif hasattr(series_or_df, 'values') and series_or_df.values.ndim == 2:
        return pd.Series(series_or_df.values.flatten(), index=series_or_df.index)
    return series_or_df

##########################################################################################
## PART 1: Define Functions for Pulling, Processing, and Creating Techincial Indicators ##
##########################################################################################

# Fetch stock data based on the ticker, period, and interval
def fetch_stock_data(ticker, period, interval):
    end_date = datetime.now()
    if period == '1wk':
        start_date = end_date - timedelta(days=7)
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    else:
        data = yf.download(ticker, period=period, interval=interval)
    return data

# Process data to ensure it is timezone-aware and has the correct format
def process_data(data):
    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join([str(c) for c in col]).rstrip('_') for col in data.columns.values]

    # Continue with timezone and naming as before
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize('UTC')
    data.index = data.index.tz_convert('US/Eastern')
    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'Datetime'}, inplace=True)
    data.columns = [str(c).lower() for c in data.columns]  # Ensure all columns are lowercase
    return data

def find_close_col(data):
    # Try common alternatives
    for col in ['close', 'adj close', 'close_sldp', 'adj close_sldp']:
        if col in data.columns:
            return col
    # Try partial match (e.g., for close_xyz)
    for col in data.columns:
        if col.startswith('close'):
            return col
        if col.startswith('adj close'):
            return col
    raise KeyError("No 'close' or 'adj close' column found in data for metrics.")

def find_col(data, base):
    if base in data.columns:
        return base
    for col in data.columns:
        if col.startswith(base):
            return col
    raise KeyError(f"No '{base}' column found in data.")

def normalize_window_columns(window, open_col, high_col, low_col, close_col):
    """Returns a copy of window with columns renamed to 'open', 'high', 'low', 'close'."""
    columns_to_rename = {
        open_col: 'open',
        high_col: 'high',
        low_col: 'low',
        close_col: 'close',
    }
    return window.rename(columns={k: v for k, v in columns_to_rename.items() if k in window.columns})

# Calculate basic metrics from the stock data
def calculate_metrics(data):
    close_col = find_close_col(data)
    close_series = flatten_column(data[close_col])
    last_close = float(close_series.iloc[-1])
    prev_close = float(close_series.iloc[0])
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100
    high_col = find_col(data, 'high')
    low_col = find_col(data, 'low')
    volume_col = find_col(data, 'volume')
    high = float(flatten_column(data[high_col]).max())
    low = float(flatten_column(data[low_col]).min())
    volume = float(flatten_column(data[volume_col]).sum())
    return last_close, change, pct_change, high, low, volume

# Add simple moving average (SMA) and exponential moving average (EMA) indicators
def add_technical_indicators(data):
    close_col = find_close_col(data)
    close_series = flatten_column(data[close_col])
    data['sma_20'] = ta.trend.sma_indicator(close_series, window=20)
    data['ema_20'] = ta.trend.ema_indicator(close_series, window=20)
    return data

###############################################
## PART 2: Creating the Dashboard App layout ##
###############################################

# Set up Streamlit page layout
st.set_page_config(layout="wide")
st.title('Real Time Stock Dashboard')

# 2A: SIDEBAR PARAMETERS ############

# Sidebar for user input parameters
with st.sidebar.form("main_form"):
    st.header('Chart Parameters')
    ticker = st.text_input('Ticker', 'ADBE')
    time_period = st.selectbox('Time Period', ['1d', '1wk', '1mo', '1y', 'max'])
    chart_type = st.selectbox('Chart Type', ['Candlestick', 'Line'])
    indicators = st.multiselect('Technical Indicators', ['SMA 20', 'EMA 20'])
    # Add pattern selection
    pattern_names = CandlestickPatterns.get_pattern_names()
    selected_patterns = st.multiselect("Patterns to scan for", pattern_names, default=pattern_names[:6])
    submitted = st.form_submit_button("Update")

# Mapping of time periods to data intervals
interval_mapping = {
    '1d': '1m',
    '1wk': '30m',
    '1mo': '1d',
    '1y': '1wk',
    'max': '1wk'
}

# Always use the session state version if available
detected_patterns = st.session_state.get('detected_patterns', [])

# 2B: MAIN CONTENT AREA ############

# Update the dashboard based on user input
if submitted:
    st.write("DEBUG: Selected ticker is", ticker)
    data = fetch_stock_data(ticker, time_period, interval_mapping[time_period])
    print("Raw data columns:", data.columns)
    print("First rows:", data.head())
    print("Data shape:", data.shape)
    data = process_data(data)

    if data.empty:
        st.error("No data returned. Data may be unavailable for this ticker/period.")
        st.write("Columns received:", data.columns.tolist())
    else:
        try:
            close_col = find_close_col(data)
            open_col = find_col(data, 'open')
            high_col = find_col(data, 'high')
            low_col = find_col(data, 'low')
        except KeyError as e:
            st.error(f"{e} Data may be unavailable for this ticker/period.")
            st.write("Columns received:", data.columns.tolist())
        else:
            data = add_technical_indicators(data)
            last_close, change, pct_change, high, low, volume = calculate_metrics(data)

            # Display main metrics
            st.metric(label=f"{ticker} Last Price", value=f"{last_close:.2f} USD", delta=f"{change:.2f} ({pct_change:.2f}%)")

            col1, col2, col3 = st.columns(3)
            col1.metric("High", f"{high:.2f} USD")
            col2.metric("Low", f"{low:.2f} USD")
            col3.metric("Volume", f"{volume:,}")

            # Plot the stock price chart
            fig = go.Figure()
            if chart_type == 'Candlestick':
                fig.add_trace(go.Candlestick(
                    x=data['datetime'],
                    open=flatten_column(data[open_col]),
                    high=flatten_column(data[high_col]),
                    low=flatten_column(data[low_col]),
                    close=flatten_column(data[close_col])
                ))
            else:
                close_series = flatten_column(data[close_col])
                x_vals = pd.Series(data['datetime']).reset_index(drop=True)
                y_vals = pd.Series(close_series).reset_index(drop=True)
                fig = px.line(x=x_vals, y=y_vals, labels={'x': 'Datetime', 'y': 'Close'})

            # Add selected technical indicators to the chart
            for indicator in indicators:
                if indicator == 'SMA 20':
                    fig.add_trace(go.Scatter(x=data['datetime'], y=flatten_column(data['sma_20']), name='SMA 20'))
                elif indicator == 'EMA 20':
                    fig.add_trace(go.Scatter(x=data['datetime'], y=flatten_column(data['ema_20']), name='EMA 20'))

            # --- Pattern Detection ---
            detected_patterns = []
            for i in range(len(data)):
                window = data.iloc[max(0, i-4):i+1].copy()
                normalized_window = normalize_window_columns(window, open_col, high_col, low_col, close_col)
                # Only detect patterns if all 4 columns present
                if all(col in normalized_window.columns for col in ['open', 'high', 'low', 'close']):
                    detected = CandlestickPatterns.detect_patterns(normalized_window)
                    for p in detected:
                        if p in selected_patterns:
                            detected_patterns.append({
                                "index": i,
                                "pattern": p,
                                "datetime": data['datetime'].iloc[i]
                            })
                # else skip (row too early in data)

            # --- Display detected patterns ---
            if detected_patterns:
                st.subheader("Detected Patterns")
                st.dataframe(pd.DataFrame(detected_patterns))
            else:
                st.info("No selected patterns detected in this data.")

            # --- Plot with pattern markers ---
            for p in detected_patterns:
                fig.add_trace(go.Scatter(
                    x=[p["datetime"]],
                    y=[data[high_col].iloc[p["index"]]],  # Use robust column access
                    mode="markers+text",
                    marker=dict(symbol="triangle-up", size=12, color="green"),
                    text=[p["pattern"]],
                    textposition="top center",
                    name=p["pattern"]
                ))

            # Format graph
            fig.update_layout(title=f'{ticker} {time_period.upper()} Chart',
                              xaxis_title='Time',
                              yaxis_title='Price (USD)',
                              height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Store detected patterns in session state
            st.session_state['detected_patterns'] = detected_patterns

# 2C: SIDEBAR PRICES ############

# Sidebar section for real-time stock prices of selected symbols
st.sidebar.header('Real-Time Stock Prices')
stock_symbols = ['AAPL', 'GOOGL', 'AMZN', 'MSFT']
for symbol in stock_symbols:
    real_time_data = fetch_stock_data(symbol, '1d', '1m')
    if not real_time_data.empty:
        real_time_data = process_data(real_time_data)
        try:
            close_col = find_close_col(real_time_data)
            open_col = find_col(real_time_data, 'open')
        except KeyError as e:
            st.sidebar.warning(f"{e} for {symbol}")
            continue
        last_price = float(flatten_column(real_time_data[close_col]).iloc[-1])
        open_price = float(flatten_column(real_time_data[open_col]).iloc[0])
        change = last_price - open_price
        pct_change = (change / open_price) * 100
        st.sidebar.metric(f"{symbol}", f"{last_price:.2f} USD", f"{change:.2f} ({pct_change:.2f}%)")

# Sidebar information section
st.sidebar.subheader('About')
st.sidebar.info('This dashboard provides stock data and technical indicators for various time periods. Use the sidebar to customize your view.')

# After your main dashboard logic, outside the `if submitted:` block

# Always use the session state version if available
detected_patterns = st.session_state.get('detected_patterns', [])

st.subheader("AI-Powered Analysis")
if detected_patterns:
    import json
    patterns_detail = json.dumps(detected_patterns, indent=2, default=str)
    patterns_section = f"Detected Patterns Detail:\n{patterns_detail}"
else:
    patterns_section = "Detected Patterns Detail: None"

summary_lines = [
    f"Ticker: {ticker}",
    f"Time Period: {time_period}",
    f"Patterns Detected: {', '.join(set(p['pattern'] for p in detected_patterns)) or 'None'}",
    patterns_section
]
summary_text = "\n".join(summary_lines)
st.text_area("Copyable Analysis Summary", summary_text, height=200)

if st.button("Get ChatGPT Insight"):
    with st.spinner("Contacting ChatGPT..."):
        try:
            openai.api_key = get_openai_api_key()
            result = get_chatgpt_insight(summary_text)
            st.markdown("**AI Analysis Results:**")
            st.write(result)
        except Exception as e:
            st.error(f"AI analysis failed: {e}")
