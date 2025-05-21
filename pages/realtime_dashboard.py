# Source: @DeepCharts Youtube Channel (https://www.youtube.com/@DeepCharts)

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import ta

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
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize('UTC')
    data.index = data.index.tz_convert('US/Eastern')
    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'Datetime'}, inplace=True)
    return data

# Calculate basic metrics from the stock data
def calculate_metrics(data):
    # Ensure 'Close' is a Series, not a DataFrame or 2D array
    close_series = flatten_column(data['Close'])
    last_close = float(close_series.iloc[-1])
    prev_close = float(close_series.iloc[0])
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100
    high = float(flatten_column(data['High']).max())
    low = float(flatten_column(data['Low']).min())
    volume = float(flatten_column(data['Volume']).sum())
    return last_close, change, pct_change, high, low, volume

# Add simple moving average (SMA) and exponential moving average (EMA) indicators
def add_technical_indicators(data):
    close_series = flatten_column(data['Close'])
    data['SMA_20'] = ta.trend.sma_indicator(close_series, window=20)
    data['EMA_20'] = ta.trend.ema_indicator(close_series, window=20)
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
    submitted = st.form_submit_button("Update")

# Mapping of time periods to data intervals
interval_mapping = {
    '1d': '1m',
    '1wk': '30m',
    '1mo': '1d',
    '1y': '1wk',
    'max': '1wk'
}


# 2B: MAIN CONTENT AREA ############

# Update the dashboard based on user input
if submitted:
    st.write("DEBUG: Selected ticker is", ticker)
    data = fetch_stock_data(ticker, time_period, interval_mapping[time_period])
    data = process_data(data)
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
            x=data['Datetime'],
            open=flatten_column(data['Open']),
            high=flatten_column(data['High']),
            low=flatten_column(data['Low']),
            close=flatten_column(data['Close'])
        ))
    else:
        # For maximum safety, bypass DataFrame entirely
        close_series = flatten_column(data['Close'])
        x_vals = pd.Series(data['Datetime']).reset_index(drop=True)
        y_vals = pd.Series(close_series).reset_index(drop=True)
        st.write("x shape:", x_vals.shape)
        st.write("y shape:", y_vals.shape)
        st.write("x example:", x_vals.head())
        st.write("y example:", y_vals.head())
        fig = px.line(x=x_vals, y=y_vals, labels={'x': 'Datetime', 'y': 'Close'})

    # Add selected technical indicators to the chart
    for indicator in indicators:
        if indicator == 'SMA 20':
            fig.add_trace(go.Scatter(x=data['Datetime'], y=flatten_column(data['SMA_20']), name='SMA 20'))
        elif indicator == 'EMA 20':
            fig.add_trace(go.Scatter(x=data['Datetime'], y=flatten_column(data['EMA_20']), name='EMA 20'))

    # Format graph
    fig.update_layout(title=f'{ticker} {time_period.upper()} Chart',
                      xaxis_title='Time',
                      yaxis_title='Price (USD)',
                      height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Display historical data and technical indicators
    st.subheader('Historical Data')
    st.dataframe(data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']])

    st.subheader('Technical Indicators')
    st.dataframe(data[['Datetime', 'SMA_20', 'EMA_20']])


# 2C: SIDEBAR PRICES ############

# Sidebar section for real-time stock prices of selected symbols
st.sidebar.header('Real-Time Stock Prices')
stock_symbols = ['AAPL', 'GOOGL', 'AMZN', 'MSFT']
for symbol in stock_symbols:
    real_time_data = fetch_stock_data(symbol, '1d', '1m')
    if not real_time_data.empty:
        real_time_data = process_data(real_time_data)
        last_price = float(flatten_column(real_time_data['Close']).iloc[-1])
        open_price = float(flatten_column(real_time_data['Open']).iloc[0])
        change = last_price - open_price
        pct_change = (change / open_price) * 100
        st.sidebar.metric(f"{symbol}", f"{last_price:.2f} USD", f"{change:.2f} ({pct_change:.2f}%)")

# Sidebar information section
st.sidebar.subheader('About')
st.sidebar.info('This dashboard provides stock data and technical indicators for various time periods. Use the sidebar to customize your view.')