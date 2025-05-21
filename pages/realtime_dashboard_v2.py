## NOTE: Set yfinance to the following version to get chart working: "pip install yfinance==0.2.40"

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import openai
import datetime
import base64
import os

from utils.security import get_openai_api_key  # Make sure this exists or replace with your API key logic

# Set up Streamlit app
st.set_page_config(layout="wide")
st.title("AI-Powered Technical Stock Analysis Dashboard")
st.sidebar.header("Configuration")

# --- Date logic ---
today = datetime.date.today()
default_start = today - datetime.timedelta(days=30)

ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
start_date = st.sidebar.date_input("Start Date", value=default_start, max_value=today)
end_date = st.sidebar.date_input("End Date", value=today, min_value=default_start, max_value=today)

# Fetch stock data
def fetch_data():
    try:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        if start > end:
            st.warning("Start date must be before or equal to end date.")
            return

        st.info(f"Fetching data for {ticker} from {start.date()} to {end.date()}...")
        data = yf.download(ticker, start=start, end=end)

        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            # For single ticker, drop ticker level
            if len(set(data.columns.get_level_values(1))) == 1:
                data.columns = data.columns.droplevel(1)
            else:
                data.columns = ['_'.join(col).strip().lower() for col in data.columns.values]
        
        # Now, convert all columns to lowercase!
        data.columns = [col.lower() for col in data.columns]

        expected_cols = ["open", "high", "low", "close", "volume"]
        data = data[[col for col in expected_cols if col in data.columns]]

        if not data.empty:
            st.session_state["stock_data"] = data
            st.success("Stock data loaded successfully!")
        else:
            st.warning("No data found for the selected ticker and date range after processing.")
    except Exception as e:
        st.error(f"Error fetching data: {e}")

if st.sidebar.button("Fetch Data"):
    fetch_data()

# Check if data is available and not empty
if "stock_data" in st.session_state:
    data = st.session_state["stock_data"]
    if data.empty:
        st.warning("No data to display. Please check your ticker and date range.")
    else:
        # --- Normalize columns to lowercase for consistency ---
        df = data.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            }
        )
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        # Sidebar: Select technical indicators
        st.sidebar.subheader("Technical Indicators")
        indicators = st.sidebar.multiselect(
            "Select Indicators:",
            ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"],
            default=["20-Day SMA"]
        )

        # --- Compute indicators for plotting ---
        traces = [
            go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Candlestick"
            )
        ]

        if "20-Day SMA" in indicators:
            sma = df["close"].rolling(window=20).mean()
            traces.append(go.Scatter(x=df.index, y=sma, mode="lines", name="SMA (20)", line=dict(color="blue")))
        if "20-Day EMA" in indicators:
            ema = df["close"].ewm(span=20).mean()
            traces.append(go.Scatter(x=df.index, y=ema, mode="lines", name="EMA (20)", line=dict(color="orange")))
        if "20-Day Bollinger Bands" in indicators:
            sma = df["close"].rolling(window=20).mean()
            std = df["close"].rolling(window=20).std()
            bb_upper = sma + 2 * std
            bb_lower = sma - 2 * std
            traces.append(go.Scatter(x=df.index, y=bb_upper, mode="lines", name="BB Upper", line=dict(color="green", width=1)))
            traces.append(go.Scatter(x=df.index, y=bb_lower, mode="lines", name="BB Lower", line=dict(color="red", width=1)))
        if "VWAP" in indicators and "volume" in df.columns:
            vwap = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
            traces.append(go.Scatter(x=df.index, y=vwap, mode="lines", name="VWAP", line=dict(color="purple", dash="dot")))

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=f"{ticker} Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            height=600,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show the dataframe below the chart for reference
        with st.expander("Show raw data"):
            if not df.empty and all(col in df.columns for col in ["open", "high", "low", "close", "volume"]):
                st.dataframe(df.reset_index())
            else:
                st.warning("No valid stock data to display.")

        # Analyze chart with OpenAI (no image, just data)
        st.subheader("AI-Powered Analysis")
        st.info(
            "Click below to send the displayed data and selected indicators to the AI for analysis. "
            "No screenshot needed!"
        )

        # Prepare a summary of the data and indicators
        preview_rows = 30  # Limit rows for prompt size
        data_to_send = df.tail(preview_rows).copy()

        # Add selected indicators as columns
        if "20-Day SMA" in indicators:
            data_to_send["SMA_20"] = data_to_send['close'].rolling(window=20).mean()
        if "20-Day EMA" in indicators:
            data_to_send["EMA_20"] = data_to_send['close'].ewm(span=20).mean()
        if "20-Day Bollinger Bands" in indicators:
            sma = data_to_send['close'].rolling(window=20).mean()
            std = data_to_send['close'].rolling(window=20).std()
            data_to_send["BB_Upper"] = sma + 2 * std
            data_to_send["BB_Lower"] = sma - 2 * std
        if "VWAP" in indicators:
            data_to_send["VWAP"] = (data_to_send['close'] * data_to_send['volume']).cumsum() / data_to_send['volume'].cumsum()

        # Format as CSV for the prompt
        csv_data = data_to_send.reset_index().to_csv(index=False)

        prompt = (
            f"You are a Stock Trader specializing in Technical Analysis at a top financial institution.\n"
            f"Analyze the following stock data and technical indicators and provide a buy/hold/sell recommendation.\n"
            f"Base your recommendation only on the data and the displayed technical indicators.\n"
            f"First, provide the recommendation, then, provide your detailed reasoning.\n\n"
            f"Ticker: {ticker}\n"
            f"Date Range: {start_date} to {end_date}\n"
            f"Selected Indicators: {', '.join(indicators)}\n\n"
            f"Here is the data (last {preview_rows} rows):\n"
            f"{csv_data}"
        )

        if st.button("Run AI Analysis"):
            with st.spinner("Analyzing the data, please wait..."):
                try:
                    openai.api_key = get_openai_api_key()
                    response = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=800,
                    )
                    st.write("**AI Analysis Results:**")
                    st.write(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"AI analysis failed: {e}")