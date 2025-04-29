import streamlit as st
import pandas as pd
import torch
import threading
import plotly.graph_objs as go
import asyncio
from performance_utils import get_candles_cached, fetch_all_candles
from etrade_candlestick_bot import (
    ETradeClient,
    CandlestickPatterns,
    PatternNN,
    train_pattern_model
)

# ─────── UI CONFIG ───────────────────────────────────────────────────────────
st.set_page_config(page_title="E*Trade Candlestick Bot Dashboard", layout="wide")

# ─────── Sidebar: Credentials & Model Settings ─────────────────────────────────
st.sidebar.title("⚙️ Configuration")
consumer_key = st.sidebar.text_input("Consumer Key", type="password")
consumer_secret = st.sidebar.text_input("Consumer Secret", type="password")
oauth_token = st.sidebar.text_input("OAuth Token", type="password")
oauth_token_secret = st.sidebar.text_input("OAuth Token Secret", type="password")
account_id = st.sidebar.text_input("Account ID")

# Training hyperparameters
st.sidebar.markdown("### 🔧 Training Hyperparameters")
epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=100, value=10)
seq_len = st.sidebar.number_input("Sequence Length", min_value=2, max_value=50, value=10)
lr = st.sidebar.number_input("Learning Rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f")

# Data refresh control
st.sidebar.markdown("### 🔄 Data Refresh")
if 'data' not in st.session_state:
    st.session_state.data = {}
if st.sidebar.button("Refresh All Data"):
    if all([consumer_key, consumer_secret, oauth_token, oauth_token_secret, account_id]):
        st.sidebar.info("Fetching data asynchronously...")
        client = ETradeClient(
            consumer_key, consumer_secret, oauth_token, oauth_token_secret, account_id, sandbox=True
        )
        try:
            st.session_state.data = asyncio.run(
                fetch_all_candles(client, st.session_state.symbols, interval='5min', days=1)
            )
            st.sidebar.success("Data refreshed.")
        except Exception as e:
            st.sidebar.error(f"Error fetching data: {e}")
    else:
        st.sidebar.error("Fill in credentials to refresh data.")

# Initialize session state defaults
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.training = False
    st.session_state.class_names = [
        "Hammer", "Bullish Engulfing", "Bearish Engulfing", "Doji", "Morning Star", "Evening Star"
    ]
if 'symbols' not in st.session_state:
    st.session_state.symbols = ["AAPL", "MSFT"]

# ─────── Train Model ──────────────────────────────────────────────────────────
if st.sidebar.button("Train Neural Model"):
    if not st.session_state.training:
        if all([consumer_key, consumer_secret, oauth_token, oauth_token_secret, account_id]):
            st.sidebar.info("Training started... please wait.")
            client = ETradeClient(
                consumer_key, consumer_secret, oauth_token, oauth_token_secret, account_id, sandbox=True
            )
            model = PatternNN()
            st.session_state.training = True
            def train():
                trained = train_pattern_model(
                    client, st.session_state.symbols, model,
                    epochs=int(epochs), seq_len=int(seq_len)
                )
                st.session_state.model = trained
                st.session_state.training = False
                st.sidebar.success("Training complete!")
            threading.Thread(target=train, daemon=True).start()
        else:
            st.sidebar.error("Please fill in credentials before training.")
    else:
        st.sidebar.warning("Training already in progress.")

# ─────── Symbol Management ────────────────────────────────────────────────────
st.sidebar.title("📈 Symbol Tracker")
new_symbol = st.sidebar.text_input("Add ticker (e.g. GOOGL)", key="new_symbol")
if st.sidebar.button("Add Symbol"):
    sym = new_symbol.strip().upper()
    if sym and sym not in st.session_state.symbols:
        st.session_state.symbols.append(sym)
        st.sidebar.success(f"Added {sym}")
    else:
        st.sidebar.warning("Empty or duplicate symbol.")

# ─────── MAIN LAYOUT ─────────────────────────────────────────────────────────
st.title("📊 E*Trade Candlestick Strategy Dashboard")

# Prepare data if not already fetched
if not st.session_state.data:
    if all([consumer_key, consumer_secret, oauth_token, oauth_token_secret, account_id]):
        for sym in st.session_state.symbols:
            st.session_state.data[sym] = get_candles_cached(
                consumer_key, consumer_secret, oauth_token, oauth_token_secret,
                account_id, sym, interval='5min', days=1, sandbox=True
            )

# Display each symbol
for sym, df in st.session_state.data.items():
    st.markdown(f"---\n### {sym}")
    try:
        # Plot candlestick
        fig = go.Figure(data=[
            go.Candlestick(
                x=df.index,
                open=df['open'], high=df['high'],
                low=df['low'], close=df['close'],
                name=sym
            )
        ])
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # Rule-based detections
        detections = []
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        third = df.iloc[-3] if len(df) > 2 else prev

        if CandlestickPatterns.is_hammer(last):
            detections.append("Hammer")
        if CandlestickPatterns.is_bullish_engulfing(prev, last):
            detections.append("Bullish Engulfing")
        if CandlestickPatterns.is_bearish_engulfing(prev, last):
            detections.append("Bearish Engulfing")
        if CandlestickPatterns.is_doji(last):
            detections.append("Doji")
        if CandlestickPatterns.is_morning_star(third, prev, last):
            detections.append("Morning Star")
        if CandlestickPatterns.is_evening_star(third, prev, last):
            detections.append("Evening Star")

        if detections:
            st.success("Rule patterns: " + ", ".join(detections))
        else:
            st.info("No rule-based patterns detected.")

        # Neural model prediction
        if st.session_state.model is not None:
            seq = torch.tensor(df.tail(seq_len).values[None], dtype=torch.float32)
            logits = st.session_state.model(seq)
            pred = int(torch.argmax(logits, dim=1).item())
            name = st.session_state.class_names[pred]
            st.info(f"Model prediction: {name}")
        else:
            st.warning("Neural model not trained yet.")

        # Order execution buttons
        buy_col, sell_col = st.columns(2)
        if buy_col.button(f"Buy {sym}", key=f"buy_{sym}"):
            resp = client.place_market_order(sym, 1, instruction="BUY")
            buy_col.success(f"BUY order placed: {resp}")
        if sell_col.button(f"Sell {sym}", key=f"sell_{sym}"):
            resp = client.place_market_order(sym, 1, instruction="SELL")
            sell_col.success(f"SELL order placed: {resp}")

    except Exception as e:
        st.error(f"Error for {sym}: {e}")

else:
    st.info("Enter your E*Trade credentials above to begin.")