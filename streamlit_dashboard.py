import streamlit as st
import pandas as pd
import torch
import threading
import plotly.graph_objs as go
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

if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.class_names = [
        "Hammer", "Bullish Engulfing", "Bearish Engulfing", "Doji", "Morning Star", "Evening Star"
    ]
    st.session_state.training = False

# Train model button
if st.sidebar.button("Train Neural Model"):
    if not st.session_state.training:
        if all([consumer_key, consumer_secret, oauth_token, oauth_token_secret, account_id]):
            st.sidebar.info("Training started... this may take a while.")
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
            st.sidebar.error("Please fill in all credentials before training.")
    else:
        st.sidebar.warning("Training already in progress.")

# ─────── Symbol Management ────────────────────────────────────────────────────
st.sidebar.title("📈 Symbol Tracker")
if 'symbols' not in st.session_state:
    st.session_state.symbols = ["AAPL", "MSFT"]

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

if all([consumer_key, consumer_secret, oauth_token, oauth_token_secret, account_id]):
    client = ETradeClient(
        consumer_key, consumer_secret, oauth_token, oauth_token_secret, account_id, sandbox=True
    )
    cols = st.columns(len(st.session_state.symbols))
    for col, sym in zip(cols, st.session_state.symbols):
        col.metric(label=sym, value="Tracking")

    for sym in st.session_state.symbols:
        st.markdown(f"---\n### {sym}")
        try:
            df = client.get_candles(sym, interval="5min", days=1)
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
            if st.button(f"Buy {sym}", key=f"buy_{sym}"):
                resp = client.place_market_order(sym, 1, instruction="BUY")
                st.success(f"BUY order placed: {resp}")
            if st.button(f"Sell {sym}", key=f"sell_{sym}"):
                resp = client.place_market_order(sym, 1, instruction="SELL")
                st.success(f"SELL order placed: {resp}")

        except Exception as e:
            st.error(f"Error for {sym}: {e}")
else:
    st.info("Enter your E*Trade credentials above to begin.")