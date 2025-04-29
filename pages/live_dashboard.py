import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import torch

from etrade_candlestick_bot import get_live_data
from model_manager import load_latest_model
from models.pattern_nn import PatternNN  # Import your actual model class

@st.cache_resource
def load_model_once():
    """Load the latest Pattern NN model once per session."""
    return load_latest_model(PatternNN)

def pattern_nn_live_predict(df: pd.DataFrame, window_size: int = 10) -> int:
    """Predict live action using the Pattern NN model."""
    if len(df) < window_size:
        return 0  # HOLD if not enough data

    model = load_model_once()
    model.eval()

    window = df.iloc[-window_size:][['open', 'high', 'low', 'close', 'volume']].values
    window = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
    output = model(window)

    prediction = torch.argmax(output).item()  # 0 = HOLD, 1 = BUY, 2 = SELL
    return prediction

# --- Streamlit UI ---

st.title("📡 Live Trading Dashboard")

st.subheader("Market Overview")
symbol = st.text_input("Ticker Symbol", value="AAPL")
refresh = st.button("Refresh Data")

if refresh:
    live_data = get_live_data(symbol)  # Should return dict or DataFrame

    if live_data:
        df_live = pd.DataFrame({
            'timestamp': live_data['timestamp'],
            'open': live_data['open'],
            'high': live_data['high'],
            'low': live_data['low'],
            'close': live_data['close'],
            'volume': live_data['volume']
        })
        df_live['timestamp'] = pd.to_datetime(df_live['timestamp'])
        df_live.set_index('timestamp', inplace=True)

        st.metric(label="Last Price", value=f"${df_live['close'].iloc[-1]:,.2f}")

        fig = go.Figure(data=[go.Candlestick(
            x=df_live.index,
            open=df_live['open'],
            high=df_live['high'],
            low=df_live['low'],
            close=df_live['close']
        )])
        st.plotly_chart(fig, use_container_width=True)

        # Model Prediction
        prediction = pattern_nn_live_predict(df_live)

        if prediction == 1:
            st.success("🚀 Model Signal: BUY")
        elif prediction == 2:
            st.error("🔻 Model Signal: SELL")
        else:
            st.info("💤 Model Signal: HOLD")
    else:
        st.error("Failed to fetch live data.")

st.subheader("Current Positions (Simulated)")
st.info("Position tracking not connected to broker yet. Future enhancement.")
