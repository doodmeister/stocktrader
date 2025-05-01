import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import torch

from utils.backtester import run_backtest_wrapper
from utils.model_manager import load_latest_model
from utils.patterns_nn import PatternNN  # <-- Import your model class (adjust if needed)

@st.cache_resource
def load_model_once():
    """Load the latest Pattern NN model once for the session."""
    return load_latest_model(PatternNN)

def pattern_nn_predict(df: pd.DataFrame, window_size: int = 10) -> pd.Series:
    """Generate trading signals using a pre-trained Pattern Neural Network."""
    model = load_model_once()
    model.eval()

    signals = []

    if len(df) < window_size:
        return pd.Series(0, index=df.index)

    for idx in range(len(df)):
        if idx < window_size:
            signals.append(0)
            continue
        window = df.iloc[idx - window_size:idx][['open', 'high', 'low', 'close', 'volume']].values
        window = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        output = model(window)

        prediction = torch.argmax(output).item()
        if prediction == 1:
            signals.append(1)
        elif prediction == 2:
            signals.append(-1)
        else:
            signals.append(0)

    return pd.Series(signals, index=df.index)

def sma_crossover_strategy(df: pd.DataFrame, fast_period: int = 10, slow_period: int = 30) -> pd.Series:
    """Simple SMA crossover trading strategy."""
    sma_fast = df["close"].rolling(window=fast_period).mean()
    sma_slow = df["close"].rolling(window=slow_period).mean()
    signals = pd.Series(0, index=df.index)
    signals[sma_fast > sma_slow] = 1
    signals[sma_fast < sma_slow] = -1
    return signals

# --- Streamlit App ---

st.title("🧪 Strategy Backtesting")

st.subheader("Backtest Configuration")
with st.form("backtest_form"):
    symbol = st.text_input("Stock Symbol", value="AAPL")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    strategy = st.selectbox("Select Strategy", ["SMA Crossover", "Pattern NN"])
    fast_period = st.number_input("Fast Period (for SMA)", min_value=1, max_value=100, value=10)
    slow_period = st.number_input("Slow Period (for SMA)", min_value=10, max_value=200, value=30)
    initial_capital = st.number_input("Initial Capital ($)", min_value=1000, value=100000)
    risk_per_trade = st.number_input("Risk Per Trade (%)", min_value=0.1, max_value=100.0, value=1.0)
    run_button = st.form_submit_button("Run Backtest")

if run_button:
    st.info(f"Running {strategy} backtest for {symbol}...")

    results = run_backtest_wrapper(symbol, start_date, end_date, strategy, initial_capital, risk_per_trade)

    if "metrics" in results and results["metrics"]["num_trades"] == 0:
        st.warning("⚠️ No trades executed—try broadening your date range or strategy parameters")

    st.success("Backtest Completed!")

    # Metrics
    st.metric("Total Return", f"{results['metrics']['total_return']:.2f}%")
    st.metric("Sharpe Ratio", f"{results['metrics']['sharpe_ratio']:.2f}")
    st.metric("Max Drawdown", f"{results['metrics']['max_drawdown']:.2f}%")

    # Equity Curve
    st.subheader("Equity Curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results['equity_curve']['date'], y=results['equity_curve']['equity'], mode='lines'))
    st.plotly_chart(fig, use_container_width=True)

    # Trades
    st.subheader("Executed Trades")
    st.dataframe(results['trades'])
