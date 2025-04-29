import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from backtester import run_backtest_wrapper
import torch
from model_manager import load_model  # Assuming you have this
import numpy as np

def pattern_nn_predict(df: pd.DataFrame, model_path: str = "models/pattern_nn_v1.pth") -> pd.Series:
    """Generate trading signals using a pre-trained Pattern Neural Network."""
    model = load_model(model_path)
    model.eval()

    signals = []

    # Assume we use last N candles as input features
    window_size = 10  # Example: past 10 days
    if len(df) < window_size:
        return pd.Series(0, index=df.index)  # Not enough data

    for idx in range(len(df)):
        if idx < window_size:
            signals.append(0)
            continue
        window = df.iloc[idx - window_size:idx][['open', 'high', 'low', 'close', 'volume']].values
        window = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        output = model(window)

        prediction = torch.argmax(output).item()  # Example: 0 = HOLD, 1 = BUY, 2 = SELL
        if prediction == 1:
            signals.append(1)
        elif prediction == 2:
            signals.append(-1)
        else:
            signals.append(0)

    return pd.Series(signals, index=df.index)

st.title("🧪 Strategy Backtesting")

st.subheader("Backtest Configuration")
with st.form("backtest_form"):
    symbol = st.text_input("Stock Symbol", value="AAPL")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    strategy = st.selectbox("Select Strategy", ["SMA Crossover"])  # Future: Add more
    fast_period = st.number_input("Fast Period (for SMA)", min_value=1, max_value=100, value=10)
    slow_period = st.number_input("Slow Period (for SMA)", min_value=10, max_value=200, value=30)
    initial_capital = st.number_input("Initial Capital ($)", min_value=1000, value=100000)
    run_button = st.form_submit_button("Run Backtest")

# Define available strategies
def sma_crossover_strategy(df: pd.DataFrame, fast_period: int = 10, slow_period: int = 30) -> pd.Series:
    """Simple SMA crossover trading strategy."""
    sma_fast = df["close"].rolling(window=fast_period).mean()
    sma_slow = df["close"].rolling(window=slow_period).mean()
    signals = pd.Series(0, index=df.index)
    signals[sma_fast > sma_slow] = 1
    signals[sma_fast < sma_slow] = -1
    return signals

if run_button:
    st.info(f"Running {strategy} backtest for {symbol}...")

    # Pick strategy function dynamically
    if strategy == "SMA Crossover":
    strategy_fn = lambda df: sma_crossover_strategy(df, fast_period, slow_period)
elif strategy == "Pattern NN":
    strategy_fn = lambda df: pattern_nn_predict(df)
else:
    st.error(f"Strategy {strategy} not implemented.")
    st.stop()

    results = run_backtest_wrapper(symbol, start_date, end_date, strategy_fn, initial_capital)

    st.success("Backtest Completed!")

    # Show key metrics
    st.metric("Total Return", f"{results['total_return']:.2f}%")
    st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
    st.metric("Max Drawdown", f"{results['max_drawdown']:.2f}%")

    # Show equity curve
    st.subheader("Equity Curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results['equity_curve']['date'], y=results['equity_curve']['equity'], mode='lines', name='Equity'))
    st.plotly_chart(fig, use_container_width=True)

    # Show trade list
    st.subheader("Executed Trades")
    st.dataframe(results['trades'])
