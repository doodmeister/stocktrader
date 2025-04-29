import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from backtester import run_backtest  # Assuming you have a function like this

st.title("🧪 Strategy Backtesting")

st.subheader("Backtest Configuration")
with st.form("backtest_form"):
    symbol = st.text_input("Stock Symbol", value="AAPL")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    strategy = st.selectbox("Select Strategy", ["SMA Crossover", "Pattern Recognition"])
    fast_period = st.number_input("Fast Period (for SMA)", min_value=1, max_value=100, value=10)
    slow_period = st.number_input("Slow Period (for SMA)", min_value=10, max_value=200, value=30)
    run_button = st.form_submit_button("Run Backtest")

if run_button:
    st.info(f"Running {strategy} backtest for {symbol}...")
    results = run_backtest(symbol, start_date, end_date, strategy, fast_period, slow_period)

    st.success("Backtest Completed!")
    
    # Show metrics
    st.metric("Total Return", f"{results['total_return']:.2f}%")
    st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
    st.metric("Max Drawdown", f"{results['max_drawdown']:.2f}%")

    # Show equity curve
    st.subheader("Equity Curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results['equity_curve']['date'], y=results['equity_curve']['equity'], mode='lines'))
    st.plotly_chart(fig, use_container_width=True)

    # Show trades
    st.subheader("Trade List")
    st.dataframe(results['trades'])
