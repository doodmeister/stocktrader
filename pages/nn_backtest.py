import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import torch
from train.model_manager import ModelManager, load_latest_model
from patterns.patterns_nn import PatternNN
from utils.dashboard_utils import initialize_dashboard_session_state
from utils.backtester import run_backtest_wrapper

@st.cache_resource
def load_pattern_nn():
    return load_latest_model(PatternNN)[0]

@st.cache_resource
def load_classic_model(model_path):
    import joblib
    return joblib.load(model_path)

def pattern_nn_predict(df: pd.DataFrame, window_size: int = 10) -> pd.Series:
    model = load_pattern_nn()
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

def classic_ml_predict(df: pd.DataFrame, model_path: str) -> pd.Series:
    model = load_classic_model(model_path)
    # Preprocess to ensure correct features for the model
    # Example: replace ['close', 'volume'] with your actual feature set
    if hasattr(model, "feature_names_in_"):
        features = df[model.feature_names_in_]
    else:
        features = df[['close', 'volume']]  # Fallback/example; adjust as needed
    preds = model.predict(features)
    # Map predictions to signals: e.g., 1=buy, 2=sell, 0=hold
    return pd.Series(preds, index=df.index)

def run_custom_strategy(symbol, start_date, end_date, signal_fn, initial_capital, risk_per_trade):
    from utils.backtester import BacktestConfig, Backtest, load_ohlcv
    df = load_ohlcv(symbol, start_date, end_date)
    config = BacktestConfig(
        initial_capital=initial_capital,
        position_size_limit=risk_per_trade
    )
    bt = Backtest(config)

    def wrapper(d):
        idx = d.index[-1]
        return signal_fn(d).get(idx, 0)

    results = bt.simulate({symbol: df}, wrapper)
    trades = pd.DataFrame([t.__dict__ for t in bt.trades])
    return {
        'metrics': results.dict(),
        'equity_curve': bt.equity_curve.reset_index().rename(columns={'index': 'date'}),
        'trade_log': trades
    }

# --- Streamlit App ---

def main():
    initialize_dashboard_session_state()
    st.title("🧪 Strategy Backtesting")

    # Model selection
    model_manager = ModelManager()
    model_files = model_manager.list_models()
    classic_models = [f for f in model_files if f.endswith(".joblib")]

    st.subheader("Backtest Configuration")
    with st.form("backtest_form"):
        symbol = st.text_input("Stock Symbol", value="AAPL")
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")
        strategy = st.selectbox(
            "Select Strategy",
            ["SMA Crossover", "Pattern NN"] + (["Classic ML Model"] if classic_models else [])
        )
        selected_classic_model = None
        if strategy == "Classic ML Model":
            selected_classic_model = st.selectbox("Select Classic ML Model", classic_models)
        initial_capital = st.number_input("Initial Capital ($)", min_value=1000, value=100000)
        risk_per_trade = st.number_input("Risk Per Trade (%)", min_value=0.1, max_value=100.0, value=1.0)
        run_button = st.form_submit_button("Run Backtest")

    if run_button:
        st.info(f"Running {strategy} backtest for {symbol}...")
        with st.spinner("Running backtest, please wait..."):
            if strategy == "Pattern NN":
                signal_fn = pattern_nn_predict
                results = run_custom_strategy(
                    symbol, start_date, end_date, signal_fn, initial_capital, risk_per_trade
                )
            elif strategy == "Classic ML Model" and selected_classic_model:
                signal_fn = lambda df: classic_ml_predict(df, selected_classic_model)
                results = run_custom_strategy(
                    symbol, start_date, end_date, signal_fn, initial_capital, risk_per_trade
                )
            else:
                from pages.nn_backtest import sma_crossover_strategy
                # For classic strategies, you may still use run_backtest_wrapper if it expects a strategy name
                results = run_backtest_wrapper(
                    symbol, start_date, end_date, "SMA Crossover", initial_capital, risk_per_trade
                )

        # Display results
        if "metrics" in results and results["metrics"]["num_trades"] == 0:
            st.warning("⚠️ No trades executed—try broadening your date range or adjusting strategy parameters")
        else:
            st.success("✅ Backtest completed")

            st.subheader("📊 Performance Metrics")
            metrics = results["metrics"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Return", f"{metrics['total_return'] * 100:.2f}%")
            col2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            col3.metric("Max Drawdown", f"{metrics['max_drawdown'] * 100:.2f}%")

            st.subheader("📈 Equity Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results['equity_curve']['date'],
                y=results['equity_curve']['equity'],
                mode='lines'
            ))
            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📄 Trade Log")
            st.dataframe(results["trade_log"])

if __name__ == "__main__":
    main()
