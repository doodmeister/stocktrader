import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import torch
import joblib
from datetime import date, timedelta
from typing import Callable, Optional, Any

from train.model_manager import ModelManager, load_latest_model
from patterns.patterns_nn import PatternNN
from core.streamlit.dashboard_utils import (
    initialize_dashboard_session_state,
    setup_page,
    handle_streamlit_error
)
from core.streamlit.session_manager import SessionManager # Add this import
from utils.backtester import run_backtest_wrapper

# Dashboard logger setup
from utils.logger import get_dashboard_logger
logger = get_dashboard_logger(__name__)

# Initialize the page (setup_page returns a logger, but we already have one)
setup_page(
    title="🧪 Strategy Backtesting",
    logger_name=__name__,
    sidebar_title="Backtest Configuration"
)

# Initialize SessionManager for conflict-free widget handling
session_manager = SessionManager(namespace_prefix="nn_backtest") # Instantiate SessionManager

# --- Model Loading Utilities ---

@st.cache_resource(show_spinner=False)
def load_pattern_nn(model_path: Optional[str] = None) -> Optional[PatternNN]:
    """
    Loads a PatternNN model from the specified path or loads the latest model.
    """
    try:
        if model_path:
            model = PatternNN()
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            return model
        model = load_latest_model(PatternNN)[0]
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Failed to load PatternNN model: {e}")
        st.error(f"Failed to load PatternNN model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_classic_model(model_path: str) -> Optional[Any]:
    """
    Loads a classic ML model from the specified path.
    """
    try:
        return joblib.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load classic ML model: {e}")
        st.error(f"Failed to load classic ML model: {e}")
        return None

# --- Prediction Functions ---

def pattern_nn_predict(df: pd.DataFrame, model_path: str, window_size: int = 10) -> pd.Series:
    """
    Generates trading signals using a PatternNN model.
    """
    model = load_pattern_nn(model_path)
    if model is None or df.empty:
        return pd.Series(0, index=df.index)
    signals = []
    if len(df) < window_size:
        return pd.Series(0, index=df.index)
    for idx in range(len(df)):
        if idx < window_size:
            signals.append(0)
            continue
        try:
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
        except Exception as e:
            logger.error(f"PatternNN prediction failed at idx {idx}: {e}")
            signals.append(0)
    return pd.Series(signals, index=df.index)

def classic_ml_predict(df: pd.DataFrame, model_path: str) -> pd.Series:
    """
    Generates trading signals using a classic ML model.
    """
    model = load_classic_model(model_path)
    if model is None or df.empty:
        return pd.Series(0, index=df.index)
    try:
        if hasattr(model, "feature_names_in_"):
            features = df[model.feature_names_in_]
        else:
            # Fallback: use a default feature set, update as needed
            features = df[['close', 'volume']]
        preds = model.predict(features)
        return pd.Series(preds, index=df.index)
    except Exception as e:
        logger.error(f"Classic ML prediction failed: {e}")
        st.error(f"Classic ML prediction failed: {e}")
        return pd.Series(0, index=df.index)

# --- Backtest Runner ---

def run_custom_strategy(
    symbol: str,
    start_date: date,
    end_date: date,
    signal_fn: Callable[[pd.DataFrame], pd.Series],
    initial_capital: float,
    risk_per_trade: float,
    model_type: Optional[str] = None
) -> Optional[dict]:
    """
    Runs a custom backtest using the provided signal function.    """
    from utils.backtester import BacktestConfig, Backtest, load_ohlcv
    from datetime import datetime
    try:
        # Convert date to datetime for load_ohlcv compatibility
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.min.time())
        
        df = load_ohlcv(symbol, start_datetime, end_datetime)
        if df.empty:
            st.error(f"No data found for {symbol} between {start_date} and {end_date}.")
            logger.warning(f"No data found for {symbol} between {start_date} and {end_date}.")
            return None
        config = BacktestConfig(
            initial_capital=initial_capital,
            position_size_limit=risk_per_trade
        )
        bt = Backtest(config)

        def wrapper(d):
            idx = d.index[-1]
            return signal_fn(d).loc[idx] if idx in d.index else 0

        results = bt.simulate({symbol: df}, wrapper)
        trades = pd.DataFrame([t.__dict__ for t in bt.trades])
        return {
            'metrics': results.dict(),
            'equity_curve': bt.equity_curve.reset_index().rename(columns={'index': 'date'}),
            'trade_log': trades,
            'model_type': model_type
        }
    except Exception as e:
        logger.error(f"Backtest simulation failed: {e}")
        st.error(f"Backtest simulation failed: {e}")
        return None

# --- Streamlit App ---

def initialize_backtest_state():
    defaults = {
        "results": None,
        "backtest_triggered": False,
        "selected_strategy": "Pattern NN",
        "trade_log_downloaded": False,
        "equity_curve_downloaded": False
    }
    for k, v in defaults.items():
        # Use set_page_state with if_not_exists logic handled manually for now
        if session_manager.get_page_state(k) is None:
            session_manager.set_page_state(k, v)

class NeuralNetworkBacktester:
    def __init__(self):
        pass
    
    def run(self):
        """Main dashboard application entry point."""
        initialize_dashboard_session_state()
        initialize_backtest_state()
        main()

def main():
    st.title("🧪 Strategy Backtesting")    # Model selection
    model_manager = ModelManager()
    try:
        model_files = model_manager.list_models()
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        st.error("Could not list models. Please check model directory and permissions.")
        return

    # Extract filenames from model info dictionaries
    classic_models = [f['name'] for f in model_files if f['name'].endswith(".joblib")]
    patternnn_models = [f['name'] for f in model_files if f['name'].endswith(".pth")]

    st.subheader("Backtest Configuration")    # Set default dates: 1 year ago to today
    default_start = date.today() - timedelta(days=365)
    default_end = date.today()

    with session_manager.form_container("backtest_form"):
        symbol = session_manager.create_text_input("Stock Symbol", value="AAPL", text_input_name="backtest_symbol")
        start_date = session_manager.create_date_input("Start Date", value=default_start, date_input_name="backtest_start_date")
        end_date = session_manager.create_date_input("End Date", value=default_end, date_input_name="backtest_end_date")
        strategy = session_manager.create_selectbox(
            "Select Strategy",
            options=["SMA Crossover", "Pattern NN"] + (["Classic ML Model"] if classic_models else []),            selectbox_name="backtest_strategy"
        )
        selected_classic_model = None
        selected_patternnn_model = None
        if strategy == "Classic ML Model":
            selected_classic_model = session_manager.create_selectbox("Select Classic ML Model", options=classic_models, selectbox_name="classic_model_select")
        if strategy == "Pattern NN":
            selected_patternnn_model = session_manager.create_selectbox("Select PatternNN Model", options=patternnn_models, selectbox_name="patternnn_model_select")
        initial_capital = session_manager.create_number_input("Initial Capital ($)", min_value=1000, value=100000, number_input_name="initial_capital")
        risk_per_trade = session_manager.create_number_input("Risk Per Trade (%)", min_value=0.1, max_value=100.0, value=1.0, number_input_name="risk_per_trade")
        run_button = st.form_submit_button("Run Backtest")

    if run_button:
        # Values from session_manager.create_* are directly available
        # No need to set them back into session_state manually here if using form_submit_button
        # st.session_state["symbol"] = symbol
        # st.session_state["start_date"] = start_date
        # st.session_state["end_date"] = end_date
        # st.session_state["strategy"] = strategy
        # st.session_state["selected_classic_model"] = selected_classic_model
        # st.session_state["selected_patternnn_model"] = selected_patternnn_model        # Validate dates
        if not start_date or not end_date:
            st.error("Please select both a start and end date.")
            return
        if start_date > end_date:
            st.error("Start date must be before end date.")
            return
        
        # Validate numeric parameters
        if initial_capital is None or initial_capital <= 0:
            st.error("Initial capital must be a positive number.")
            return
        if risk_per_trade is None or risk_per_trade <= 0:
            st.error("Risk per trade must be a positive number.")
            return

        st.info(f"Running {strategy} backtest for {symbol}...")
        with st.spinner("Running backtest, please wait..."):
            # Convert date to datetime for compatibility
            from datetime import datetime
            start_datetime = datetime.combine(start_date, datetime.min.time())
            end_datetime = datetime.combine(end_date, datetime.min.time())
            
            if strategy == "Pattern NN" and selected_patternnn_model:
                fn = lambda df: pattern_nn_predict(df, selected_patternnn_model)
                results = run_custom_strategy(
                    symbol, start_date, end_date, fn, initial_capital, risk_per_trade / 100, model_type=strategy
                )
                session_manager.set_page_state("results", results)
            elif strategy == "Classic ML Model" and selected_classic_model:
                fn = lambda df: classic_ml_predict(df, selected_classic_model)
                results = run_custom_strategy(
                    symbol, start_date, end_date, fn, initial_capital, risk_per_trade / 100, model_type=strategy
                )
                session_manager.set_page_state("results", results)
            else:
                results = run_backtest_wrapper(
                    symbol, start_datetime, end_datetime, "SMA Crossover", initial_capital, risk_per_trade / 100
                )
                if results is not None:
                    results["model_type"] = "SMA Crossover"
                session_manager.set_page_state("results", results)

        # Display results
        current_results = session_manager.get_page_state("results")
        if not current_results:
            st.warning("No results to display. Please check your configuration and data.")
            return

        if "metrics" in current_results and current_results["metrics"].get("num_trades", 0) == 0:
            st.warning("⚠️ No trades executed—try broadening your date range or adjusting strategy parameters")
        else:
            st.success("✅ Backtest completed")

            st.subheader("📊 Performance Metrics")
            metrics = current_results["metrics"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Return", f"{metrics.get('total_return', 0) * 100:.2f}%")
            col2.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
            col3.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0) * 100:.2f}%")

            st.subheader("📈 Equity Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=current_results['equity_curve']['date'],
                y=current_results['equity_curve']['equity'],
                mode='lines'
            ))
            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

            current_results_from_state = session_manager.get_page_state("results")
            if current_results_from_state:
                # results = st.session_state["results"] # Already have current_results
                st.subheader("📄 Trade Log")
                st.dataframe(current_results_from_state["trade_log"])

                # Download Trade Log
                if st.download_button(
                    "Download Trade Log",
                    current_results_from_state["trade_log"].to_csv(index=False).encode(),
                    "trade_log.csv",
                    "text/csv",
                    key=session_manager.get_unique_key("download_trade_log", "button")
                ):
                    session_manager.set_page_state("trade_log_downloaded", True)

                if session_manager.get_page_state("trade_log_downloaded"):
                    st.info("✅ Trade log download started!")
                    session_manager.set_page_state("trade_log_downloaded", False) # Reset after showing message

                # Download Equity Curve
                if st.download_button(
                    "Download Equity Curve",
                    current_results_from_state["equity_curve"].to_csv(index=False).encode(),
                    "equity_curve.csv",
                    "text/csv",
                    key=session_manager.get_unique_key("download_equity_curve", "button")
                ):
                    session_manager.set_page_state("equity_curve_downloaded", True)

                if session_manager.get_page_state("equity_curve_downloaded"):
                    st.info("✅ Equity curve download started!")
                    session_manager.set_page_state("equity_curve_downloaded", False) # Reset after showing message

# Execute the main function
if __name__ == "__main__":
    try:
        dashboard = NeuralNetworkBacktester()
        dashboard.run()
    except Exception as e:
        handle_streamlit_error(e, "Neural Network Backtester")

