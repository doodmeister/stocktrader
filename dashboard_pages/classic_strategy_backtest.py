import streamlit as st
import pandas as pd
from datetime import datetime
from utils.backtester import run_backtest  # Ensure this function is properly defined in backtester.py
from core.dashboard_utils import (
    initialize_dashboard_session_state,
    setup_page,
    handle_streamlit_error
)

# Dashboard logger setup
from utils.logger import get_dashboard_logger
logger = get_dashboard_logger(__name__)

# Initialize the page (setup_page returns a logger, but we already have one)
setup_page(
    title="📈 Trading Strategy Backtester",
    logger_name=__name__,
    sidebar_title="Backtest Configuration"
)

# Set Streamlit page configuration
# st.set_page_config(  # Handled by main dashboard
#     page_title="Trading Strategy Backtester",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# Initialize session state
initialize_dashboard_session_state()

class ClassicStrategyBacktester:
    def __init__(self):
        pass
    
    def run(self):
        """Main dashboard application entry point."""
        # Sidebar inputs
        st.sidebar.header("Backtest Configuration")

        symbol = st.sidebar.text_input("Symbol", value="AAPL").upper()
        start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1))
        end_date = st.sidebar.date_input("End Date", value=datetime.today())

        strategy_options = ["Moving Average Crossover", "RSI Strategy", "MACD Strategy"]
        strategy = st.sidebar.selectbox("Strategy", options=strategy_options)

        initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=1000, value=10000, step=1000)
        risk_per_trade = st.sidebar.slider("Risk per Trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

        # Validate date input
        if start_date >= end_date:
            st.sidebar.error("Start date must be before end date.")

        # Main content
        st.title("📈 Trading Strategy Backtester")

        def some_func():
            logger.info("Starting dashboard")
            logger.warning("Something might be wrong")

        if st.sidebar.button("Run Backtest"):
            try:
                # Run backtest
                results = run_backtest(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    strategy=strategy,
                    initial_capital=initial_capital,
                    risk_per_trade=risk_per_trade
                )

                # Display results
                st.subheader("Performance Metrics")
                st.write(results["metrics"])

                st.subheader("Equity Curve")
                st.line_chart(results["equity_curve"])

                st.subheader("Trade Log")
                st.dataframe(results["trade_log"])

            except Exception as e:
                st.error(f"An error occurred during backtesting: {e}")
                logger.error(f"Backtest error for {symbol} from {start_date} to {end_date}: {e}")

# Execute the main function
if __name__ == "__main__":
    try:
        dashboard = ClassicStrategyBacktester()
        dashboard.run()
    except Exception as e:
        handle_streamlit_error(e, "Classic Strategy Backtester")
