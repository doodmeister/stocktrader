import streamlit as st
from datetime import datetime

from core.streamlit.session_manager import SessionManager  # Added
from core.streamlit.dashboard_utils import (
    handle_streamlit_error,
    # initialize_dashboard_session_state, # Removed
    setup_page
)
from utils.backtester import run_backtest
from utils.logger import get_dashboard_logger

# Initialize SessionManager
session_manager = SessionManager(namespace_prefix="classic_strategy_backtest")  # Added

# Dashboard logger setup
logger = get_dashboard_logger(__name__)

# Initialize the page
setup_page(
    title="📈 Trading Strategy Backtester",
    logger_name=__name__,
    sidebar_title="Backtest Configuration"
)

# initialize_dashboard_session_state() # Removed

class ClassicStrategyBacktester:
    def __init__(self):
        pass
    
    def run(self):
        """Main dashboard application entry point."""
        st.sidebar.header("Backtest Configuration")

        symbol_value = session_manager.get_page_state("symbol", "AAPL")
        # Ensure symbol is a string before calling .upper()
        symbol_input = st.sidebar.text_input(
            "Symbol", 
            value=symbol_value,
            key=session_manager.get_widget_key("text_input", "symbol_input")
        )
        symbol = str(symbol_input).upper() if symbol_input is not None else ""
        session_manager.set_page_state("symbol", symbol)

        start_date_value = session_manager.get_page_state("start_date", datetime(2020, 1, 1))
        start_date = st.sidebar.date_input(
            "Start Date", 
            value=start_date_value, 
            key=session_manager.get_widget_key("date_input", "start_date_input")
        )
        session_manager.set_page_state("start_date", start_date)

        end_date_value = session_manager.get_page_state("end_date", datetime.today())
        end_date = st.sidebar.date_input(
            "End Date", 
            value=end_date_value, 
            key=session_manager.get_widget_key("date_input", "end_date_input")
        )
        session_manager.set_page_state("end_date", end_date)

        strategy_options = ["Moving Average Crossover", "RSI Strategy", "MACD Strategy"]
        strategy_value = session_manager.get_page_state("strategy", strategy_options[0])
        strategy = st.sidebar.selectbox(
            "Strategy", 
            options=strategy_options,
            index=strategy_options.index(strategy_value) if strategy_value in strategy_options else 0,
            key=session_manager.get_widget_key("selectbox", "strategy_selectbox")
        )
        session_manager.set_page_state("strategy", strategy)

        initial_capital_value = session_manager.get_page_state("initial_capital", 10000)
        initial_capital = st.sidebar.number_input(
            "Initial Capital ($)", 
            min_value=1000, 
            value=initial_capital_value, 
            step=1000,
            key=session_manager.get_widget_key("number_input", "initial_capital_input")
        )
        session_manager.set_page_state("initial_capital", initial_capital)

        risk_per_trade_value = session_manager.get_page_state("risk_per_trade", 1.0)
        risk_per_trade = st.sidebar.slider(
            "Risk per Trade (%)", 
            min_value=0.5, 
            max_value=5.0, 
            value=risk_per_trade_value, 
            step=0.5,
            key=session_manager.get_widget_key("slider", "risk_per_trade_slider")
        )
        session_manager.set_page_state("risk_per_trade", risk_per_trade)

        if start_date >= end_date:
            st.sidebar.error("Start date must be before end date.")

        st.title("📈 Trading Strategy Backtester")

        if st.sidebar.button("Run Backtest", key=session_manager.get_widget_key("button", "run_backtest_button")):
            try:
                results = run_backtest(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    strategy=strategy,
                    initial_capital=initial_capital,
                    risk_per_trade=risk_per_trade
                )

                st.subheader("Performance Metrics")
                st.write(results["metrics"])

                st.subheader("Equity Curve")
                st.line_chart(results["equity_curve"])

                st.subheader("Trade Log")
                st.dataframe(results["trade_log"])

            except Exception as e:
                st.error(f"An error occurred during backtesting: {e}")
                logger.error(f"Backtest error for {symbol} from {start_date} to {end_date}: {e}")
        
        # Display debug information (optional)
        if st.checkbox("Show Debug Info", key=session_manager.get_widget_key("checkbox", "show_debug_info_checkbox")):
            st.subheader("Debug Information")
            session_manager.debug_session_state()

if __name__ == "__main__":
    try:
        dashboard = ClassicStrategyBacktester()
        dashboard.run()
    except Exception as e:
        handle_streamlit_error(e, "Classic Strategy Backtester")
