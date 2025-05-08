import streamlit as st
from pathlib import Path

# Import your logger setup
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)

st.set_page_config(
    page_title="StockTrader Dashboard",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Welcome to StockTrader Dashboard")

# Log that the dashboard was loaded
logger.info("StockTrader Dashboard loaded.")

# Load description from README.md (first section)
readme_path = Path(__file__).parent / "readme.md"
description = ""
if readme_path.exists():
    with open(readme_path, encoding="utf-8") as f:
        lines = f.readlines()
        # Get lines until the first '---' (markdown horizontal rule)
        for line in lines:
            if line.strip().startswith("---"):
                break
            description += line
    logger.info("Loaded description from readme.md.")
else:
    description = (
        "An enterprise-grade trading platform that combines classic technical analysis "
        "with machine learning for automated E*Trade trading. Features a Streamlit dashboard "
        "for real-time monitoring, robust risk management, and a comprehensive backtesting & ML pipeline."
    )
    logger.warning("readme.md not found. Using default description.")

st.markdown(description)

st.header("📂 Available Dashboards & Tools")

# List of pages (add or remove as needed)
pages = [
    {"name": "Live Trading Dashboard", "file": "live_dashboard.py"},
    {"name": "Data Visualization Dashboard", "file": "data_dashboard.py"},
    {"name": "Technical Analysis Tools", "file": "data_analysis.py"},
    {"name": "Model Training & Deployment", "file": "model_training.py"},
    {"name": "Neural Net Backtesting", "file": "nn_backtest.py"},
    {"name": "Classic Strategy Backtesting", "file": "classic_strategy_backtest.py"},
    {"name": "Candlestick Patterns Editor", "file": "patterns.py"},
    {"name": "Advanced AI Trading", "file": "advanced_ai_trade.py"},
    {"name": "Simple Trade", "file": "simple_trade.py"},
]

for page in pages:
    st.markdown(
        f"- [{page['name']}](./pages/{page['file']})"
    )
    logger.debug(f"Listed page: {page['name']} ({page['file']})")

st.info(
    "Use the sidebar or the links above to navigate to the different dashboards and tools. "
    "All features are accessible from this main entry point."
)