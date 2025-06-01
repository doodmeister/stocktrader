"""
StockTrader Dashboard - Main Entry Point

Modular entry point for the StockTrader platform dashboard.
"""

import streamlit as st
from pathlib import Path
import sys
import time
from dotenv import load_dotenv

# Load environment variables at application startup
load_dotenv()

# Configure Streamlit page FIRST, before any other Streamlit commands
st.set_page_config(
    page_title="StockTrader Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/stocktrader',
        'Report a bug': 'https://github.com/your-repo/stocktrader/issues',
        'About': "StockTrader - Enterprise Trading Platform v1.0"
    }
)

# Hide Streamlit's default menu and footer for a cleaner look
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Add project root to path for imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import logging utilities
try:
    from utils.logger import configure_dashboard_logging, get_dashboard_logger
    from security.authentication import validate_session_security
    from core.dashboard_utils import DashboardStateManager
    
    # For main dashboard
    logger = configure_dashboard_logging()
    
except ImportError as e:
    # Fallback logging setup
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.error(f"Critical import error: {e}")
    
    st.error(f"Critical import error: {e}")
    
    # Provide fallback functions
    def configure_dashboard_logging(*args, **kwargs):
        return logging.getLogger(__name__)
    
    def validate_session_security():
        return True  # Allow access in development mode
    
    class DashboardStateManager:
        def initialize_session_state(self):
            pass
    
    st.warning("Some modules not found. Running in limited mode.")

# Import the main dashboard controller
from core.dashboard_controller import StockTraderMainDashboard


def main():
    """Main entry point for the StockTrader dashboard."""
    start_time = time.time()
    
    try:
        # Create and run the main dashboard
        dashboard = StockTraderMainDashboard()
        dashboard.run()
        
        # Log successful initialization
        load_time = time.time() - start_time
        logger.info(f"StockTrader Dashboard started successfully in {load_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Critical error starting dashboard: {e}")
        st.error("Failed to start dashboard. Please check logs for details.")
        st.exception(e)


if __name__ == "__main__":
    main()
