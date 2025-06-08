"""
StockTrader Dashboard - Main Entry Point

Modular entry point for the StockTrader platform dashboard.
"""

import streamlit as st
from pathlib import Path
import sys
import time
from dotenv import load_dotenv
from typing import Literal, Callable, Any, Type # Added Callable, Any, Type

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
# These names will be defined either in the try block by imports
# or in the except block by fallback definitions/assignments.
logger: Any 
validate_session_security: Callable[[], Literal[True]]
DashboardStateManager: Type[Any]
PageLoader: Type[Any]
HealthChecker: Type[Any]
configure_dashboard_logging: Callable[..., Any]


try:
    from utils.logger import configure_dashboard_logging as imported_configure_dashboard_logging, get_dashboard_logger
    from security.authentication import validate_session_security as _original_validate_session_security
    from core.streamlit.dashboard_utils import DashboardStateManager as ImportedDashboardStateManager
    from core.streamlit.page_loader import PageLoader as ImportedPageLoader
    from core.streamlit.health_checks import HealthChecker as ImportedHealthChecker
    
    configure_dashboard_logging = imported_configure_dashboard_logging
    logger = configure_dashboard_logging()
    DashboardStateManager = ImportedDashboardStateManager
    PageLoader = ImportedPageLoader
    HealthChecker = ImportedHealthChecker

    # Define the wrapper that conforms to the expected Literal[True] signature
    def validate_session_security_impl() -> Literal[True]:
        """
        Wrapper for security.authentication.validate_session_security
        to ensure it conforms to the expected () -> Literal[True] signature.
        Raises RuntimeError if the underlying function returns False.
        """
        is_valid = _original_validate_session_security()
        if not is_valid: # This means is_valid is False
            error_message = (
                "CRITICAL: Session validation failed. "
                "security.authentication.validate_session_security returned False, "
                "but the system expected True. Halting execution."
            )
            logger.critical(error_message)
            raise RuntimeError(error_message)
        return True # If is_valid was True, we return True, conforming to Literal[True]
    validate_session_security = validate_session_security_impl
    
except ImportError as e:
    # Fallback logging setup
    import logging as py_logging # Alias to avoid conflict
    py_logging.basicConfig(level=py_logging.INFO)
    logger = py_logging.getLogger(__name__) # Assign to global logger
    logger.error(f"Critical import error: {e}")
    
    if 'st' in globals() and hasattr(st, 'error'): # Check if st is available
        st.error(f"Critical import error: {e}")
    
    # Provide fallback functions
    def configure_dashboard_logging_fallback(*args, **kwargs): # type: ignore
        return logger # Return the already created fallback logger
    configure_dashboard_logging = configure_dashboard_logging_fallback
    
    # Fallback validate_session_security with explicit Literal[True] type hint
    def validate_session_security_fallback() -> Literal[True]: # MODIFIED
        """Fallback session validation. Always returns True in case of import errors."""
        logger.warning(
            "Using fallback validate_session_security due to import error. "
            "Authentication/authorization checks may be bypassed."
        )
        return True
    validate_session_security = validate_session_security_fallback
    
    # Define fallback classes with different names
    class _FallbackDashboardStateManager:
        def initialize_session_state(self):
            logger.info("Using fallback DashboardStateManager.initialize_session_state")
            pass

    class _FallbackPageLoader: # type: ignore
        def __init__(self, *args, **kwargs): logger.info("Using fallback PageLoader.")
        def discover_pages(self, *args, **kwargs): 
            logger.info("Fallback PageLoader.discover_pages called.")
            return {}
        def run_page(self, *args, **kwargs):
            logger.info("Fallback PageLoader.run_page called.")
            if 'st' in globals() and hasattr(st, 'warning'):
                st.warning("Page loading is disabled due to import errors (fallback PageLoader).")

    class _FallbackHealthChecker: # type: ignore
        def __init__(self, *args, **kwargs): logger.info("Using fallback HealthChecker.")
        def run_checks(self, *args, **kwargs):
            logger.info("Fallback HealthChecker.run_checks called.")
            return True # Minimal success
        def display_results(self, *args, **kwargs):
            logger.info("Fallback HealthChecker.display_results called.")
            if 'st' in globals() and hasattr(st, 'info'):
                 st.info("Health checks running in minimal mode due to import errors (fallback HealthChecker).")

    # Assign fallbacks to the global names
    DashboardStateManager = _FallbackDashboardStateManager # type: ignore
    PageLoader = _FallbackPageLoader # type: ignore
    HealthChecker = _FallbackHealthChecker # type: ignore
    
    if 'st' in globals() and hasattr(st, 'warning'): # Check if st is available
        st.warning("Some core modules (e.g., PageLoader, HealthChecker) might not have been imported. Running in limited mode.")

# Import the main dashboard controller
from core.streamlit.dashboard_controller import StockTraderMainDashboard # Corrected path


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
