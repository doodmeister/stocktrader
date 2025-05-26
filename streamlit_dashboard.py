"""
StockTrader Dashboard - Main Entry Point

Enterprise-grade trading platform dashboard that serves as the central hub
for accessing all trading tools, analysis features, and ML capabilities.

This module provides:
- Dynamic page discovery and navigation
- Robust error handling and logging
- Performance monitoring
- Security validation
- User session management
"""

import streamlit as st
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import sys
import os
from dataclasses import dataclass

# Add project root to path for imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from utils.logger import setup_logger
    from utils.security import validate_session_security
    from core.dashboard_utils import DashboardStateManager
except ImportError as e:
    st.error(f"Critical import error: {e}")
    st.error("Please ensure all dependencies are installed and the project structure is correct.")
    st.stop()


@dataclass
class PageConfig:
    """Configuration for dashboard pages."""
    name: str
    file: str
    description: str
    category: str
    requires_auth: bool = False
    is_active: bool = True


class StockTraderMainDashboard:
    """
    Main dashboard controller for the StockTrader platform.
    
    Handles navigation, security, performance monitoring, and provides
    a centralized entry point for all platform features.
    """
    
    def __init__(self):
        """Initialize the main dashboard."""
        self.logger = self._setup_logging()
        self.state_manager = self._initialize_state_manager()
        self.pages_config = self._load_pages_configuration()
        
    def _setup_logging(self) -> object:
        """Setup logging with error handling."""
        try:
            logger = setup_logger(__name__)
            return logger
        except Exception as e:
            st.error(f"Failed to initialize logging: {e}")
            # Create a basic logger as fallback
            import logging
            logging.basicConfig(level=logging.INFO)
            return logging.getLogger(__name__)
    
    def _initialize_state_manager(self) -> Optional[object]:
        """Initialize dashboard state manager with error handling."""
        try:
            return DashboardStateManager()
        except Exception as e:
            self.logger.error(f"Failed to initialize state manager: {e}")
            return None
    
    def _load_pages_configuration(self) -> List[PageConfig]:
        """
        Load and validate page configurations.
        
        Returns:
            List of validated page configurations
        """
        try:
            # Define available pages with metadata
            pages = [
                PageConfig(
                    name="Live Trading Dashboard",
                    file="advanced_ai_trade.py",
                    description="Real-time AI-powered trading with risk management",
                    category="Trading",
                    requires_auth=True
                ),
                PageConfig(
                    name="Data Visualization Dashboard",
                    file="data_dashboard_v2.py",
                    description="Download and visualize market data with quality checks",
                    category="Data"
                ),
                PageConfig(
                    name="Technical Analysis Tools",
                    file="data_analysis_v2.py",
                    description="Advanced technical analysis and pattern detection",
                    category="Analysis"
                ),
                PageConfig(
                    name="Model Training & Deployment",
                    file="model_training.py",
                    description="Train and deploy ML models for pattern recognition",
                    category="Machine Learning"
                ),
                PageConfig(
                    name="Neural Net Backtesting",
                    file="nn_backtest.py",
                    description="Backtest neural network trading strategies",
                    category="Backtesting"
                ),
                PageConfig(
                    name="Classic Strategy Backtesting",
                    file="classic_strategy_backtest.py",
                    description="Backtest traditional technical analysis strategies",
                    category="Backtesting"
                ),
                PageConfig(
                    name="Candlestick Patterns Editor",
                    file="patterns.py",
                    description="Create and edit custom candlestick patterns",
                    category="Analysis"
                ),
                PageConfig(
                    name="Model Visualizer",
                    file="model_visualizer.py",
                    description="Visualize model performance and predictions",
                    category="Machine Learning"
                )
            ]
            
            # Validate that page files exist
            validated_pages = []
            pages_dir = project_root / "pages"
            
            for page in pages:
                page_path = pages_dir / page.file
                if page_path.exists():
                    validated_pages.append(page)
                    self.logger.debug(f"Validated page: {page.name}")
                else:
                    self.logger.warning(f"Page file not found: {page.file}")
                    # Mark as inactive but keep in list for debugging
                    page.is_active = False
                    validated_pages.append(page)
            
            return validated_pages
            
        except Exception as e:
            self.logger.error(f"Failed to load page configurations: {e}")
            return []
    
    def _configure_streamlit_page(self) -> None:
        """Configure Streamlit page settings with error handling."""
        try:
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
        except Exception as e:
            self.logger.error(f"Failed to configure Streamlit page: {e}")
    
    def _load_project_description(self) -> str:
        """
        Load project description from README with robust error handling.
        
        Returns:
            Project description string
        """
        readme_path = project_root / "readme.md"
        default_description = (
            "An enterprise-grade trading platform that combines classic technical analysis "
            "with machine learning for automated E*Trade trading. Features a Streamlit dashboard "
            "for real-time monitoring, robust risk management, and a comprehensive backtesting & ML pipeline."
        )
        
        try:
            if not readme_path.exists():
                self.logger.warning("readme.md not found. Using default description.")
                return default_description
            
            with open(readme_path, encoding="utf-8") as f:
                lines = f.readlines()
            
            # Extract content until first horizontal rule
            description_lines = []
            for line in lines:
                line_stripped = line.strip()
                if line_stripped.startswith("---") and description_lines:
                    break
                if line_stripped and not line_stripped.startswith("#"):
                    description_lines.append(line)
            
            if description_lines:
                description = "".join(description_lines).strip()
                self.logger.info("Loaded description from readme.md.")
                return description
            else:
                self.logger.warning("No content found in readme.md. Using default description.")
                return default_description
                
        except Exception as e:
            self.logger.error(f"Error reading readme.md: {e}")
            return default_description
    
    def _render_header(self) -> None:
        """Render dashboard header with branding and status."""
        st.title("📊 StockTrader Dashboard")
        
        # Add status indicators
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col2:
            if self.state_manager:
                st.success("🟢 System Online")
            else:
                st.warning("🟡 Limited Mode")
        
        with col3:
            st.metric("Pages Available", len([p for p in self.pages_config if p.is_active]))
    
    def _render_description(self) -> None:
        """Render project description section."""
        description = self._load_project_description()
        
        with st.expander("📖 About StockTrader Platform", expanded=False):
            st.markdown(description)
            
            # Add key features
            st.subheader("🚀 Key Features")
            features = [
                "Real-time trading with E*Trade integration",
                "Advanced ML pattern recognition",
                "Comprehensive risk management",
                "Backtesting & strategy validation",
                "Multi-channel notifications",
                "Enterprise-grade security"
            ]
            
            for feature in features:
                st.markdown(f"• {feature}")
    
    def _render_navigation_menu(self) -> None:
        """Render categorized navigation menu."""
        st.header("🧭 Navigation Menu")
        
        # Group pages by category
        categories = {}
        for page in self.pages_config:
            if page.category not in categories:
                categories[page.category] = []
            categories[page.category].append(page)
        
        # Render each category
        for category, pages in categories.items():
            with st.expander(f"📁 {category}", expanded=True):
                for page in pages:
                    self._render_page_link(page)
    
    def _render_page_link(self, page: PageConfig) -> None:
        """
        Render individual page link with status and description.
        
        Args:
            page: Page configuration object
        """
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if page.is_active:
                page_url = f"./pages/{page.file}"
                st.markdown(f"**[{page.name}]({page_url})**")
                st.caption(page.description)
            else:
                st.markdown(f"**{page.name}** ⚠️ *Unavailable*")
                st.caption(f"{page.description} (File not found)")
        
        with col2:
            if page.is_active:
                if page.requires_auth:
                    st.caption("🔒 Auth Required")
                else:
                    st.caption("🌐 Public")
            else:
                st.caption("❌ Inactive")
    
    def _render_system_info(self) -> None:
        """Render system information and health checks."""
        with st.sidebar:
            st.header("🔧 System Info")
            
            # Performance metrics
            if hasattr(st, 'session_state'):
                if 'load_time' not in st.session_state:
                    st.session_state.load_time = time.time()
                
                uptime = time.time() - st.session_state.load_time
                st.metric("Session Uptime", f"{uptime:.1f}s")
            
            # Health checks
            health_checks = self._perform_health_checks()
            
            st.subheader("🏥 Health Status")
            for check_name, status in health_checks.items():
                if status:
                    st.success(f"✅ {check_name}")
                else:
                    st.error(f"❌ {check_name}")
    
    def _perform_health_checks(self) -> Dict[str, bool]:
        """
        Perform system health checks.
        
        Returns:
            Dictionary of health check results
        """
        checks = {}
        
        try:
            # Check critical directories
            checks["Data Directory"] = (project_root / "data").exists()
            checks["Models Directory"] = (project_root / "models").exists()
            checks["Logs Directory"] = (project_root / "logs").exists()
            
            # Check configuration
            env_file = project_root / ".env"
            checks["Environment Config"] = env_file.exists()
            
            # Check page files
            active_pages = len([p for p in self.pages_config if p.is_active])
            total_pages = len(self.pages_config)
            checks["Page Availability"] = active_pages == total_pages
            
            # Check state manager
            checks["State Manager"] = self.state_manager is not None
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            checks["Health Check System"] = False
        
        return checks
    
    def _render_footer(self) -> None:
        """Render dashboard footer with additional information."""
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📚 Documentation**")
            st.markdown("• [User Manual](./docs/manual.md)")
            st.markdown("• [API Reference](./docs/api.md)")
        
        with col2:
            st.markdown("**🔧 Support**")
            st.markdown("• [GitHub Issues](https://github.com/your-repo/stocktrader/issues)")
            st.markdown("• [Configuration Guide](./docs/config.md)")
        
        with col3:
            st.markdown("**⚖️ Legal**")
            st.markdown("• [License](./LICENSE)")
            st.markdown("• [Terms of Use](./docs/terms.md)")
        
        # Version and copyright
        st.markdown(
            "<div style='text-align: center; color: #666; margin-top: 2rem;'>"
            "StockTrader Platform v1.0 | © 2025 | Licensed under MIT"
            "</div>",
            unsafe_allow_html=True
        )
    
    def _handle_security_validation(self) -> bool:
        """
        Perform security validation for the session.
        
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Validate session security if module is available
            if 'validate_session_security' in globals():
                return validate_session_security()
            else:
                self.logger.warning("Security validation module not available")
                return True
        except Exception as e:
            self.logger.error(f"Security validation failed: {e}")
            st.error("Security validation failed. Please contact support.")
            return False
    
    def run(self) -> None:
        """
        Main dashboard execution method.
        
        Orchestrates the entire dashboard rendering process with
        comprehensive error handling and performance monitoring.
        """
        start_time = time.time()
        
        try:
            # Configure Streamlit
            self._configure_streamlit_page()
            
            # Perform security validation
            if not self._handle_security_validation():
                st.stop()
            
            # Initialize session state
            if self.state_manager:
                self.state_manager.initialize_session_state()
            
            # Render dashboard components
            self._render_header()
            self._render_description()
            self._render_navigation_menu()
            self._render_system_info()
            self._render_footer()
            
            # Log successful load
            load_time = time.time() - start_time
            self.logger.info(f"StockTrader Dashboard loaded successfully in {load_time:.2f}s")
            
            # Performance warning for slow loads
            if load_time > 3.0:
                self.logger.warning(f"Dashboard load time was slow: {load_time:.2f}s")
                st.warning("⚠️ Dashboard loaded slowly. Consider checking system resources.")
            
        except Exception as e:
            self.logger.error(f"Critical error in dashboard execution: {e}")
            st.error("A critical error occurred. Please refresh the page or contact support.")
            st.exception(e)  # Show exception in development
        
        finally:
            # Cleanup and final logging
            self.logger.debug("Dashboard render cycle completed")


def main():
    """Main entry point for the StockTrader dashboard."""
    try:
        dashboard = StockTraderMainDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Failed to initialize StockTrader Dashboard: {e}")
        st.error("Please check the logs and ensure all dependencies are properly installed.")


if __name__ == "__main__":
    main()