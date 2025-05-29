"""
Dashboard Controller Module

Main dashboard controller for the StockTrader platform.
Orchestrates UI rendering, page management, and system health monitoring.
"""

import streamlit as st
import time
import logging
from typing import Dict, List, Optional
from pathlib import Path

from core.page_loader import PageLoader
from core.health_checks import HealthChecker
from core.ui_renderer import UIRenderer
from utils.logger import get_dashboard_logger


class StockTraderMainDashboard:
    """
    Main dashboard controller for the StockTrader platform.
    Orchestrates UI rendering, page management, and system health monitoring.
    """
    
    def __init__(self):
        """Initialize the main dashboard."""
        self.start_time = time.time()
        
        # Use the global logger
        self.logger = get_dashboard_logger(__name__)
          # Initialize session state early for better performance
        self._initialize_session_state()
          # Initialize core components
        self.page_loader = PageLoader(self.logger)
        self.health_checker = HealthChecker()
        self.ui_renderer = UIRenderer(self.logger)
        
        # Initialize state manager
        self.state_manager = self._initialize_state_manager()
        
        # Cache pages configuration to avoid reloading on every run
        if 'pages_config_cache' not in st.session_state:
            st.session_state.pages_config_cache = self.page_loader.load_pages_configuration()
        self.pages_config = st.session_state.pages_config_cache
    
    def _initialize_session_state(self) -> None:
        """Initialize all session state variables with defaults."""
        session_defaults = {
            'current_page': 'home',
            'page_history': ['home'],
            'load_time': time.time(),
            'dashboard_initialized': True,
            'security_validated': False,
            'last_health_check': 0,            'navigation_count': 0
        }
        
        for key, default_value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def _initialize_state_manager(self) -> Optional[object]:
        """Initialize dashboard state manager with error handling."""
        try:
            from core.dashboard_utils import DashboardStateManager
            return DashboardStateManager()
        except Exception as e:
            self.logger.error(f"Failed to initialize state manager: {e}")
            return None
    
    def _render_home_page(self) -> None:
        """Render the main dashboard home page."""
        self.ui_renderer.render_home_page(self.pages_config, self.state_manager)
    
    def _load_project_description(self) -> str:
        """Load project description from README with robust error handling."""
        try:
            # Try different possible README file names
            readme_files = ["readme.md", "README.md", "README.txt", "readme.txt"]
            project_root = Path(__file__).parent.parent
            
            for readme_file in readme_files:
                readme_path = project_root / readme_file
                if readme_path.exists():
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract first few paragraphs for description
                    lines = content.split('\n')
                    description_lines = []
                    found_content = False
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            if found_content:
                                break
                            continue
                        if line.startswith('#'):
                            continue  # Skip headers for now
                        description_lines.append(line)
                        found_content = True
                        if len(description_lines) >= 3:  # Limit to first 3 content lines
                            break
                    
                    return '\n\n'.join(description_lines) if description_lines else "StockTrader Platform - Enterprise trading solution"
            
            # Fallback if no README found
            return "StockTrader Platform - Advanced trading and analysis tools"
            
        except Exception as e:
            self.logger.warning(f"Could not load project description: {e}")
            return "StockTrader Platform - Trading tools and analytics"    # Note: UI rendering methods moved to UIRenderer class
    
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
            
            # Show current page info
            if st.session_state.current_page != 'home':
                st.subheader("📄 Current Page")
                current_page_config = next(
                    (p for p in self.pages_config if p.file == st.session_state.current_page),
                    None                )
                if current_page_config:
                    st.info(f"**{current_page_config.name}**\n\n{current_page_config.description}")
            
            # Health checks - delegate to health checker
            self.health_checker.render_health_status(self.pages_config, self.state_manager)
              # Page statistics
            st.subheader("📊 Page Statistics")
            total_pages = len(self.pages_config)
            active_pages = len([p for p in self.pages_config if p.is_active])
            
            st.metric("Total Pages", total_pages)
            st.metric("Active Pages", active_pages)
            if total_pages > 0:
                availability_pct = (active_pages / total_pages) * 100
                st.metric("Availability", f"{availability_pct:.1f}%")
    
    def _handle_security_validation(self) -> bool:
        """
        Perform security validation for the session.
        
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Import here to avoid circular imports
            from ..security.authentication import validate_session_security
            return validate_session_security()
        except ImportError:
            self.logger.warning("Security validation module not available")
            return True
        except Exception as e:
            self.logger.error(f"Security validation failed: {e}")
            st.error("Security validation failed. Please contact support.")
            return False
    
    def _validate_dependencies(self) -> bool:
        """Validate that all critical dependencies are available."""
        critical_modules = [
            'streamlit',
            'pandas',
            'numpy',
            'plotly',
            'yfinance'
        ]
        
        missing_modules = []
        for module in critical_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            self.logger.error(f"Missing critical dependencies: {missing_modules}")
            st.error(f"Missing dependencies: {', '.join(missing_modules)}")
            return False
        
        return True
    
    def run(self) -> None:
        """
        Main dashboard execution method.
        
        Handles page routing, security validation, and rendering.
        """
        try:
            start_time = time.time()
            
            # Security validation
            if not self._handle_security_validation():
                st.stop()
            
            # Dependency validation
            if not self._validate_dependencies():
                st.stop()
            
            # Render system info in sidebar
            self._render_system_info()
            
            # Main content routing
            current_page = st.session_state.get('current_page', 'home')
            
            if current_page == 'home':
                self._render_home_page()
            else:
                # Load and execute the selected page
                self.page_loader.load_and_execute_page(current_page)
            
            # Track performance
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
