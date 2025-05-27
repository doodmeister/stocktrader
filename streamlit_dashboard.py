"""
StockTrader Dashboard - Main Entry Point

Enterprise-grade trading platform dashboard that serves as the central hub
for accessing all trading tools, analysis features, and ML capabilities.
"""

import streamlit as st
from pathlib import Path
import time
import logging
from typing import Dict, List, Optional, Tuple
import sys
import os
import shutil
from dataclasses import dataclass
import importlib.util

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
    from utils.security import validate_session_security
    from core.dashboard_utils import DashboardStateManager
    
    # For main dashboard
    logger = configure_dashboard_logging()
    
except ImportError as e:
    # Fallback logging setup
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
    """
    def __init__(self):
        """Initialize the main dashboard."""
        self.start_time = time.time()
        
        # Use the global logger instead of creating a new one
        self.logger = logger
        
        # Initialize session state early for better performance
        self._initialize_session_state()
        
        self.state_manager = self._initialize_state_manager()
        
        # Cache pages configuration to avoid reloading on every run
        if 'pages_config_cache' not in st.session_state:
            st.session_state.pages_config_cache = self._load_pages_configuration()
        self.pages_config = st.session_state.pages_config_cache
    
    def _initialize_session_state(self) -> None:
        """Initialize all session state variables with defaults."""
        session_defaults = {
            'current_page': 'home',
            'page_history': ['home'],
            'load_time': time.time(),
            'dashboard_initialized': True,
            'security_validated': False,
            'last_health_check': 0,
            'navigation_count': 0        }
        
        for key, default_value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def _initialize_state_manager(self) -> Optional[object]:
        """Initialize dashboard state manager with error handling."""
        try:
            return DashboardStateManager()
        except Exception as e:
            self.logger.error(f"Failed to initialize state manager: {e}")
            return None
    
    def _discover_pages(self) -> List[str]:
        """Dynamically discover all Python files in dashboard_pages directory."""
        pages_dir = project_root / "dashboard_pages"
        if not pages_dir.exists():
            self.logger.warning(f"Dashboard pages directory not found: {pages_dir}")
            # Create the directory if it doesn't exist
            try:
                pages_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created dashboard pages directory: {pages_dir}")
            except Exception as e:
                self.logger.error(f"Failed to create dashboard pages directory: {e}")
            return []
        
        discovered_pages = []
        try:
            for file_path in pages_dir.glob("*.py"):
                if file_path.name != "__init__.py" and file_path.is_file():
                    discovered_pages.append(file_path.name)
            
            self.logger.info(f"Discovered {len(discovered_pages)} pages in dashboard_pages/")
        except Exception as e:
            self.logger.error(f"Error discovering pages: {e}")
            
        return discovered_pages
    
    def _load_pages_configuration(self) -> List[PageConfig]:
        """Load and validate page configurations with dynamic discovery."""
        try:
            # Define known pages with metadata
            known_pages = {
                "advanced_ai_trade.py": PageConfig(
                    name="Live Trading Dashboard",
                    file="advanced_ai_trade.py",
                    description="Real-time AI-powered trading with risk management",
                    category="Trading",
                    requires_auth=True
                ),                "data_dashboard.py": PageConfig(
                    name="Simple Data Dashboard",
                    file="data_dashboard.py",
                    description="Simple data download",
                    category="Data"
                ),
                "data_dashboard_v2.py": PageConfig(
                    name="Data Dashboard V2",
                    file="data_dashboard_v2.py",
                    description="Robust data download with file handling",
                    category="Data"
                ),
                "data_analysis_v2.py": PageConfig(
                    name="Technical Analysis Tools",
                    file="data_analysis_v2.py",
                    description="Advanced technical analysis and pattern detection",
                    category="Analysis"
                ),
                "model_training.py": PageConfig(
                    name="Model Training & Deployment",
                    file="model_training.py",
                    description="Train and deploy ML models for pattern recognition",
                    category="Machine Learning"
                ),
                "nn_backtest.py": PageConfig(
                    name="Neural Net Backtesting",
                    file="nn_backtest.py",
                    description="Backtest neural network trading strategies",
                    category="Backtesting"
                ),
                "classic_strategy_backtest.py": PageConfig(
                    name="Classic Strategy Backtesting",
                    file="classic_strategy_backtest.py",
                    description="Backtest traditional technical analysis strategies",
                    category="Backtesting"
                ),
                "patterns_management.py": PageConfig(
                    name="Candlestick Patterns Editor",
                    file="patterns_management.py",
                    description="Create and edit custom candlestick patterns",
                    category="Analysis"
                ),
                "model_visualizer.py": PageConfig(
                    name="Model Visualizer",
                    file="model_visualizer.py",
                    description="Visualize model performance and predictions",
                    category="Machine Learning"
                ),
                "realtime_dashboard_v3.py": PageConfig(
                    name="Real-time Dashboard v3",
                    file="realtime_dashboard_v3.py",
                    description="Advanced real-time stock analysis with AI insights",
                    category="Analysis"
                ),
                "realtime_dashboard_v2.py": PageConfig(
                    name="Real-time Dashboard v2",
                    file="realtime_dashboard_v2.py",
                    description="AI-powered technical stock analysis",
                    category="Analysis"
                ),
                "realtime_dashboard.py": PageConfig(
                    name="Real-time Dashboard (Classic)",
                    file="realtime_dashboard.py",
                    description="Classic real-time stock analysis dashboard",
                    category="Analysis"
                ),
                "simple_trade.py": PageConfig(
                    name="Simple Trading Interface",
                    file="simple_trade.py",
                    description="Simple trading interface for quick trades",
                    category="Trading"
                )
            }
            
            # Discover all available pages
            discovered_files = self._discover_pages()
            
            # Validate and create page configurations
            validated_pages = []
            pages_dir = project_root / "dashboard_pages"
            
            # Process known pages first
            for file_name in discovered_files:
                page_path = pages_dir / file_name
                
                if file_name in known_pages:
                    # Use predefined configuration
                    page_config = known_pages[file_name]
                    page_config.is_active = page_path.exists()
                    validated_pages.append(page_config)
                    
                    if page_config.is_active:
                        self.logger.debug(f"Validated known page: {page_config.name}")
                    else:
                        self.logger.warning(f"Known page file not found: {file_name}")
                else:
                    # Create generic configuration for unknown pages
                    if page_path.exists():
                        # Try to determine category from filename
                        category = self._determine_category_from_filename(file_name)
                        
                        # Create readable name from filename
                        display_name = self._create_display_name(file_name)
                        
                        generic_page = PageConfig(
                            name=display_name,
                            file=file_name,
                            description=f"Custom dashboard page: {display_name}",
                            category=category,
                            is_active=True
                        )
                        validated_pages.append(generic_page)
                        self.logger.info(f"Added unknown page: {display_name}")
            
            # Sort pages by category and name for better organization
            validated_pages.sort(key=lambda p: (p.category, p.name))
            
            self.logger.info(f"Loaded {len(validated_pages)} total pages ({len([p for p in validated_pages if p.is_active])} active)")
            return validated_pages
            
        except Exception as e:
            self.logger.error(f"Failed to load page configurations: {e}")
            return []
    
    def _determine_category_from_filename(self, filename: str) -> str:
        """Determine page category based on filename patterns."""
        filename_lower = filename.lower()
        
        if any(keyword in filename_lower for keyword in ['trade', 'trading', 'order']):
            return "Trading"
        elif any(keyword in filename_lower for keyword in ['data', 'download', 'fetch']):
            return "Data"
        elif any(keyword in filename_lower for keyword in ['analysis', 'technical', 'realtime', 'dashboard']):
            return "Analysis"
        elif any(keyword in filename_lower for keyword in ['model', 'train', 'ml', 'neural', 'nn']):
            return "Machine Learning"
        elif any(keyword in filename_lower for keyword in ['backtest', 'strategy']):
            return "Backtesting"
        elif any(keyword in filename_lower for keyword in ['pattern']):
            return "Analysis"
        else:
            return "Utilities"
    
    def _create_display_name(self, filename: str) -> str:
        """Create a human-readable display name from filename."""
        # Remove .py extension
        name = filename.replace('.py', '')
        
        # Replace underscores with spaces
        name = name.replace('_', ' ')
        
        # Capitalize each word
        name = ' '.join(word.capitalize() for word in name.split())
        
        # Handle common abbreviations
        replacements = {
            'Ai': 'AI',
            'Ml': 'ML',
            'Nn': 'Neural Network',
            'V2': 'v2',
            'V3': 'v3',
            'Api': 'API'
        }
        
        for old, new in replacements.items():
            name = name.replace(old, new)
        
        return name
    def _load_and_execute_page(self, page_file: str) -> None:
        """Load and execute a dashboard page with enhanced compatibility and caching."""
        pages_dir = project_root / "dashboard_pages"
        page_path = pages_dir / page_file
        
        if not page_path.exists():
            st.error(f"Page file not found: {page_file}")
            self.logger.error(f"Page file not found: {page_path}")
            return
          # Performance tracking
        page_start_time = time.time()
        
        try:
            # Show loading indicator for slow pages
            with st.spinner(f"Loading {page_file}..."):
                # Ensure page isolation
                self._ensure_page_isolation(page_file)
                
                # Check if page content is cached
                cache_key = f"page_content_{page_file}_{page_path.stat().st_mtime}"
                
                if cache_key not in st.session_state:
                    # Read and cache the file content
                    with open(page_path, 'r', encoding='utf-8') as f:
                        page_content = f.read()
                    
                    # Apply compatibility fixes
                    page_content = self._apply_page_fixes(page_content)
                    st.session_state[cache_key] = page_content
                    self.logger.debug(f"Cached content for {page_file}")
                else:
                    page_content = st.session_state[cache_key]
                    self.logger.debug(f"Using cached content for {page_file}")
            
            # Create enhanced execution environment
            page_globals = self._create_page_execution_environment(page_path)
            
            # Execute the page code with comprehensive error handling
            try:
                exec(page_content, page_globals)
                
                # Auto-execute main function if available and not already executed
                if 'main' in page_globals and callable(page_globals['main']):
                    execution_key = f"executed_{page_file}"
                    if execution_key not in st.session_state:
                        page_globals['main']()
                        st.session_state[execution_key] = True
                        self.logger.debug(f"Auto-executed main() for {page_file}")
                
                # Track successful page load
                load_time = time.time() - page_start_time
                self.logger.info(f"Successfully loaded {page_file} in {load_time:.2f}s")
                
                # Performance warning for slow pages
                if load_time > 2.0:
                    st.warning(f"⚠️ Page loaded slowly ({load_time:.1f}s). Consider optimization.")
            
            except SystemExit:
                # Handle st.stop() calls gracefully
                self.logger.debug(f"Page {page_file} called st.stop()")
                pass
            except Exception as exec_error:
                self._handle_page_execution_error(page_file, exec_error, page_content)
        
        except Exception as e:
            self._handle_critical_page_error(page_file, e)
    
    def _apply_page_fixes(self, page_content: str) -> str:
        """Apply compatibility fixes to page content."""
        fixes = [
            ("st._is_running_with_streamlit", "True"),
            ("if __name__ == \"__main__\" or st._is_running_with_streamlit:", "if True:"),
            ("if __name__ == '__main__' or st._is_running_with_streamlit:", "if True:"),
            # Handle the main function pattern more robustly
            ("if __name__ == \"__main__\":\n    main()", "# main() will be called automatically"),
            ("if __name__ == '__main__':\n    main()", "# main() will be called automatically"),
        ]
        
        for old, new in fixes:
            page_content = page_content.replace(old, new)
        
        return page_content
    
    def _create_page_execution_environment(self, page_path: Path) -> dict:
        """Create a safe and enhanced execution environment for pages."""
        return {
            '__name__': '__main__',
            '__file__': str(page_path),
            'st': st,
            'project_root': project_root,
            'logger': self.logger,
            # Common imports that pages might need
            'pd': None,  # Will be imported by page if needed
            'np': None,
            'plt': None,
            'go': None,
            'time': time,
            'Path': Path,
            'os': os,
            'sys': sys,
        }
    
    def _handle_page_execution_error(self, page_file: str, error: Exception, page_content: str) -> None:
        """Handle page execution errors with detailed debugging."""
        st.error(f"Error executing page: {error}")
        self.logger.error(f"Page execution error in {page_file}: {error}")
        
        # Show debug information
        with st.expander("🐛 Debug Information", expanded=False):
            st.code(str(error))
            st.text("Recent code snippet:")
            lines = page_content.split('\n')
            # Show a snippet around where error might be
            st.code('\n'.join(lines[-10:]))
            
            # Show error type and location
            st.json({
                "error_type": type(error).__name__,
                "error_message": str(error),
                "page_file": page_file,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
    
    def _handle_critical_page_error(self, page_file: str, error: Exception) -> None:
        """Handle critical page loading errors with recovery options."""
        self.logger.error(f"Critical error loading page {page_file}: {error}")
        st.error(f"Failed to load page: {page_file}")
        
        # Provide recovery options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Retry", key=f"retry_{page_file}"):
                # Clear execution state and cache
                for key in list(st.session_state.keys()):
                    if key.startswith(f"executed_{page_file}") or key.startswith(f"page_content_{page_file}"):
                        del st.session_state[key]
                st.rerun()
        
        with col2:
            if st.button("🏠 Home", key=f"home_{page_file}"):
                st.session_state.current_page = 'home'
                st.session_state.page_history = ['home']
                st.rerun()
        
        with col3:
            if st.button("🔧 Debug Mode", key=f"debug_{page_file}"):
                st.session_state[f"debug_mode_{page_file}"] = True
                st.rerun()
        
        # Show debug information if requested
        if st.session_state.get(f"debug_mode_{page_file}", False):
            with st.expander("🔍 Full Error Details", expanded=True):
                st.exception(error)
                
                # Show page info
                st.json({
                    "page_file": page_file,
                    "error_type": type(error).__name__,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "session_state_keys": list(st.session_state.keys())
                })
    
    def _render_navigation_menu(self) -> None:
        """Render categorized navigation menu with proper page navigation."""
        st.header("🧭 Navigation Menu")
        
        # Add breadcrumb navigation
        if len(st.session_state.page_history) > 1:
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("⬅️ Back", help="Go back to previous page"):
                    if len(st.session_state.page_history) > 1:
                        st.session_state.page_history.pop()  # Remove current page
                        st.session_state.current_page = st.session_state.page_history[-1]
                        # Clear execution states
                        for key in list(st.session_state.keys()):
                            if key.startswith("executed_"):
                                del st.session_state[key]
                        st.rerun()
            
            with col2:
                breadcrumb = " → ".join(st.session_state.page_history)
                st.caption(f"📍 {breadcrumb}")
        
        # Group pages by category
        categories = {}
        for page in self.pages_config:
            if page.category not in categories:
                categories[page.category] = []
            categories[page.category].append(page)
        
        # Define category order for better UX
        category_order = ["Trading", "Data", "Analysis", "Machine Learning", "Backtesting", "Utilities"]
        
        # Render categories in preferred order
        for category in category_order:
            if category in categories:
                pages = categories[category]
                with st.expander(f"📁 {category} ({len([p for p in pages if p.is_active])}/{len(pages)})", expanded=True):
                    for page in sorted(pages, key=lambda p: p.name):
                        self._render_page_button(page)
        
        # Render any remaining categories not in the predefined order
        remaining_categories = set(categories.keys()) - set(category_order)
        for category in sorted(remaining_categories):
            pages = categories[category]
            with st.expander(f"📁 {category} ({len([p for p in pages if p.is_active])}/{len(pages)})", expanded=True):
                for page in sorted(pages, key=lambda p: p.name):
                    self._render_page_button(page)
    def _render_page_button(self, page: PageConfig) -> None:
        """
        Render individual page as a button for navigation with enhanced UI.
        
        Args:
            page: Page configuration object
        """
        col1, col2 = st.columns([4, 1])
        
        with col1:
            if page.is_active:
                # Create a unique key for each button
                button_key = f"nav_{page.file}_{hash(page.name)}"
                
                # Show different styling for current page
                is_current = st.session_state.current_page == page.file
                button_text = f"{'📌' if is_current else '📄'} {page.name}"
                
                if st.button(
                    button_text,
                    key=button_key,
                    help=page.description,
                    use_container_width=True,
                    type="primary" if is_current else "secondary"
                ):
                    if not is_current:  # Only navigate if not already on this page
                        # Track navigation
                        st.session_state.navigation_count += 1
                        
                        # Clear any previous page execution state
                        for key in list(st.session_state.keys()):
                            if key.startswith("executed_"):
                                del st.session_state[key]
                        
                        # Navigate to the selected page
                        st.session_state.current_page = page.file
                        if page.file not in st.session_state.page_history:
                            st.session_state.page_history.append(page.file)
                        
                        # Limit history size to prevent memory issues
                        if len(st.session_state.page_history) > 10:
                            st.session_state.page_history = st.session_state.page_history[-10:]
                        
                        self.logger.info(f"Navigated to page: {page.name}")
                        st.rerun()
                
                # Show enhanced description with status
                if is_current:
                    st.success(f"🟢 Current: {page.description}")
                else:
                    st.caption(page.description)
            else:
                st.markdown(f"**{page.name}** ⚠️ *Unavailable*")
                st.caption(f"{page.description} (File not found)")
        
        with col2:
            if page.is_active:
                # Show page status and metadata
                status_items = []
                
                if page.requires_auth:
                    status_items.append("🔒 Auth")
                else:
                    status_items.append("🌐 Public")
                
                # Check if page has been visited
                execution_key = f"executed_{page.file}"
                if execution_key in st.session_state:
                    status_items.append("✅ Loaded")
                
                st.caption(" | ".join(status_items))
            else:
                st.caption("❌ Inactive")
    
    def _render_home_page(self) -> None:
        """Render the main dashboard home page."""
        self._render_header()
        self._render_description()
        self._render_navigation_menu()
        self._render_footer()
    
    def _load_project_description(self) -> str:
        """Load project description from README with robust error handling."""
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
        # Create header with home button
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("🏠 Home", help="Return to main dashboard"):
                st.session_state.current_page = 'home'
                st.session_state.page_history = ['home']
                # Clear execution states
                for key in list(st.session_state.keys()):
                    if key.startswith("executed_"):
                        del st.session_state[key]
                st.rerun()
        
        with col2:
            st.title("📊 StockTrader Dashboard")
        
        with col3:
            # Show current page
            if st.session_state.current_page != 'home':
                current_page_config = next(
                    (p for p in self.pages_config if p.file == st.session_state.current_page),
                    None
                )
                if current_page_config:
                    st.caption(f"📍 {current_page_config.name}")
        
        # Add status indicators
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col2:
            if self.state_manager:
                st.success("🟢 System Online")
            else:
                st.warning("🟡 Limited Mode")
        
        with col3:
            active_pages = len([p for p in self.pages_config if p.is_active])
            st.metric("Pages Available", active_pages)
    
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
                    None
                )
                if current_page_config:
                    st.info(f"**{current_page_config.name}**\n\n{current_page_config.description}")
            
            # Health checks
            health_checks = self._perform_health_checks()
            
            st.subheader("🏥 Health Status")
            for check_name, status in health_checks.items():
                if status:
                    st.success(f"✅ {check_name}")
                else:
                    st.error(f"❌ {check_name}")
              # Page statistics
            st.subheader("📊 Page Statistics")
            total_pages = len(self.pages_config)
            active_pages = len([p for p in self.pages_config if p.is_active])
            
            st.metric("Total Pages", total_pages)
            st.metric("Active Pages", active_pages)
            if total_pages > 0:
                availability_pct = (active_pages / total_pages) * 100
                st.metric("Availability", f"{availability_pct:.1f}%")
    
    def _perform_health_checks(self) -> Dict[str, bool]:
        """
        Perform comprehensive system health checks with caching.
        
        Returns:
            Dictionary of health check results
        """
        # Cache health checks for 30 seconds to improve performance
        current_time = time.time()
        cache_key = "health_checks_cache"
        cache_time_key = "health_checks_timestamp"
        
        if (cache_key in st.session_state and 
            cache_time_key in st.session_state and
            current_time - st.session_state[cache_time_key] < 30):
            return st.session_state[cache_key]
        
        checks = {}
        
        try:
            # Check critical directories
            critical_dirs = {
                "Data Directory": "data",
                "Models Directory": "models", 
                "Logs Directory": "logs",
                "Dashboard Pages": "dashboard_pages",
                "Utils Module": "utils",
                "Core Module": "core",
                "Patterns Module": "patterns"
            }
            
            for check_name, dir_name in critical_dirs.items():
                checks[check_name] = (project_root / dir_name).exists()
            
            # Check configuration files
            config_files = {
                "Environment Config": ".env",
                "Requirements": "requirements.txt",
                "Project Plan": "project_plan.md",
                "README": "readme.md"
            }
            
            for check_name, file_name in config_files.items():
                checks[check_name] = (project_root / file_name).exists()
            
            # Check page availability
            if self.pages_config:
                active_pages = len([p for p in self.pages_config if p.is_active])
                total_pages = len(self.pages_config)
                checks["Page Availability"] = active_pages >= max(1, total_pages * 0.7)  # 70% threshold
            else:
                checks["Page Availability"] = False
            
            # Check state manager
            checks["State Manager"] = self.state_manager is not None
            
            # Check logging system
            checks["Logging System"] = self.logger is not None
            
            # Check session state health
            required_session_keys = ['current_page', 'page_history', 'dashboard_initialized']
            checks["Session State"] = all(key in st.session_state for key in required_session_keys)
              # Check disk space (basic check)
            try:
                total, used, free = shutil.disk_usage(project_root)
                free_gb = free // (2**30)  # Convert to GB
                checks["Disk Space"] = free_gb > 1  # At least 1GB free
            except Exception:
                checks["Disk Space"] = True  # Assume OK if can't check
            
            # Update cache
            st.session_state[cache_key] = checks
            st.session_state[cache_time_key] = current_time
            
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
            st.markdown("• [Project Plan](./project_plan.md)")
        
        with col2:
            st.markdown("**🔧 Support**")
            st.markdown("• [GitHub Issues](https://github.com/your-repo/stocktrader/issues)")
            st.markdown("• [Configuration Guide](./docs/config.md)")
            st.markdown("• [Troubleshooting](./docs/troubleshooting.md)")
        
        with col3:
            st.markdown("**⚖️ Legal**")
            st.markdown("• [License](./LICENSE)")
            st.markdown("• [Terms of Use](./docs/terms.md)")
            st.markdown("• [Privacy Policy](./docs/privacy.md)")
        
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
    
    def _validate_dependencies(self) -> bool:
        """Validate that all critical dependencies are available."""
        critical_imports = [
            ('streamlit', 'st'),
            ('pandas', 'pd'),
            ('pathlib', 'Path'),
            ('logging', 'logging'),
        ]
        
        missing_deps = []
        for module_name, import_name in critical_imports:
            try:
                __import__(module_name)
            except ImportError:
                missing_deps.append(module_name)
        
        if missing_deps:
            self.logger.error(f"Missing critical dependencies: {missing_deps}")
            st.error(f"Missing dependencies: {', '.join(missing_deps)}")
            st.error("Please install required packages with: pip install -r requirements.txt")
            return False
        
        return True

    def run(self) -> None:
        """
        Main dashboard execution method.
        
        Orchestrates the entire dashboard rendering process with
        comprehensive error handling and performance monitoring.
        """
        start_time = time.time()
        
        try:
            # Perform security validation
            if not self._handle_security_validation():
                st.stop()
            
            # Validate critical dependencies
            if not self._validate_dependencies():
                st.stop()
            
            # Initialize session state
            if self.state_manager:
                self.state_manager.initialize_session_state()
            
            # Render system info in sidebar (always visible)
            self._render_system_info()
            
            # Handle page navigation
            if st.session_state.current_page == 'home':
                # Render main dashboard
                self._render_home_page()
            else:
                # Render selected page
                self._render_header()
                self._load_and_execute_page(st.session_state.current_page)
            
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
    
    def _ensure_page_isolation(self, page_file: str) -> None:
        """Ensure page doesn't interfere with main dashboard configuration."""
        # Clear any potential Streamlit state conflicts
        isolation_checks = [
            # Remove any cached config attempts
            '_config',
            '_is_config_set',
            'config',
        ]
        
        for check in isolation_checks:
            if hasattr(st, check):
                try:
                    # Reset config-related state if it exists
                    if check in st.__dict__:
                        delattr(st, check)
                except:
                    pass  # Ignore any errors during cleanup
        
        # Log page isolation
        self.logger.debug(f"Applied page isolation for {page_file}")


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