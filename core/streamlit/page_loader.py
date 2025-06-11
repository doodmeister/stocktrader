"""
Page Loader Module

Handles page discovery, configuration management, and execution.
"""

import streamlit as st
import time
import logging
from pathlib import Path
from typing import List
from dataclasses import dataclass


@dataclass
class PageConfig:
    """Configuration for dashboard pages."""
    name: str
    file: str
    description: str
    category: str
    requires_auth: bool = False
    is_active: bool = True


class PageLoader:
    """
    Handles page discovery, configuration, and execution.
    """
    
    def __init__(self, logger: logging.Logger):
        """Initialize the page loader."""
        self.logger = logger
        self.project_root = Path(__file__).parent.parent.parent
    
    def discover_pages(self) -> List[str]:
        """Dynamically discover all Python files in dashboard_pages directory."""
        pages_dir = self.project_root / "dashboard_pages"
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
    
    def determine_category_from_filename(self, filename: str) -> str:
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
    
    def create_display_name(self, filename: str) -> str:
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
    
    def load_pages_configuration(self) -> List[PageConfig]:
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
                ),
                "data_dashboard.py": PageConfig(
                    name="Simple Data Dashboard",
                    file="data_dashboard.py",
                    description="Simple data download",
                    category="Data"
                ),
                "data_analysis.py": PageConfig(
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
                    category="Analysis"                ),
                "model_visualizer.py": PageConfig(
                    name="Model Visualizer",
                    file="model_visualizer.py",
                    description="Visualize model performance and predictions",
                    category="Machine Learning"
                ),
                "realtime_dashboard.py": PageConfig(
                    name="Real-time Dashboard",
                    file="realtime_dashboard.py",
                    description="Advanced real-time stock analysis with AI insights",
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
            discovered_files = self.discover_pages()
            
            # Validate and create page configurations
            validated_pages = []
            pages_dir = self.project_root / "dashboard_pages"
            
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
                        category = self.determine_category_from_filename(file_name)
                        
                        # Create readable name from filename
                        display_name = self.create_display_name(file_name)
                        
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
    
    def apply_page_fixes(self, page_content: str) -> str:
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
    
    def create_page_execution_environment(self, page_path: Path) -> dict:
        """Create a safe and enhanced execution environment for pages."""
        return {
            '__name__': '__main__',
            '__file__': str(page_path),
            'st': st,
            'project_root': self.project_root,
            'logger': self.logger,
            # Common imports that pages might need
            'pd': None,  # Will be imported by page if needed
            'np': None,
            'plt': None,
            'go': None,
            'time': time,
            'Path': Path,
            'os': __import__('os'),
            'sys': __import__('sys'),
        }
    
    def handle_page_execution_error(self, page_file: str, error: Exception, page_content: str) -> None:
        """Handle page execution errors with detailed debugging."""
        self.logger.error(f"Page execution error in {page_file}: {error}")
        
        # Show user-friendly error message
        st.error(f"❌ Error loading {page_file}")
        
        with st.expander("🔍 Error Details", expanded=False):
            st.error(f"**Error Type:** {type(error).__name__}")
            st.error(f"**Error Message:** {str(error)}")
            
            # Show potential fixes based on error type
            if "ModuleNotFoundError" in str(type(error)):
                st.info("💡 **Possible Fix:** Install missing dependencies or check import paths")
            elif "AttributeError" in str(type(error)):
                st.info("💡 **Possible Fix:** Check for API changes or missing attributes")
            elif "NameError" in str(type(error)):
                st.info("💡 **Possible Fix:** Check variable names and scope")
            
            # Recovery options
            st.subheader("🔧 Recovery Options")
            if st.button(f"🔄 Retry {page_file}", key=f"retry_{page_file}"):
                # Clear page cache and retry
                cache_key = f"page_content_{page_file}"
                if cache_key in st.session_state:
                    del st.session_state[cache_key]
                st.rerun()
    
    def handle_critical_page_error(self, page_file: str, error: Exception) -> None:
        """Handle critical page loading errors with recovery options."""
        self.logger.error(f"Critical error loading {page_file}: {error}")
        
        st.error(f"❌ Critical Error: Unable to load {page_file}")
        st.error(f"**Error:** {str(error)}")
        
        # Provide navigation back to home
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏠 Return Home", key=f"home_{page_file}"):
                st.session_state.current_page = 'home'
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh Dashboard", key=f"refresh_{page_file}"):
                # Clear all cache and restart
                for key in list(st.session_state.keys()):
                    if isinstance(key, str) and key.startswith("page_"):
                        del st.session_state[key]
                st.rerun()
    
    def ensure_page_isolation(self, page_file: str) -> None:
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
                except Exception: # Changed bare except
                    pass  # Ignore any errors during cleanup
        
        # Log page isolation
        self.logger.debug(f"Applied page isolation for {page_file}")
    
    def load_and_execute_page(self, page_file: str) -> None:
        """Load and execute a dashboard page with enhanced compatibility and caching."""
        pages_dir = self.project_root / "dashboard_pages"
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
                self.ensure_page_isolation(page_file)
                
                # Check if page content is cached
                cache_key = f"page_content_{page_file}_{page_path.stat().st_mtime}"
                
                if cache_key not in st.session_state:
                    # Read and cache the file content
                    with open(page_path, 'r', encoding='utf-8') as f:
                        page_content = f.read()
                    
                    # Apply compatibility fixes
                    page_content = self.apply_page_fixes(page_content)
                    st.session_state[cache_key] = page_content
                    self.logger.debug(f"Cached content for {page_file}")
                else:
                    page_content = st.session_state[cache_key]
                    self.logger.debug(f"Using cached content for {page_file}")
            
            # Create enhanced execution environment
            page_globals = self.create_page_execution_environment(page_path)
            
            # Execute the page code with comprehensive error handling
            try:
                exec(page_content, page_globals)
                  # Auto-execute main function if available 
                # Always execute main() to ensure pages respond to user interactions
                if 'main' in page_globals and callable(page_globals['main']):
                    page_globals['main']()
                    self.logger.debug(f"Executed main() for {page_file}")
                
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
                self.handle_page_execution_error(page_file, exec_error, page_content)
        
        except Exception as e:
            self.handle_critical_page_error(page_file, e)
