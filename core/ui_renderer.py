"""
UI Renderer Module

Handles all UI rendering functions for the StockTrader dashboard.
Provides clean separation of UI rendering logic from orchestration logic.
"""

import streamlit as st
import time
import logging
from typing import List, Dict, Optional
from pathlib import Path

from utils.logger import get_dashboard_logger


class UIRenderer:
    """
    UI Renderer class for StockTrader dashboard.
    Handles all UI rendering functions with clean separation of concerns.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the UI renderer."""
        self.logger = logger or get_dashboard_logger(__name__)
    
    def render_home_page(self, pages_config: List, state_manager) -> None:
        """Render the main dashboard home page."""
        self.render_header(pages_config, state_manager)
        self.render_description()
        self.render_navigation_menu(pages_config)
        self.render_footer(pages_config)
    
    def render_header(self, pages_config: List, state_manager) -> None:
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
                    (p for p in pages_config if p.file == st.session_state.current_page),
                    None
                )
                if current_page_config:
                    st.caption(f"📍 {current_page_config.name}")
        
        # Add status indicators
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col2:
            if state_manager:
                st.success("🟢 System Online")
            else:
                st.warning("🟡 Limited Mode")
        
        with col3:
            active_pages = len([p for p in pages_config if p.is_active])
            st.metric("Pages Available", active_pages)
    
    def render_description(self) -> None:
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
    
    def render_navigation_menu(self, pages_config: List) -> None:
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
        for page in pages_config:
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
                        self.render_page_button(page)
        
        # Render any remaining categories not in the predefined order
        remaining_categories = set(categories.keys()) - set(category_order)
        for category in sorted(remaining_categories):
            pages = categories[category]
            with st.expander(f"📁 {category} ({len([p for p in pages if p.is_active])}/{len(pages)})", expanded=True):
                for page in sorted(pages, key=lambda p: p.name):
                    self.render_page_button(page)
    
    def render_page_button(self, page) -> None:
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
    
    def render_footer(self, pages_config: List) -> None:
        """Render dashboard footer with additional information."""
        st.markdown("---")
        
        # Footer navigation and links
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🔗 Quick Links")
            st.markdown("• [Documentation](./docs/)")
            st.markdown("• [Project Plan](./project_plan.md)")
            st.markdown("• [Logs](./logs/)")
        
        with col2:
            st.subheader("📊 Statistics")
            total_pages = len(pages_config)
            active_pages = len([p for p in pages_config if p.is_active])
            st.metric("Pages", f"{active_pages}/{total_pages}")
            
            nav_count = st.session_state.get('navigation_count', 0)
            st.metric("Navigations", nav_count)
        
        with col3:
            st.subheader("ℹ️ Legal")
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
            return "StockTrader Platform - Trading tools and analytics"


# Convenience functions for backward compatibility
def render_home_page(pages_config: List, state_manager) -> None:
    """Convenience function for rendering home page."""
    renderer = UIRenderer()
    renderer.render_home_page(pages_config, state_manager)


def render_header(pages_config: List, state_manager) -> None:
    """Convenience function for rendering header."""
    renderer = UIRenderer()
    renderer.render_header(pages_config, state_manager)


def render_description() -> None:
    """Convenience function for rendering description."""
    renderer = UIRenderer()
    renderer.render_description()


def render_navigation_menu(pages_config: List) -> None:
    """Convenience function for rendering navigation menu."""
    renderer = UIRenderer()
    renderer.render_navigation_menu(pages_config)


def render_page_button(page) -> None:
    """Convenience function for rendering page button."""
    renderer = UIRenderer()
    renderer.render_page_button(page)


def render_footer(pages_config: List) -> None:
    """Convenience function for rendering footer."""
    renderer = UIRenderer()
    renderer.render_footer(pages_config)


def load_project_description() -> str:
    """Convenience function for loading project description."""
    renderer = UIRenderer()
    return renderer._load_project_description()
