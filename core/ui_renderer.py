"""
UI Renderer Module

Handles all UI rendering functions for the StockTrader dashboard.
Provides clean separation of UI rendering logic from orchestration logic.
"""

import streamlit as st
import logging
from typing import List, Optional
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
    
    def _update_query_params(self, page: str) -> None:
        """Helper function to update query params across different Streamlit versions."""
        try:
            # Try new Streamlit API first
            st.query_params["page"] = page
        except AttributeError:
            try:
                # Fallback to experimental API for older versions
                st.experimental_set_query_params(page=page)
            except:
                # If neither works, just continue without query params
                self.logger.warning("Unable to update query params - Streamlit API not available")
                pass
    
    def render_home_page(self, pages_config: List, state_manager) -> None:
        """Render the main dashboard home page."""
        self.render_header(pages_config, state_manager)
        self.render_description()
        self.render_navigation_menu(pages_config)
        self.render_footer(pages_config)
    
    def render_header(self, pages_config: List, state_manager) -> None:
        """Render dashboard header with branding and status."""
        # Main header: Home button, Title, Current Page
        header_cols = st.columns([1, 3, 1.5]) # Adjusted column ratios
        
        with header_cols[0]:
            if st.button("🏠 Home", help="Return to main dashboard", use_container_width=True):
                st.session_state.current_page = 'home'
                st.session_state.page_history = ['home']
                # Clear execution states - this is a broad clear.
                # Consider if more granular clearing is needed for specific pages
                # if they store complex state under "executed_" prefixes.
                # For now, assuming "executed_" is primarily for "has run" status.
                for key in list(st.session_state.keys()):
                    if key.startswith("executed_"):
                        del st.session_state[key]
                  # Update query params
                self._update_query_params("home")
                st.rerun()
        
        with header_cols[1]:
            st.title("📊 StockTrader Dashboard")
        
        with header_cols[2]:
            if st.session_state.current_page != 'home':
                current_page_config = next(
                    (p for p in pages_config if p.file == st.session_state.current_page),
                    None
                )
                if current_page_config:
                    st.caption(f"📍 Current Page: {current_page_config.name}")
            else:
                st.caption("📍 On Home Page")

        st.markdown("---") # Visual separator

        # Metrics bar
        metric_cols = st.columns(3)
        with metric_cols[0]:
            if state_manager: # Assuming state_manager implies full functionality
                st.metric("System Status", "🟢 Online")
            else:
                st.metric("System Status", "🟡 Limited Mode")
        
        with metric_cols[1]:
            active_pages = len([p for p in pages_config if p.is_active])
            st.metric("Pages Available", active_pages)

        with metric_cols[2]:
            # Example of another potential metric - replace with actual data if available
            # For instance, if you have a health_checker object passed or accessible
            # last_health_check_status = st.session_state.get('last_health_check_status', 'N/A')
            # st.metric("Health Check", last_health_check_status)
            st.metric("Navigation Count", st.session_state.get('navigation_count', 0))
            
        st.markdown("---") # Visual separator below metrics
    
    def render_description(self) -> None:
        """Render project description section."""
        description = self._load_project_description()
        
        with st.expander("📖 About StockTrader Platform", expanded=False):
            with st.container(border=True): # Added border to the container within expander
                st.markdown(description)
                
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
                        # Clear execution states - see comment in render_header
                        for key in list(st.session_state.keys()):
                            if key.startswith("executed_"):
                                del st.session_state[key]
                          # Update query params
                        self._update_query_params(st.session_state.current_page)
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
        Render individual page as a button for navigation with enhanced UI using a card layout.
        
        Args:
            page: Page configuration object
        """
        with st.container(border=True):
            col1, col2 = st.columns([3, 1]) # Column for title/description and button

            with col1:
                is_current = st.session_state.current_page == page.file
                icon = '📌' if is_current else '📄'
                st.subheader(f"{icon} {page.name}")
                
                if page.is_active:
                    if is_current:
                        st.success(f"Current: {page.description}")
                    else:
                        st.caption(page.description)
                else:
                    st.markdown("⚠️ *Unavailable*")
                    st.caption(f"{page.description} (File not found or inactive)")

            with col2:
                if page.is_active:
                    button_key = f"nav_button_{page.file}_{hash(page.name)}"
                    button_type = "primary" if is_current else "secondary"
                    
                    if st.button(
                        "Go to Page" if not is_current else "Current Page",
                        key=button_key,
                        help=f"Navigate to {page.name}",
                        use_container_width=True,
                        type=button_type,
                        disabled=is_current 
                    ):
                        if not is_current:
                            st.session_state.navigation_count += 1
                            # Clear execution states - see comment in render_header
                            for key_to_clear in list(st.session_state.keys()):
                                if key_to_clear.startswith("executed_"):
                                    del st.session_state[key_to_clear]
                            
                            st.session_state.current_page = page.file
                            if page.file not in st.session_state.page_history:
                                st.session_state.page_history.append(page.file)
                            
                            if len(st.session_state.page_history) > 10:
                                st.session_state.page_history = st.session_state.page_history[-10:]
                            
                            self.logger.info(f"Navigated to page: {page.name}")                            # Update query params
                            self._update_query_params(page.file)
                            st.rerun()
                    
                    # Metadata below the button or alongside
                    status_items = []
                    if page.requires_auth:
                        status_items.append("🔒 Auth")
                    else:
                        status_items.append("🌐 Public")
                    
                    execution_key = f"executed_{page.file}"
                    if execution_key in st.session_state:
                        status_items.append("✅ Loaded")
                    st.caption(" | ".join(status_items))

                else: # Not active
                    st.caption("❌ Inactive")
            
            st.markdown("<br>", unsafe_allow_html=True) # Add a little space after each card

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
