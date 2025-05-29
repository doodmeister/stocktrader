"""
StockTrader Dashboard - Legacy Entry Point

This file has been refactored and modularized for better maintainability.
Please use main.py as the new entry point for the dashboard.

The dashboard has been split into focused modules:
- main.py: Entry point with Streamlit configuration
- core/dashboard_controller.py: Main orchestration logic
- core/page_loader.py: Page discovery and management
- core/health_checks.py: System health monitoring

Legacy entry point - redirects to new modular structure.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def main():
    """Legacy entry point that redirects to the new modular dashboard."""
    
    # Basic page configuration
    st.set_page_config(
        page_title="StockTrader - Legacy Entry",
        page_icon="📊", 
        layout="wide"
    )
    
    # Show migration notice
    st.title("🔄 Dashboard Modernization Notice")
    
    st.info("""
    **The StockTrader dashboard has been modernized and modularized!**
    
    This file (`streamlit_dashboard.py`) is now a legacy entry point.
    """)
    
    st.markdown("""
    ### 🚀 New Modular Structure
    
    The dashboard has been refactored into focused modules for better maintainability:
    
    - **`main.py`** - New entry point with Streamlit configuration
    - **`core/dashboard_controller.py`** - Main orchestration and UI logic  
    - **`core/page_loader.py`** - Page discovery and management
    - **`core/health_checks.py`** - System health monitoring
    
    ### ✨ Benefits
    
    - **Better Organization**: Code is now split into logical modules
    - **Easier Maintenance**: Each module has a single responsibility
    - **Improved Testing**: Smaller, focused modules are easier to test
    - **Better Performance**: Optimized loading and caching
    """)
    
    # Redirect options
    st.markdown("### 🎯 How to Access the Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Method 1: Use the new entry point**
        ```bash
        streamlit run main.py
        ```
        """)
    
    with col2:
        st.markdown("""
        **Method 2: Automatic redirect**
        
        Click the button below to automatically launch the new dashboard:
        """)
        
        if st.button("🚀 Launch New Dashboard", type="primary"):
            st.info("Starting the new modular dashboard...")
            
            # Import and run the new dashboard
            try:
                from main import main as new_main
                new_main()
            except Exception as e:
                st.error(f"Error launching new dashboard: {e}")
                st.markdown("""
                **Manual startup required:**
                
                Please run: `streamlit run main.py`
                """)
    
    # Technical details in expander
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        ### File Changes
        
        **Original Structure (1000+ lines):**
        ```
        streamlit_dashboard.py  # All functionality in one file
        ```
        
        **New Modular Structure:**
        ```
        main.py                           # Entry point (50 lines)
        ├── core/dashboard_controller.py  # Main logic (465 lines)
        ├── core/page_loader.py          # Page management (350 lines)  
        └── core/health_checks.py        # Health monitoring (319 lines)
        ```
        
        ### Migration Benefits
        - **Separation of Concerns**: Each module has a focused responsibility
        - **Reusability**: Modules can be imported and used independently
        - **Maintainability**: Smaller files are easier to understand and modify
        - **Testing**: Individual components can be unit tested
        - **Performance**: Better caching and optimized loading
        """)
    
    st.markdown("---")
    st.caption("💡 This legacy file will be removed in a future version. Please update your shortcuts and scripts to use `main.py`.")

if __name__ == "__main__":
    main()