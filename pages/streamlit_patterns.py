"""
Candlestick Pattern Editor

This module provides a Streamlit interface for viewing, exploring, and managing
candlestick pattern definitions in the stocktrader application.
"""

import os
import logging
import inspect
import streamlit as st
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure proper import path regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from patterns import CandlestickPatterns, PatternDetectionError

# Configure logging
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Candlestick Patterns Editor",
    layout="wide"
)

class PatternExplorer:
    """Handles pattern exploration and management functionality."""
    
    @staticmethod
    def get_pattern_names() -> List[str]:
        """
        Retrieve list of registered pattern names from CandlestickPatterns.
        
        Returns:
            List[str]: Names of all registered patterns
        """
        try:
            return CandlestickPatterns.get_pattern_names()
        except Exception as e:
            logger.error(f"Failed to get pattern names: {str(e)}")
            st.error(f"Could not retrieve pattern names: {str(e)}")
            return []
    
    @staticmethod
    def get_pattern_method(pattern_name: str) -> Optional[callable]:
        """
        Get the method corresponding to the selected pattern.
        
        Args:
            pattern_name: Name of the pattern
            
        Returns:
            Optional[callable]: Method implementing the pattern or None if not found
        """
        method_name = f"is_{pattern_name.lower().replace(' ', '_')}"
        return getattr(CandlestickPatterns, method_name, None)
    
    @staticmethod
    def get_pattern_source_and_doc(pattern_method: callable) -> Tuple[str, Optional[str]]:
        """
        Get source code and documentation for a pattern method.
        
        Args:
            pattern_method: The pattern detection method
            
        Returns:
            Tuple[str, Optional[str]]: Source code and docstring
        """
        source_code = inspect.getsource(pattern_method)
        doc = inspect.getdoc(pattern_method)
        return source_code, doc
    
    @staticmethod
    def read_patterns_file() -> Tuple[str, Optional[Exception]]:
        """
        Read the contents of patterns.py file.
        
        Returns:
            Tuple[str, Optional[Exception]]: File content and exception if occurred
        """
        patterns_path = PROJECT_ROOT / "patterns.py"
        try:
            with open(patterns_path, "r", encoding="utf-8") as f:
                return f.read(), None
        except Exception as e:
            logger.error(f"Failed to read patterns.py: {str(e)}")
            return "", e
    
    @staticmethod
    def write_patterns_file(content: str) -> Optional[Exception]:
        """
        Update the patterns.py file with new content.
        
        Args:
            content: New file content
            
        Returns:
            Optional[Exception]: Exception if error occurred during write
        """
        patterns_path = PROJECT_ROOT / "patterns.py"
        try:
            # Create backup before writing
            backup_path = patterns_path.with_suffix('.py.bak')
            with open(patterns_path, "r", encoding="utf-8") as src:
                with open(backup_path, "w", encoding="utf-8") as dst:
                    dst.write(src.read())
            
            # Write new content
            with open(patterns_path, "w", encoding="utf-8") as f:
                f.write(content)
            return None
        except Exception as e:
            logger.error(f"Failed to update patterns.py: {str(e)}")
            return e

def render_title_and_header():
    """Render the page title and header sections."""
    st.title("🔧 Candlestick Patterns Editor")
    st.markdown("""
    This tool allows you to view, explore, and modify candlestick pattern definitions 
    used throughout the Stock Trader application.
    """)

def render_current_patterns(patterns: List[str]):
    """
    Render the current patterns section.
    
    Args:
        patterns: List of pattern names
    """
    st.header("Current Patterns")
    if not patterns:
        st.warning("No patterns were detected in the system.")
        return
        
    st.write(f"Detected **{len(patterns)}** patterns:")
    for name in patterns:
        st.write(f"- {name}")

def render_pattern_explorer(patterns: List[str]):
    """
    Render the pattern explorer section.
    
    Args:
        patterns: List of pattern names
    """
    st.header("Pattern Explorer")
    
    if not patterns:
        st.warning("No patterns available to explore.")
        return
        
    selected_pattern = st.selectbox(
        "Select a pattern to explore:",
        options=patterns,
        key="pattern_selector"
    )
    
    if not selected_pattern:
        return
        
    st.subheader(f"📊 {selected_pattern} Pattern")
    
    try:
        # Get the method corresponding to the selected pattern
        pattern_method = PatternExplorer.get_pattern_method(selected_pattern)
        
        if not pattern_method:
            st.warning(f"Implementation for '{selected_pattern}' pattern not found.")
            return
            
        # Get the source code and documentation
        source_code, doc = PatternExplorer.get_pattern_source_and_doc(pattern_method)
        
        # Display the implementation
        with st.expander("🔍 Pattern Implementation (Technical)", expanded=False):
            st.code(source_code, language="python")
        
        # Display documentation or fallback info
        st.markdown("### Pattern Explanation")
        if doc:
            st.write(doc)
        else:
            st.write(f"The {selected_pattern} pattern is used to identify potential market reversals or continuations.")
            st.info("💡 Detailed explanation not available. Consider enhancing the pattern's docstring in patterns.py.")
        
        # Visual representation placeholder
        st.markdown("### Visual Example")
        st.info("Visual representation would be shown here. Consider adding a function to generate pattern visualizations.")
    
    except Exception as e:
        logger.exception(f"Error exploring pattern '{selected_pattern}'")
        st.error(f"Error retrieving pattern details: {str(e)}")

def render_export_section():
    """Render the export section for patterns.py."""
    st.header("Export patterns.py for Editing")
    
    code, error = PatternExplorer.read_patterns_file()
    if error:
        st.error(f"Could not read patterns.py: {str(error)}")
        return
        
    st.text_area(
        "Copy the current patterns.py code below and paste it into ChatGPT for updates:",
        code,
        height=350,
        key="export_code"
    )
    
    # Add download button
    st.download_button(
        label="Download patterns.py",
        data=code,
        file_name="patterns.py",
        mime="text/plain"
    )

def render_upload_section():
    """Render the upload section for updating patterns.py."""
    st.header("Upload Updated patterns.py")
    st.markdown("Upload your modified patterns.py file (it will overwrite the existing one).")
    
    uploaded = st.file_uploader(
        label="Choose Updated patterns.py",
        type="py",
        key="upload_patterns"
    )
    
    if not uploaded:
        return
        
    # Validate file content before updating
    try:
        new_code = uploaded.getvalue().decode("utf-8")
        
        # Basic validation check for Python syntax
        if not validate_python_code(new_code):
            st.error("The uploaded file contains invalid Python code. Please check and try again.")
            return
            
        # Check if it contains the expected class
        if "class CandlestickPatterns" not in new_code:
            st.warning("⚠️ The uploaded file doesn't seem to contain the CandlestickPatterns class. Are you sure this is correct?")
            if not st.button("Yes, continue anyway"):
                return
        
        # Update the file
        error = PatternExplorer.write_patterns_file(new_code)
        if error:
            st.error(f"Failed to write to patterns.py: {str(error)}")
        else:
            st.success("✅ patterns.py has been updated successfully!")
            st.info("To see the updated patterns, please refresh the page.")
            
    except UnicodeDecodeError:
        st.error("The uploaded file could not be decoded. Please ensure it's a valid text file.")
    except Exception as e:
        logger.exception("Error processing uploaded file")
        st.error(f"An error occurred: {str(e)}")

def validate_python_code(code: str) -> bool:
    """
    Perform basic validation of Python code.
    
    Args:
        code: Python code to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False

def render_sidebar_documentation():
    """Render the sidebar with pattern documentation guidance."""
    with st.sidebar.expander("How to Document Patterns", expanded=False):
        st.markdown("""
        ### How to Write Good Pattern Explanations
        
        When updating pattern implementations in `patterns.py`, add detailed docstrings to explain:
        
        1. **What the pattern looks like**
        2. **Market psychology behind the pattern**
        3. **Mathematical conditions** that define the pattern
        4. **Plain English explanation** of what it indicates
        5. **Trading implications** (bullish/bearish signals)
        
        Example format:
        ```python
        def is_doji(self, candle):
            """
            Identifies a Doji candlestick pattern.
            
            A Doji forms when the opening and closing prices are virtually equal.
            It represents market indecision where bulls and bears reached equilibrium.
            
            Mathematical condition:
            - |close - open| <= (high - low) * 0.05
            
            In plain terms: The body of the candle is very small (less than 5% of the total range),
            while the wicks can be of any length.
            
            Trading implication: Potential reversal signal, especially at support/resistance levels.
            """
            # Implementation code...
        ```
        """)
    
    st.sidebar.info("Use the sections above to explore, inspect, export, and upload your candlestick pattern definitions.")
    
    # Add extra utilities in the sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Utilities")
    
    if st.sidebar.button("Reload Patterns"):
        st.experimental_rerun()

def main():
    """Main entry point for the Streamlit application."""
    try:
        render_title_and_header()
        
        # Get pattern names
        pattern_names = PatternExplorer.get_pattern_names()
        
        # Render the different sections
        render_current_patterns(pattern_names)
        render_pattern_explorer(pattern_names)
        render_export_section()
        render_upload_section()
        render_sidebar_documentation()
        
    except Exception as e:
        logger.exception("Unhandled exception in Streamlit Patterns Editor")
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please check the application logs for more details.")

if __name__ == "__main__":
    main()
