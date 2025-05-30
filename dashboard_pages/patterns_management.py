"""
Candlestick Patterns Editor - Pure UI layer delegating to existing business logic.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List
import streamlit as st
import pandas as pd

# Import existing business logic (no duplication)
from core.dashboard_utils import (
    initialize_dashboard_session_state,
    create_candlestick_chart,
    validate_ohlc_dataframe,
    safe_file_write,
    normalize_dataframe_columns,
    render_file_upload_section,
    show_success_with_actions,
    handle_streamlit_error,
    setup_page
)

from patterns.pattern_utils import (
    read_patterns_file,
    get_pattern_names, 
        get_pattern_method,
    validate_python_code
    # Removed PatternBackupManager - not available
)
from patterns.patterns import CandlestickPatterns, create_pattern_detector  # Use existing detection
from utils.data_validator import DataValidator

# Import SessionManager to solve button key conflicts and session state issues
from core.session_manager import create_session_manager, show_session_debug_info

# Dashboard logger setup
from utils.logger import get_dashboard_logger
logger = get_dashboard_logger(__name__)

# Initialize the page (setup_page returns a logger, but we already have one)
setup_page(
    title="🕯️ Candlestick Patterns Editor",
    logger_name=__name__,
    sidebar_title="Pattern Tools"
)

# UI Configuration only
MAX_FILE_SIZE_MB = 10
DEFAULT_CHART_HEIGHT = 500
PATTERN_CHART_CONFIG = {
    "displayModeBar": False,
    "staticPlot": False,
    "responsive": True
}

@dataclass
class PatternExample:
    """Data class for pattern examples with metadata."""
    name: str
    data: pd.DataFrame
    description: str
    signal_type: str
    expected_index: int = -1

class PatternEditorUI:
    """Pure UI controller - delegates to existing business logic."""
    
    def __init__(self):
        # Use existing classes, no custom implementations
        self.data_validator = DataValidator()
        # Remove backup_manager since PatternBackupManager doesn't exist
        # self.backup_manager = PatternBackupManager()

    def _analyze_single_pattern(self, pattern_name: str, df: pd.DataFrame):
        """Analyze single pattern using existing business logic."""
        try:
            # Use existing validation from utils
            is_valid, validation_msg = validate_ohlc_dataframe(df)
            if not is_valid:
                st.error(f"❌ {validation_msg}")
                return
            
            # Normalize using existing utility
            df_normalized = normalize_dataframe_columns(df)
            
            # Use existing pattern detection from patterns.py
            detections = []
            method = get_pattern_method(pattern_name)
            
            if method is None:
                st.error(f"❌ Pattern method not found: {pattern_name}")
                return
            
            # Simple detection using existing method
            for i in range(2, len(df_normalized)):  # Most patterns need 2-3 candles
                window = df_normalized.iloc[max(0, i-2):i+1]
                try:
                    if method(window):
                        detections.append((i, 1.0))  # Simple boolean result
                except Exception as e:
                    logger.debug(f"Pattern detection error at index {i}: {e}")
                    continue
            
            # Use existing chart creation
            fig = create_candlestick_chart(
                df=df_normalized,
                title=f"{pattern_name} Pattern Analysis",
                detections=detections,
                pattern_name=pattern_name,
                height=DEFAULT_CHART_HEIGHT
            )
            
            st.plotly_chart(fig, use_container_width=True, config=PATTERN_CHART_CONFIG)
            
            # Show results
            if detections:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Detections", len(detections))
                with col2:
                    st.metric("Pattern Method", method.__name__)
                with col3:
                    st.metric("Detection Rate", f"{len(detections)/len(df):.1%}")
            else:
                st.info("No pattern detections found in the provided data.")
                
        except Exception as e:
            handle_streamlit_error(e, "pattern analysis")

    def _analyze_multiple_patterns(self, pattern_names: List[str], df: pd.DataFrame):
        """Analyze multiple patterns using existing CandlestickPatterns.detect_patterns."""
        try:
            # Use existing validation
            is_valid, validation_msg = validate_ohlc_dataframe(df)
            if not is_valid:
                st.error(f"❌ {validation_msg}")
                return
              # Normalize data
            df_normalized = normalize_dataframe_columns(df)
            # Use existing detect_patterns method from CandlestickPatterns
            all_detections = {}
            
            for i in range(len(df_normalized)):
                # Get a reasonable window for pattern detection
                window_start = max(0, i - 2)
                window = df_normalized.iloc[window_start:i+1]
                
                if len(window) >= 1:  # Minimum window size
                    try:
                        # Use existing detection method
                        pattern_detector = create_pattern_detector()
                        detected_results = pattern_detector.detect_patterns(window)
                        # Extract pattern names from PatternResult objects
                        detected_patterns = [result.name for result in detected_results if result.detected]
                        
                        # Filter to only requested patterns
                        relevant_patterns = [p for p in detected_patterns if p in pattern_names]
                        
                        for pattern in relevant_patterns:
                            if pattern not in all_detections:
                                all_detections[pattern] = []
                            all_detections[pattern].append((i, 1.0))
                            
                    except Exception as e:
                        logger.debug(f"Multi-pattern detection error at index {i}: {e}")
                        continue
            
            # Display results
            st.subheader("Pattern Comparison Results")
            
            # Summary metrics
            total_detections = sum(len(detections) for detections in all_detections.values())
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Patterns Analyzed", len(pattern_names))
            with col2:
                st.metric("Patterns Found", len(all_detections))
            with col3:
                st.metric("Total Detections", total_detections)
            
            # Individual pattern charts
            for pattern_name in pattern_names:
                if pattern_name in all_detections:
                    with st.expander(f"📊 {pattern_name} Results ({len(all_detections[pattern_name])} detections)"):
                        fig = create_candlestick_chart(
                            df=df_normalized,
                            title=f"{pattern_name} Detections",
                            detections=all_detections[pattern_name],
                            pattern_name=pattern_name,
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    with st.expander(f"📊 {pattern_name} Results (0 detections)"):
                        st.info(f"No {pattern_name} patterns detected in this data.")
                        
        except Exception as e:
            handle_streamlit_error(e, "multiple pattern analysis")

    def _render_data_upload(self):
        """Render data upload using existing utilities."""
        uploaded_file = render_file_upload_section(
            title="📊 Upload Market Data for Pattern Analysis",
            file_types=["csv"],
            max_size_mb=MAX_FILE_SIZE_MB,
            help_text="Upload CSV with OHLC data for pattern analysis"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Use existing validation
                is_valid, validation_msg = validate_ohlc_dataframe(df)
                if not is_valid:
                    st.error(f"❌ {validation_msg}")
                    return None
                
                # Use existing normalization
                df = normalize_dataframe_columns(df)
                
                # Show preview
                st.markdown("**Data Preview:**")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Data statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    price_range = df['high'].max() - df['low'].min()
                    st.metric("Price Range", f"${price_range:.2f}")
                with col4:
                    volatility = ((df['high'] - df['low']) / df['close']).mean()
                    st.metric("Avg Volatility", f"{volatility:.1%}")
                
                return df
                
            except Exception as e:
                handle_streamlit_error(e, "data upload")
                return None
        
        return None

    def _save_pattern_changes(self, code: str):
        """Save pattern changes using existing utilities."""
        try:
            # Use existing validation
            is_valid, validation_msg = validate_python_code(code)
            if not is_valid:
                st.error(f"❌ Invalid syntax: {validation_msg}")
                return
            
            # Use existing safe file write
            patterns_path = Path("patterns/patterns.py")
            success, message, backup_path = safe_file_write(
                patterns_path, code, create_backup=True
            )
            
            if success:
                show_success_with_actions(
                    f"✅ {message}", 
                    {"🔄 Refresh Page": lambda: st.rerun()}, 
                    show_balloons=True
                )
                st.cache_data.clear()
                logger.info("Patterns saved successfully")
            else:
                st.error(f"❌ {message}")
                
        except Exception as e:
            handle_streamlit_error(e, "pattern save")

# Rest of UI methods remain focused on presentation...

def main():
    """Main entry point - pure UI coordination."""
    initialize_dashboard_session_state()
    
    # Initialize SessionManager to prevent button conflicts and session state issues
    session_manager = create_session_manager("patterns_management")
    
    st.title("🕯️ Candlestick Patterns Editor")
    
    editor = PatternEditorUI()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🔍 Explorer", "✏️ Editor"])
    
    with tab1:
        st.header("📊 Pattern Analysis")
        # Data upload
        df = editor._render_data_upload()
        
        if df is not None:
            # Get available patterns from existing system
            available_patterns = get_pattern_names()
            if available_patterns:
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.subheader("Single Pattern Analysis")
                    selected_pattern = session_manager.create_selectbox(
                        "Select Pattern", 
                        options=available_patterns,
                        selectbox_name="single_pattern_select"
                    )
                    if session_manager.create_button("🔍 Analyze", "analyze_single_pattern"):
                        editor._analyze_single_pattern(selected_pattern, df)
                
                with col2:
                    st.subheader("Multi-Pattern Comparison")
                    selected_patterns = session_manager.create_multiselect(
                        "Select Patterns", 
                        options=available_patterns,
                        multiselect_name="multi_pattern_select"
                    )
                    if session_manager.create_button("📈 Compare", "compare_patterns") and selected_patterns:
                        editor._analyze_multiple_patterns(selected_patterns, df)
            else:
                st.warning("⚠️ No patterns available. Check patterns file.")
        else:
            st.info("👆 Upload CSV data to begin pattern analysis.")
    
    with tab2:
        st.header("🔍 Pattern Explorer")
        # Pattern documentation and information viewer
        available_patterns = get_pattern_names()
        if available_patterns:
            selected_pattern_info = session_manager.create_selectbox(
                "Select Pattern to Explore", 
                options=available_patterns,
                selectbox_name="pattern_info_select"
            )
            
            if selected_pattern_info:
                # Get pattern method and documentation
                pattern_method = get_pattern_method(selected_pattern_info)
                if pattern_method:
                    st.subheader(f"📋 {selected_pattern_info} Pattern Details")
                    
                    # Show pattern documentation
                    doc = pattern_method.__doc__ if pattern_method.__doc__ else "No documentation available."
                    st.markdown(f"**Description:** {doc}")
                    
                    # Show pattern source code
                    with st.expander("🔍 View Source Code", expanded=False):
                        try:
                            import inspect
                            source = inspect.getsource(pattern_method)
                            st.code(source, language="python")
                        except Exception as e:
                            st.error(f"Could not retrieve source code: {e}")
                    
                    # Pattern characteristics
                    try:
                        pattern_detector = create_pattern_detector()
                        pattern_obj = pattern_detector._patterns.get(selected_pattern_info)
                        if pattern_obj:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Min Rows Required", pattern_obj.min_rows)
                            with col2:
                                st.metric("Pattern Type", pattern_obj.pattern_type.value)
                            with col3:
                                st.metric("Strength", pattern_obj.strength.value)
                    except Exception as e:
                        st.warning(f"Could not load pattern details: {e}")
                else:
                    st.error(f"Pattern method for '{selected_pattern_info}' not found.")
        else:
            st.warning("⚠️ No patterns available to explore.")
        
    with tab3:
        st.header("✏️ Code Editor")
        st.markdown("Edit the patterns.py file directly")
        
        # Read current patterns file
        try:
            patterns_content = read_patterns_file()
            if patterns_content:
                # Code editor
                edited_code = st.text_area(
                    "Edit Patterns Code",
                    value=patterns_content,
                    height=400,
                    help="Edit the patterns.py file. Be careful with syntax!"
                )
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if session_manager.create_button("💾 Save Changes", "save_pattern_code"):
                        editor._save_pattern_changes(edited_code)
                
                with col2:
                    if session_manager.create_button("✅ Validate Syntax", "validate_syntax"):
                        is_valid, msg = validate_python_code(edited_code)
                        if is_valid:
                            st.success("✅ Code syntax is valid!")
                        else:
                            st.error(f"❌ Syntax error: {msg}")
                
                with col3:
                    if session_manager.create_button("🔄 Reset to Original", "reset_code"):
                        st.rerun()
                
                # Show code statistics
                lines = len(edited_code.split('\n'))
                words = len(edited_code.split())
                st.caption(f"📊 Stats: {lines} lines, {words} words")
                
            else:
                st.error("❌ Could not read patterns file.")
                
        except Exception as e:
            st.error(f"❌ Error loading patterns file: {e}")
            
        # Additional tools
        with st.expander("🛠️ Additional Tools", expanded=False):
            st.markdown("**Pattern Management Tools:**")
            if session_manager.create_button("📂 Backup Patterns", "backup_patterns"):
                try:
                    backup_path = Path("patterns/patterns_backup.py")
                    current_content = read_patterns_file()
                    if current_content:
                        backup_path.write_text(current_content)
                        st.success(f"✅ Backup created: {backup_path}")
                    else:
                        st.error("❌ Could not create backup - no content to backup")
                except Exception as e:
                    st.error(f"❌ Backup failed: {e}")
            
            if session_manager.create_button("🔍 Pattern Count", "count_patterns"):
                try:
                    pattern_count = len(get_pattern_names())
                    st.info(f"📊 Total patterns available: {pattern_count}")
                except Exception as e:
                    st.error(f"❌ Could not count patterns: {e}")

class PatternsManagementDashboard:
    def __init__(self):
        # Initialize SessionManager for conflict-free button handling
        self.session_manager = create_session_manager("patterns_management")
    
    def run(self):
        """Main dashboard application entry point."""
        main()
        
        # Add session debug info to the sidebar
        with st.sidebar.expander("🔧 Session Debug Info", expanded=False):
            show_session_debug_info()

# Execute the main function
if __name__ == "__main__":
    try:
        dashboard = PatternsManagementDashboard()
        dashboard.run()
    except Exception as e:
        handle_streamlit_error(e, "Patterns Management Dashboard")

