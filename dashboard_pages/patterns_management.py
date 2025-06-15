"""
Patterns Management - Focused interface for pattern viewing, description, download, and addition.
Core functionalities:
1. View existing patterns in patterns.py
2. Provide descriptions of patterns
3. Allow downloading of patterns data
4. Add new patterns to the system
"""

import json
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import streamlit as st
import pandas as pd
from enum import Enum

# Import existing business logic
from core.streamlit.dashboard_utils import (
    setup_page
)
from core.streamlit.session_manager import SessionManager # Add this import

from patterns.pattern_utils import (
    read_patterns_file,
    get_pattern_names,
    validate_python_code,
)
from utils.file_io_utils import safe_file_write # Updated import

# Import CandlestickPatterns and its enums if not already
from patterns.patterns import CandlestickPatterns, PatternType, PatternStrength, create_pattern_detector # Add Enums if needed for type checking

# Dashboard logger setup
from utils.logger import get_dashboard_logger
logger = get_dashboard_logger(__name__)

# Defensive cleanup for legacy/generic keys (until all keys are managed)
LEGACY_KEYS = [
    'select', 'input', 'button', 'download_json', 'download_csv', 'save_source',
    'patterns_viewer_main_select', 'catalog_type_filter', 'catalog_strength_filter',
    'catalog_rows_filter', 'pattern_template_select', 'new_pattern_code',
    'validate_new_pattern', 'save_new_pattern', 'reset_new_pattern'
]
for key in LEGACY_KEYS:
    if key in st.session_state:
        del st.session_state[key]

def initialize_patterns_page():
    """Initialize the Streamlit page - call this when running in Streamlit context."""
    setup_page(
        title="🎯 Pattern Management",
        logger_name=__name__,
        sidebar_title="Pattern Tools"
    )


class PatternsManager:
    """Core business logic for pattern management."""

    def __init__(self):
        self.patterns_file_path = Path("patterns/patterns.py")

    @st.cache_data  # Cache the results of this method
    def get_all_patterns(_self) -> List[Dict[str, Any]]:  # Changed self to _self
        """Get all patterns with their metadata."""
        patterns_data = []  # Renamed for clarity
        try:
            # Create an instance of the main pattern engine
            # create_pattern_detector() is from patterns.patterns
            # It likely returns a CandlestickPatterns instance
            pattern_engine = create_pattern_detector() 
            pattern_names = pattern_engine.get_pattern_names()

            for name in pattern_names:
                # Get the actual PatternDetector instance
                # Assuming CandlestickPatterns has a way to get a detector by name
                # This might be: pattern_engine._patterns.get(name)
                # Or a public method: pattern_engine.get_detector(name)
                
                # Option 1: If CandlestickPatterns has get_detector(name)
                pattern_instance = pattern_engine.get_detector_by_name(name)

                # Option 2: Accessing the internal _patterns dictionary (less ideal but might be necessary)
                # This requires _patterns to be accessible and map names to detector instances
                # pattern_instance = pattern_engine._patterns.get(name)

                if pattern_instance:
                    # Description from the detect method's docstring
                    description = inspect.getdoc(pattern_instance.detect) or 'No description available'
                    
                    source_code = 'Source code not available'
                    try:
                        # Show the full class source code, not just the detect method
                        source_code = inspect.getsource(type(pattern_instance))
                    except Exception:
                        pass # Keep 'Source code not available'

                    # Get pattern_type, convert to value if enum
                    ptype_attr = getattr(pattern_instance, 'pattern_type', 'Unknown')
                    if isinstance(ptype_attr, Enum):
                        pattern_type_value = ptype_attr.value
                    else:
                        pattern_type_value = str(ptype_attr)

                    # Get strength, convert to value if enum
                    strength_attr = getattr(pattern_instance, 'strength', 'Unknown')
                    if isinstance(strength_attr, Enum): # Check if it's an Enum instance
                        if hasattr(strength_attr, 'value'): # e.g., IntEnum
                            strength_display_value = strength_attr.value
                        else: # e.g., basic Enum, use name
                            strength_display_value = strength_attr.name 
                    else: # Fallback to string if not an Enum
                        strength_display_value = str(strength_attr)
                    
                    # Get min_rows
                    min_rows_value = getattr(pattern_instance, 'min_rows', 'Unknown')

                    patterns_data.append({
                        'name': pattern_instance.name, # Use name from instance
                        'description': description,
                        'source_code': source_code,
                        'pattern_type': pattern_type_value,
                        'strength': strength_display_value,
                        'min_rows': min_rows_value
                    })
                else:
                    # Fallback if pattern method/instance isn't found
                    patterns_data.append({
                        'name': name,
                        'description': f'Details for {name} not found via instance.',
                        'source_code': '', 'pattern_type': 'Unknown', 'strength': 'Unknown', 'min_rows': 'Unknown'
                    })
        except Exception as e:
            logger.error(f"Error in get_all_patterns: {e}", exc_info=True) # Add exc_info for more details
        return patterns_data

    @st.cache_data
    def export_patterns_data(_self, format_type: str = 'json') -> Optional[str]:  # Changed self to _self
        """Export patterns data in specified format."""
        try:
            patterns = _self.get_all_patterns()  # Call with _self

            if format_type == 'json':
                return json.dumps(patterns, indent=2, default=str)
            elif format_type == 'csv':
                # Create a simplified version for CSV
                simplified = []
                for p in patterns:
                    simplified.append({
                        'name': p['name'],
                        'description': p['description'].replace('\n', ' ')[:100] + '...',
                        'pattern_type': p['pattern_type'],
                        'strength': p['strength'],
                        'min_rows': p['min_rows']
                    })
                df = pd.DataFrame(simplified)
                return df.to_csv(index=False)

        except Exception as e:
            logger.error(f"Error exporting patterns: {e}")
            return None

    @st.cache_data
    def get_patterns_source_code(_self) -> Optional[str]:  # Changed self to _self
        """Get the complete patterns.py source code."""
        try:
            return read_patterns_file()
        except Exception as e:
            logger.error(f"Error reading patterns file: {e}")
            return None

    # save_new_pattern should not be cached as it performs a write operation
    def save_new_pattern(self, pattern_code: str) -> tuple[bool, str]:
        """Save new pattern to patterns.py file."""
        try:
            # Validate the code
            is_valid = validate_python_code(pattern_code)
            if not is_valid:
                return False, "Invalid Python syntax in the pattern code"

            # Get current content
            current_content = read_patterns_file()
            if not current_content:
                return False, "Could not read current patterns file"

            # Insert new pattern above the marker
            marker = '# End of Pattern Implementations'
            lines = current_content.split('\n')
            try:
                marker_idx = next(i for i, line in enumerate(lines) if marker in line)
            except StopIteration:
                return False, f"Marker '{marker}' not found in patterns.py"

            # Insert pattern above the marker
            new_content = '\n'.join(lines[:marker_idx]) + f"\n\n{pattern_code}\n" + '\n'.join(lines[marker_idx:])

            # Save with backup
            success, message, backup_path = safe_file_write(
                self.patterns_file_path,
                new_content,
                create_backup=True
            )

            if success:
                # Clear cache
                if hasattr(st, 'cache_data'):
                    st.cache_data.clear()
                return True, f"Pattern saved successfully. Backup created at: {backup_path}"
            else:
                return False, message

        except Exception as e:
            logger.error(f"Error saving pattern: {e}")
            return False, f"Error saving pattern: {str(e)}"


class PatternsManagementUI:
    """UI controller for patterns management."""

    def __init__(self, page_name: str = "patterns_management", tab: Optional[str] = None):
        self.manager = PatternsManager()
        self.session_manager = SessionManager(namespace_prefix=page_name, tab=tab)  # Pass tab context for tab-safe keys

    def render_patterns_viewer(self):
        """Render individual pattern viewer section."""
        st.header("🔍 Pattern Viewer")

        patterns = self.manager.get_all_patterns()
        if not patterns:
            st.warning("No patterns found in the system.")
            return

        pattern_names = [p['name'] for p in patterns]

        # Use SessionManager for stable pattern selection
        # Create a stable key for the dropdown selection
        selected_pattern_name = self.session_manager.create_selectbox(
            "Select a pattern to view details:",
            options=pattern_names,
            selectbox_name="patterns_viewer_main_select"
        )

        # Find the selected pattern's data
        pattern_data = next((p for p in patterns if p['name'] == selected_pattern_name), None)
        
        # Fallback to first pattern if selection is invalid
        if not pattern_data and pattern_names:
            pattern_data = patterns[0]
            selected_pattern_name = pattern_data['name']

        if pattern_data:
            # Pattern header
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.subheader(f"📊 {pattern_data['name']}")
            with col2:
                st.metric("Type", pattern_data['pattern_type'])
            with col3:
                st.metric("Strength", pattern_data['strength'])

            # Pattern description
            st.markdown("**Description:**")
            st.info(pattern_data['description'])

            # Pattern details
            with st.expander("📋 Pattern Details", expanded=False):
                st.markdown(f"**Minimum Rows Required:** {pattern_data['min_rows']}")
                st.markdown(f"**Pattern Type:** {pattern_data['pattern_type']}")
                st.markdown(f"**Pattern Strength:** {pattern_data['strength']}")

            # Source code viewer
            with st.expander("💻 Source Code", expanded=False):
                if pattern_data['source_code']:
                    st.code(pattern_data['source_code'], language='python')
                else:
                    st.warning("Source code not available for this pattern.")

        else:
            st.info("Select a pattern to view its details.")

    def render_patterns_catalog(self):
        """Render patterns catalog with grid view."""
        st.header("📚 Pattern Catalog")

        patterns = self.manager.get_all_patterns()
        if not patterns:
            st.warning("No patterns found in the system.")
            return

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            pattern_types = list(set(p['pattern_type'] for p in patterns))
            selected_type = self.session_manager.create_selectbox(
                "Filter by Type:",
                options=['All'] + pattern_types,
                selectbox_name="catalog_type_filter"
            )
        with col2:
            strengths = list(set(p['strength'] for p in patterns))
            selected_strength = self.session_manager.create_selectbox(
                "Filter by Strength:",
                options=['All'] + strengths,
                selectbox_name="catalog_strength_filter"
            )
        with col3:
            min_rows_values = [p['min_rows'] for p in patterns if str(p['min_rows']).isdigit()]
            if min_rows_values:
                min_rows_filter = self.session_manager.create_slider(
                    "Minimum Rows Required:",
                    min_value=1,
                    max_value=max(int(r) for r in min_rows_values) if min_rows_values else 5,
                    value=1,
                    slider_name="catalog_rows_filter"
                )
            else:
                min_rows_filter = 1

        # Apply filters
        filtered_patterns = patterns
        if selected_type != 'All':
            filtered_patterns = [p for p in filtered_patterns if p['pattern_type'] == selected_type]
        if selected_strength != 'All':
            filtered_patterns = [p for p in filtered_patterns if p['strength'] == selected_strength]

        # Filter by min_rows
        filtered_patterns = [
            p for p in filtered_patterns
            if str(p['min_rows']).isdigit() and int(p['min_rows']) >= min_rows_filter
        ]

        st.markdown(f"**Showing {len(filtered_patterns)} of {len(patterns)} patterns**")

        # Display patterns in a grid
        if filtered_patterns:
            cols_per_row = 2
            for i in range(0, len(filtered_patterns), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(filtered_patterns):
                        pattern = filtered_patterns[i + j]
                        with col:
                            with st.container():
                                st.markdown(f"**{pattern['name']}**")
                                st.caption(f"Type: {pattern['pattern_type']} | Strength: {pattern['strength']}")
                                description = pattern['description'][:100] + "..." if len(pattern['description']) > 100 else pattern['description']
                                st.text(description)
                                if self.session_manager.create_button(
                                    "View Details",
                                    button_name=f"catalog_view_details_{pattern['name']}"
                                ):
                                    # Update the pattern viewer selection and switch to viewer section
                                    viewer_key = self.session_manager.get_unique_key("patterns_viewer_main_select", "selectbox")
                                    st.session_state[viewer_key] = pattern['name']
                                    st.session_state['active_section'] = "viewer"
                                    st.rerun()
        else:
            st.info("No patterns match the selected filters.")

    def render_download_section(self):
        """Render download section for patterns data."""
        st.header("📥 Download Patterns")

        st.markdown("Export pattern data in various formats:")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("JSON Export")
            st.markdown("Complete pattern data with descriptions and metadata")
            if self.session_manager.create_button("📄 Download JSON", button_name=f"{self.session_manager.namespace_prefix}_download_json"):
                json_data = self.manager.export_patterns_data('json')
                if json_data:
                    st.download_button(
                        label="💾 Save JSON File",
                        data=json_data,
                        file_name=f"patterns_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key=f"{self.session_manager.namespace_prefix}_save_json"
                    )
                else:
                    st.error("Failed to export JSON data")
        with col2:
            st.subheader("CSV Export")
            st.markdown("Simplified pattern data for spreadsheet analysis")
            if self.session_manager.create_button("📊 Download CSV", button_name=f"{self.session_manager.namespace_prefix}_download_csv"):
                csv_data = self.manager.export_patterns_data('csv')
                if csv_data:
                    st.download_button(
                        label="💾 Save CSV File",
                        data=csv_data,
                        file_name=f"patterns_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key=f"{self.session_manager.namespace_prefix}_save_csv"
                    )
                else:
                    st.error("Failed to export CSV data")
        with col3:
            st.subheader("Source Code")
            st.markdown("Complete patterns.py source code")
            if self.session_manager.create_button("💻 Download Source", button_name=f"{self.session_manager.namespace_prefix}_download_source"):
                source_code = self.manager.get_patterns_source_code()
                if source_code:
                    st.download_button(
                        label="💾 Save Python File",
                        data=source_code,
                        file_name=f"patterns_source_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
                        mime="text/python",
                        key=f"{self.session_manager.namespace_prefix}_save_source"
                    )
                else:
                    st.error("Failed to get source code")

        # Export statistics
        patterns = self.manager.get_all_patterns()
        if patterns:
            st.markdown("---")
            st.subheader("📊 Export Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Patterns", len(patterns))
            with col2:
                types_count = len(set(p['pattern_type'] for p in patterns))
                st.metric("Pattern Types", types_count)
            with col3:
                with_docs = len([p for p in patterns if p['description'] != 'No description available'])
                st.metric("With Documentation", with_docs)
            with col4:
                with_source = len([p for p in patterns if p['source_code'] != 'Source code not available'])
                st.metric("With Source Code", with_source)

    def render_add_pattern_section(self):
        """Render section for adding new patterns."""
        st.header("➕ Add New Pattern")

        st.markdown("Add a new candlestick pattern to the system:")

        # Pattern template selector
        template_type = self.session_manager.create_selectbox(
            "Select Pattern Template:",
            options=[
                "Empty Pattern",
                "Bullish Pattern Template", 
                "Bearish Pattern Template",
                "Reversal Pattern Template",
                "Continuation Pattern Template"
            ],
            selectbox_name="pattern_template_select"
        )

        # Generate template code
        template_code = self._get_pattern_template(template_type or "Empty Pattern")

        # Code editor
        st.subheader("Pattern Code")
        pattern_code = st.text_area(
            "Enter your pattern detection code:",
            value=template_code,
            height=400,
            help="Write a function that detects your candlestick pattern. Follow the existing pattern structure.",
            key=self.session_manager.get_unique_key("new_pattern_code", "text_area")
        )

        # Validation and save buttons
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if self.session_manager.create_button("✅ Validate Code", button_name="validate_new_pattern"):
                if pattern_code.strip():
                    is_valid = validate_python_code(pattern_code)
                    if is_valid:
                        st.success("✅ Code syntax is valid!")
                    else:
                        st.error("❌ Syntax error in the pattern code")
                else:
                    st.warning("Please enter some code to validate.")

        with col2:
            if self.session_manager.create_button("💾 Save Pattern", button_name="save_new_pattern"):
                if pattern_code.strip():
                    # Check for potential encoding issues in the pasted code
                    try:
                        # Test if the code can be encoded/decoded properly
                        test_encode = pattern_code.encode('utf-8').decode('utf-8')
                        
                        # Try to save the pattern
                        success, message = self.manager.save_new_pattern(pattern_code)
                        if success:
                            st.success(f"✅ {message}")
                            st.balloons()
                            st.session_state['new_pattern_code'] = ""
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
                            
                    except UnicodeEncodeError as e:
                        st.error(f"❌ Encoding error in pattern code: {e}")
                        st.info("💡 Try removing any special characters or symbols from your code")
                    except UnicodeDecodeError as e:
                        st.error(f"❌ Decoding error in pattern code: {e}")
                        st.info("💡 The pasted code contains characters that can't be processed")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {e}")
                        logger.error(f"Pattern save error: {e}", exc_info=True)
                else:
                    st.warning("Please enter pattern code before saving.")

        with col3:
            if self.session_manager.create_button("🔄 Reset Code", button_name="reset_new_pattern"):
                st.session_state['new_pattern_code'] = template_code
                st.rerun()

        # Help section
        with st.expander("📖 Pattern Development Guide", expanded=False):
            st.markdown("""
            **Pattern Development Guidelines:**

            1. **Function Structure**: Each pattern should be a function that takes a DataFrame and returns a boolean
            2. **Data Access**: Use column names like 'open', 'high', 'low', 'close', 'volume'
            3. **Window Size**: Consider how many candles your pattern needs (usually 1-3)
            4. **Return Type**: Return True when pattern is detected, False otherwise
            5. **Documentation**: Add a docstring explaining what the pattern detects

            **Example Pattern:**
            ```python
            def detect_hammer(df):
                \"\"\"Detect hammer candlestick pattern.\"\"\"
                if len(df) < 1:
                    return False

                row = df.iloc[-1]
                body = abs(row['close'] - row['open'])
                lower_shadow = min(row['open'], row['close']) - row['low']
                upper_shadow = row['high'] - max(row['open'], row['close'])

                return lower_shadow > (2 * body) and upper_shadow < body
            ```
            """)

    def _get_pattern_template(self, template_type: str) -> str:
        """Get template code based on selected type."""
        templates = {
            "Empty Pattern": '''def detect_new_pattern(df):
    """
    Detect new pattern.

    Args:
        df: DataFrame with OHLC data

    Returns:
        bool: True if pattern is detected, False otherwise
    """
    if len(df) < 1:
        return False

    # Your pattern detection logic here
    return False''',

            "Bullish Pattern Template": '''def detect_bullish_pattern(df):
    """
    Detect bullish candlestick pattern.

    Args:
        df: DataFrame with OHLC data

    Returns:
        bool: True if bullish pattern is detected, False otherwise
    """
    if len(df) < 2:
        return False

    current = df.iloc[-1]
    previous = df.iloc[-2]

    # Example: Bullish engulfing logic
    current_bullish = current['close'] > current['open']
    previous_bearish = previous['close'] < previous['open']

    engulfing = (current['open'] < previous['close'] and
                current['close'] > previous['open'])

    return current_bullish and previous_bearish and engulfing''',

            "Bearish Pattern Template": '''def detect_bearish_pattern(df):
    """
    Detect bearish candlestick pattern.

    Args:
        df: DataFrame with OHLC data

    Returns:
        bool: True if bearish pattern is detected, False otherwise
    """
    if len(df) < 2:
        return False

    current = df.iloc[-1]
    previous = df.iloc[-2]

    # Example: Bearish engulfing logic
    current_bearish = current['close'] < current['open']
    previous_bullish = previous['close'] > previous['open']

    engulfing = (current['open'] > previous['close'] and
                current['close'] < previous['open'])

    return current_bearish and previous_bullish and engulfing''',

            "Reversal Pattern Template": '''def detect_reversal_pattern(df):
    """
    Detect reversal candlestick pattern.

    Args:
        df: DataFrame with OHLC data

    Returns:
        bool: True if reversal pattern is detected, False otherwise
    """
    if len(df) < 3:
        return False

    first = df.iloc[-3]
    second = df.iloc[-2]
    third = df.iloc[-1]

    # Example: Morning star pattern logic
    first_bearish = first['close'] < first['open']
    second_small = abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3
    third_bullish = third['close'] > third['open']

    gap_down = second['high'] < first['close']
    gap_up = third['open'] > second['high']

    return first_bearish and second_small and third_bullish and gap_down and gap_up''',

            "Continuation Pattern Template": '''def detect_continuation_pattern(df):
    """
    Detect continuation candlestick pattern.

    Args:
        df: DataFrame with OHLC data

    Returns:
        bool: True if continuation pattern is detected, False otherwise
    """
    if len(df) < 3:
        return False

    # Look at recent trend
    trend_candles = df.iloc[-5:] if len(df) >= 5 else df

    # Example: Flag pattern in uptrend
    uptrend = trend_candles['close'].iloc[-1] > trend_candles['close'].iloc[0]

    current = df.iloc[-1]
    previous = df.iloc[-2]

    # Small consolidation after strong move
    small_body = abs(current['close'] - current['open']) < abs(previous['close'] - previous['open']) * 0.5

    return uptrend and small_body'''
        }

        return templates.get(template_type, templates["Empty Pattern"])


def main():
    initialize_patterns_page()
    ui = PatternsManagementUI(page_name="patterns_management")
    st.title("Candlestick Patterns Management")
    
    # --- Defensive: Always default to viewer on page load unless explicitly set by navigation ---
    if not ui.session_manager.get_page_state('active_section'):
        ui.session_manager.set_page_state('active_section', 'viewer')

    # Left toolbar navigation using SessionManager buttons
    st.markdown("### Navigation")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if ui.session_manager.create_button("🔍 Pattern Viewer", button_name="nav_viewer"):
            ui.session_manager.set_page_state('active_section', 'viewer')

    with col2:
        if ui.session_manager.create_button("📚 Pattern Catalog", button_name="nav_catalog"):
            ui.session_manager.set_page_state('active_section', 'catalog')

    with col3:
        if ui.session_manager.create_button("📥 Download", button_name="nav_download"):
            ui.session_manager.set_page_state('active_section', 'download')

    with col4:
        if ui.session_manager.create_button("➕ Add Pattern", button_name="nav_add"):
            ui.session_manager.set_page_state('active_section', 'add')

    st.markdown("---")

    # Render the active section
    active_section = ui.session_manager.get_page_state('active_section', 'viewer')
    if active_section == "viewer":
        ui.render_patterns_viewer()
    elif active_section == "catalog":
        ui.render_patterns_catalog()
    elif active_section == "download":
        ui.render_download_section()
    elif active_section == "add":
        ui.render_add_pattern_section()

if __name__ == "__main__":
    main()
