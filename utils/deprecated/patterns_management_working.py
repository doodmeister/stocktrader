# filepath: c:\dev\stocktrader\dashboard_pages\patterns_management.py
"""
Patterns Management - Focused interface for pattern viewing, description, download, and addition.
"""

import json
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import streamlit as st
import pandas as pd

from core.dashboard_utils import (
    initialize_dashboard_session_state,
    safe_file_write,
    handle_streamlit_error,
    setup_page
)

from patterns.pattern_utils import (
    read_patterns_file,
    get_pattern_names, 
    get_pattern_method,
    validate_python_code
)

from utils.logger import get_dashboard_logger
logger = get_dashboard_logger(__name__)

setup_page(
    title="🎯 Pattern Management",
    logger_name=__name__,
    sidebar_title="Pattern Tools"
)


class PatternsManager:
    """Core business logic for pattern management."""
    
    def __init__(self):
        self.patterns_file_path = Path("patterns/patterns.py")
    
    def get_all_patterns(self) -> List[Dict[str, Any]]:
        """Get all patterns with their metadata."""
        patterns = []
        try:
            pattern_names = get_pattern_names()
            
            for name in pattern_names:
                pattern_info = {
                    'name': name,
                    'description': 'No description available',
                    'source_code': '',
                    'pattern_type': 'Unknown',
                    'strength': 'Unknown',
                    'min_rows': 'Unknown'
                }
                
                method = get_pattern_method(name)
                if method:
                    pattern_info['description'] = method.__doc__ or 'No description available'
                    try:
                        pattern_info['source_code'] = inspect.getsource(method)
                    except Exception:
                        pattern_info['source_code'] = 'Source code not available'
                
                patterns.append(pattern_info)
                
        except Exception as e:
            logger.error(f"Error getting patterns: {e}")
            
        return patterns
    
    def export_patterns_data(self, format_type: str = 'json') -> Optional[str]:
        """Export patterns data in specified format."""
        try:
            patterns = self.get_all_patterns()
            
            if format_type == 'json':
                return json.dumps(patterns, indent=2, default=str)
            elif format_type == 'csv':
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
    
    def get_patterns_source_code(self) -> Optional[str]:
        """Get the complete patterns.py source code."""
        try:
            return read_patterns_file()
        except Exception as e:
            logger.error(f"Error reading patterns file: {e}")
            return None


class PatternsManagementUI:
    """UI controller for patterns management."""
    
    def __init__(self):
        self.manager = PatternsManager()
        
    def render_patterns_viewer(self):
        """Render individual pattern viewer section."""
        st.header("🔍 Pattern Viewer")
        
        patterns = self.manager.get_all_patterns()
        if not patterns:
            st.warning("No patterns found in the system.")
            return
        
        pattern_names = [p['name'] for p in patterns]
        selected_pattern = st.selectbox(
            "Select a pattern to view details:",
            pattern_names,
            key="pattern_viewer_select"
        )
        
        if selected_pattern:
            pattern_data = next((p for p in patterns if p['name'] == selected_pattern), None)
            if pattern_data:
                st.subheader(f"📊 {pattern_data['name']}")
                
                st.markdown("**Description:**")
                st.info(pattern_data['description'])
                
                with st.expander("💻 Source Code", expanded=False):
                    if pattern_data['source_code']:
                        st.code(pattern_data['source_code'], language='python')
                    else:
                        st.warning("Source code not available for this pattern.")
    
    def render_download_section(self):
        """Render download section for patterns data."""
        st.header("📥 Download Patterns")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("JSON Export")
            if st.button("📄 Download JSON", key="download_json"):
                json_data = self.manager.export_patterns_data('json')
                if json_data:
                    st.download_button(
                        label="💾 Save JSON File",
                        data=json_data,
                        file_name=f"patterns_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="save_json"
                    )
        
        with col2:
            st.subheader("CSV Export")
            if st.button("📊 Download CSV", key="download_csv"):
                csv_data = self.manager.export_patterns_data('csv')
                if csv_data:
                    st.download_button(
                        label="💾 Save CSV File",
                        data=csv_data,
                        file_name=f"patterns_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="save_csv"
                    )
        
        with col3:
            st.subheader("Source Code")
            if st.button("💻 Download Source", key="download_source"):
                source_code = self.manager.get_patterns_source_code()
                if source_code:
                    st.download_button(
                        label="💾 Save Python File",
                        data=source_code,
                        file_name=f"patterns_source_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
                        mime="text/python",
                        key="save_source"
                    )


def main():
    """Main entry point for patterns management."""
    initialize_dashboard_session_state()
    
    st.title("🎯 Pattern Management System")
    st.markdown("Comprehensive pattern viewing, management, and development tools")
    
    tab1, tab2 = st.tabs(["🔍 Pattern Viewer", "📥 Download"])
    
    ui = PatternsManagementUI()
    
    with tab1:
        ui.render_patterns_viewer()
    
    with tab2:
        ui.render_download_section()
    
    patterns = ui.manager.get_all_patterns()
    st.caption(f"📊 System Status: {len(patterns)} patterns available")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        handle_streamlit_error(e, "Patterns Management System")
