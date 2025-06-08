#!/usr/bin/env python3
"""
Demo script for patterns management system
This creates a simple Streamlit app to test the patterns management functionality
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the patterns management module
from dashboard_pages.patterns_management import PatternsManager, PatternsManagementUI, initialize_patterns_page

def main():
    """Demo main function"""
    # Initialize the page
    initialize_patterns_page()
    
    st.title("🎯 Patterns Management Demo")
    st.markdown("Testing the patterns management system functionality")
    
    # Create instances
    manager = PatternsManager()
    ui = PatternsManagementUI()
    
    # Show basic stats
    patterns = manager.get_all_patterns()
    st.success(f"✅ Successfully loaded {len(patterns)} patterns!")
    
    # Show pattern list
    if patterns:
        st.subheader("📋 Available Patterns")
        for i, pattern in enumerate(patterns[:5]):  # Show first 5
            st.write(f"{i+1}. **{pattern['name']}** - {pattern['pattern_type']}")
        
        if len(patterns) > 5:
            st.info(f"... and {len(patterns) - 5} more patterns")
    
    # Test export functionality
    st.subheader("📤 Export Test")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Test JSON Export"):
            json_data = manager.export_patterns_data('json')
            st.success(f"✅ JSON export ready: {len(json_data)} characters")
    
    with col2:
        if st.button("Test CSV Export"):
            csv_data = manager.export_patterns_data('csv')
            st.success(f"✅ CSV export ready: {len(csv_data)} characters")
    
    st.markdown("---")
    st.caption("🎉 Patterns Management System is working correctly!")

if __name__ == "__main__":
    main()
