#!/usr/bin/env python3
"""
Test script to validate patterns management Streamlit functionality
"""

import os
import sys
import streamlit as st
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_patterns_import():
    """Test basic import functionality"""
    try:
        # Test import without running Streamlit
        import dashboard_pages.patterns_management as pm
        print("✅ Import successful")
        
        # Test class instantiation
        manager = pm.PatternsManager()
        ui = pm.PatternsManagementUI()
        print("✅ Classes instantiated successfully")
        
        # Test pattern loading
        patterns = manager.get_all_patterns()
        print(f"✅ Loaded {len(patterns)} patterns")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_ui_components():
    """Test UI component functionality without actual rendering"""
    try:
        import dashboard_pages.patterns_management as pm
        
        # Mock Streamlit to prevent actual UI rendering
        class MockStreamlit:
            def __init__(self):
                self.session_state = {}
            
            def header(self, text): pass
            def subheader(self, text): pass
            def markdown(self, text): pass
            def warning(self, text): pass
            def info(self, text): pass
            def error(self, text): pass
            def selectbox(self, label, options, **kwargs): 
                return options[0] if options else None
            def button(self, label, **kwargs): return False
            def columns(self, num): return [self] * num
            def metric(self, label, value): pass
            def expander(self, label, **kwargs): return self
            def code(self, code, **kwargs): pass
            def text_area(self, label, **kwargs): return ""
            def slider(self, label, **kwargs): return 1
            def text(self, text): pass
            def caption(self, text): pass
            def container(self): return self
            def download_button(self, **kwargs): pass
            def balloons(self): pass
            def rerun(self): pass
            def tabs(self, labels): return [self] * len(labels)
            def title(self, text): pass
            
            def __enter__(self): return self
            def __exit__(self, *args): pass
        
        # Replace streamlit temporarily
        original_st = sys.modules.get('streamlit')
        sys.modules['streamlit'] = MockStreamlit()
        
        # Test UI instantiation
        manager = pm.PatternsManager()
        ui = pm.PatternsManagementUI()
        
        print("✅ UI components created successfully")
        
        # Restore original streamlit
        if original_st:
            sys.modules['streamlit'] = original_st
        
        return True
        
    except Exception as e:
        print(f"❌ UI Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Patterns Management System...")
    print("=" * 50)
    
    success = True
    
    # Test 1: Basic imports
    print("\n📦 Testing imports...")
    success &= test_patterns_import()
    
    # Test 2: UI components
    print("\n🎨 Testing UI components...")
    success &= test_ui_components()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Patterns management is ready.")
    else:
        print("❌ Some tests failed. Check the errors above.")
    
    sys.exit(0 if success else 1)
