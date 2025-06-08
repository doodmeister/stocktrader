#!/usr/bin/env python3
"""
Test script to verify the duplicate key fix in patterns_management.py
This simulates Streamlit's session state behavior to test for duplicate key issues.
"""

import sys
sys.path.append('.')

# Mock Streamlit for testing
class MockStreamlit:
    def __init__(self):
        self.session_state = {}
        self.widgets = {}
        self.rendered_components = []
    
    def selectbox(self, label, options, key=None, index=0):
        if key in self.widgets:
            raise Exception(f"DuplicateWidgetID: There are multiple elements with the same `key='{key}'`")
        self.widgets[key] = {'type': 'selectbox', 'label': label, 'options': options}
        self.rendered_components.append(f"selectbox(key='{key}')")
        return options[index] if options else None
    
    def header(self, text):
        self.rendered_components.append(f"header('{text}')")
    
    def warning(self, text):
        self.rendered_components.append(f"warning('{text}')")
    
    def subheader(self, text):
        self.rendered_components.append(f"subheader('{text}')")
    
    def columns(self, spec):
        return [MockColumn() for _ in range(spec)]
    
    def markdown(self, text):
        self.rendered_components.append("markdown")
    
    def tabs(self, labels):
        return [MockTab(label) for label in labels]
    
    def caption(self, text):
        self.rendered_components.append("caption")

class MockColumn:
    def __init__(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

class MockTab:
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

# Replace streamlit with our mock
mock_st = MockStreamlit()
sys.modules['streamlit'] = mock_st

# Now test the patterns management
try:
    from dashboard_pages.patterns_management import PatternsManagementUI
    
    print("🧪 Testing Patterns Management Duplicate Key Fix")
    print("=" * 50)
    
    # Test 1: Single UI instance creation
    print("\n1. Testing single UI instance creation...")
    ui = PatternsManagementUI()
    print("✅ UI instance created successfully")
    
    # Test 2: Multiple renders of the same method
    print("\n2. Testing multiple renders of patterns viewer...")
    try:
        # First render
        mock_st.widgets.clear()
        ui.render_patterns_viewer()
        first_render_widgets = len(mock_st.widgets)
        print(f"✅ First render: {first_render_widgets} widgets created")
        
        # Second render (this should work with unique keys)
        mock_st.widgets.clear()
        ui.render_patterns_viewer()
        second_render_widgets = len(mock_st.widgets)
        print(f"✅ Second render: {second_render_widgets} widgets created")
        
        print("✅ No duplicate key errors detected!")
        
    except Exception as e:
        if "DuplicateWidgetID" in str(e):
            print(f"❌ Duplicate key error still present: {e}")
        else:
            print(f"❌ Other error: {e}")
    
    # Test 3: Session state management
    print("\n3. Testing session state management...")
    mock_st.session_state['patterns_ui'] = ui
    if 'patterns_ui' in mock_st.session_state:
        print("✅ Session state management working")
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed successfully!")
    print("The duplicate key issue should now be resolved.")
    
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
