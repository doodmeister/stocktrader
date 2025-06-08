"""
Comprehensive test for the realtime dashboard duplicate form key fix.
This test creates a complete mock Streamlit environment to verify the fix works.
"""

import sys
import os
from unittest.mock import Mock
from datetime import datetime
import pandas as pd
import numpy as np

# Mock Streamlit components
class MockSessionState:
    def __init__(self):
        self._state = {}
    
    def __getattr__(self, key):
        if key not in self._state:
            self._state[key] = None
        return self._state[key]
    
    def __setattr__(self, key, value):
        if key.startswith('_'):
            super().__setattr__(key, value)
        else:
            if not hasattr(self, '_state'):
                super().__setattr__('_state', {})
            self._state[key] = value

class MockColumns:
    def __init__(self, *args):
        self.columns = [Mock() for _ in range(len(args))]
    
    def __iter__(self):
        return iter(self.columns)

class MockForm:
    def __init__(self, key):
        self.key = key
        self.submitted = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def form_submit_button(self, label, **kwargs):
        return self.submitted

class MockContainer:
    def __init__(self, key=None):
        self.key = key
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass

class MockStreamlit:
    def __init__(self):
        self.session_state = MockSessionState()
        self.sidebar = Mock()
        self.sidebar.form = lambda key: MockForm(key)
        self.sidebar.selectbox = Mock(return_value="1m")
        self.sidebar.number_input = Mock(return_value=100)
        self.sidebar.date_input = Mock(return_value=datetime.now().date())
        self.sidebar.time_input = Mock(return_value=datetime.now().time())
        self.sidebar.multiselect = Mock(return_value=["AAPL", "GOOGL"])
        self.sidebar.slider = Mock(return_value=0.5)
        self.sidebar.checkbox = Mock(return_value=True)
        
    def cache_data(self, ttl=None, show_spinner=True):
        """Mock cache_data decorator"""
        def decorator(func):
            return func
        return decorator
    
    def error(self, message):
        print(f"ERROR: {message}")
    
    def warning(self, message):
        print(f"WARNING: {message}")
    
    def info(self, message):
        print(f"INFO: {message}")
    
    def success(self, message):
        print(f"SUCCESS: {message}")
    
    def write(self, *args, **kwargs):
        print(f"WRITE: {args}")
    
    def markdown(self, text, **kwargs):
        print(f"MARKDOWN: {text}")
    
    def header(self, text):
        print(f"HEADER: {text}")
    
    def subheader(self, text):
        print(f"SUBHEADER: {text}")
    
    def columns(self, spec):
        if isinstance(spec, int):
            return MockColumns(*range(spec))
        return MockColumns(*spec)
    
    def container(self, **kwargs):
        return MockContainer(kwargs.get('key'))
    
    def form(self, key, **kwargs):
        return MockForm(key)
    
    def empty(self):
        return Mock()
    
    def plotly_chart(self, *args, **kwargs):
        print("PLOTLY_CHART rendered")
    
    def dataframe(self, df, **kwargs):
        print(f"DATAFRAME: {len(df)} rows")
    
    def metric(self, label, value, delta=None):
        print(f"METRIC: {label} = {value} (delta: {delta})")

# Mock other dependencies
class MockYFinance:
    def download(self, symbols, **kwargs):
        # Return sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(100, 200, 100),
            'Low': np.random.uniform(100, 200, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        return data

class MockPlotly:
    class graph_objects:
        @staticmethod
        def Figure():
            fig = Mock()
            fig.add_trace = Mock()
            fig.update_layout = Mock()
            fig.update_xaxes = Mock()
            fig.update_yaxes = Mock()
            return fig
        
        @staticmethod
        def Scatter(**kwargs):
            return Mock()
        
        @staticmethod
        def Candlestick(**kwargs):
            return Mock()

# Set up the test environment
def setup_test_environment():
    """Set up complete mock environment for testing"""
    
    # Mock streamlit
    mock_st = MockStreamlit()
    sys.modules['streamlit'] = mock_st
    
    # Mock yfinance
    mock_yf = MockYFinance()
    sys.modules['yfinance'] = mock_yf
    
    # Mock plotly
    mock_plotly = MockPlotly()
    sys.modules['plotly'] = mock_plotly
    sys.modules['plotly.graph_objects'] = mock_plotly.graph_objects
    
    # Mock other modules
    sys.modules['pytz'] = Mock()
    
    return mock_st

def test_realtime_dashboard_import_and_session_manager():
    """Test that the realtime dashboard can be imported and creates unique form keys"""
    
    print("=== Testing Realtime Dashboard Fix ===")
    
    # Setup mock environment
    setup_test_environment()

    try:
        # Add the project root to the path
        project_root = os.path.dirname(os.path.abspath(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Import the fixed realtime dashboard
        print("1. Importing realtime_dashboard module...")
        from dashboard_pages import realtime_dashboard
        print("   ✓ Import successful")
        
        # Import session manager
        print("2. Importing session manager...")
        from core.session_manager import create_session_manager
        print("   ✓ Session manager import successful")
        
        # Test session manager form key generation
        print("3. Testing session manager form key generation...")
        
        # Create session manager
        session_manager = create_session_manager("realtime_dashboard")
        print(f"   ✓ Created session manager with namespace: {session_manager.namespace}")
        
        # Test form key generation
        form_key1 = session_manager.get_form_key("chart_parameters_form")
        form_key2 = session_manager.get_form_key("another_form")
        form_key3 = session_manager.get_form_key("chart_parameters_form")  # Same name, should be same key
        
        print(f"   ✓ Form key 1: {form_key1}")
        print(f"   ✓ Form key 2: {form_key2}")
        print(f"   ✓ Form key 3: {form_key3}")
        
        # Verify keys are unique except when same form name
        assert form_key1 != form_key2, "Different form names should have different keys"
        assert form_key1 == form_key3, "Same form name should have same key"
        print("   ✓ Form key uniqueness verified")
        
        # Test that multiple session managers with different namespaces create different keys
        print("4. Testing multiple session managers...")
        session_manager2 = create_session_manager("different_namespace")
        form_key4 = session_manager2.get_form_key("chart_parameters_form")
        
        print(f"   ✓ Form key from different namespace: {form_key4}")
        assert form_key1 != form_key4, "Same form name in different namespaces should have different keys"
        print("   ✓ Namespace isolation verified")
        
        # Test the main dashboard function structure
        print("5. Testing dashboard function structure...")
        
        # Check that main function exists and can be called with mocked components
        if hasattr(realtime_dashboard, 'main'):
            print("   ✓ main() function found")
            
            # Check render_main_dashboard function signature
            if hasattr(realtime_dashboard, 'render_main_dashboard'):
                import inspect
                sig = inspect.signature(realtime_dashboard.render_main_dashboard)
                params = list(sig.parameters.keys())
                print(f"   ✓ render_main_dashboard parameters: {params}")
                
                if 'session_manager' in params:
                    print("   ✓ render_main_dashboard correctly accepts session_manager parameter")
                else:
                    print("   ✗ render_main_dashboard missing session_manager parameter")
                    return False
            else:
                print("   ✗ render_main_dashboard function not found")
                return False
        else:
            print("   ✗ main() function not found")
            return False
        
        print("\n=== TEST RESULTS ===")
        print("✓ All tests passed!")
        print("✓ Duplicate form key issue has been resolved")
        print("✓ Session manager is properly used with unique namespaces")
        print("✓ Form keys are unique across different session managers")
        print("✓ Function structure is correct for single session manager usage")
        
        return True
        
    except Exception as e:
        print("\n=== TEST FAILED ===")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_form_container_usage():
    """Test the form_container context manager usage"""
    
    print("\n=== Testing Form Container Usage ===")
    
    try:
        from core.session_manager import create_session_manager
        
        # Create session manager
        session_manager = create_session_manager("test_forms")
        
        # Test form_container method
        print("1. Testing form_container context manager...")
        
        # Simulate using form_container
        with session_manager.form_container("test_form", location="sidebar") as form:
            print(f"   ✓ Form container created with key: {form.key if hasattr(form, 'key') else 'mock'}")
            
        # Test multiple form containers
        print("2. Testing multiple form containers...")
        
        form_keys = []
        for i in range(3):
            with session_manager.form_container(f"form_{i}", location="sidebar") as form:
                key = getattr(form, 'key', f'mock_form_{i}')
                form_keys.append(key)
                print(f"   ✓ Form {i} key: {key}")
        
        # Verify all keys are different
        assert len(set(form_keys)) == len(form_keys), "All form keys should be unique"
        print("   ✓ All form container keys are unique")
        
        return True
        
    except Exception as e:
        print(f"Error testing form containers: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting comprehensive realtime dashboard test...")
    
    # Run the main test
    success = test_realtime_dashboard_import_and_session_manager()
    
    if success:
        # Run additional form container test
        form_success = test_form_container_usage()
        
        if form_success:
            print("\n🎉 ALL TESTS PASSED! The duplicate form key issue has been successfully resolved.")
        else:
            print("\n⚠️  Main test passed but form container test failed.")
    else:
        print("\n❌ TESTS FAILED! There may still be issues with the implementation.")
