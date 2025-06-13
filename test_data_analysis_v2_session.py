#!/usr/bin/env python3
"""
Test script to verify session state management in data_analysis_v2.py
This script simulates page loads and checks that session state is managed correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

# Mock streamlit before importing the module
class MockStreamlit:
    def __init__(self):
        self.session_state = {}
    
    def get(self, key, default=None):
        return self.session_state.get(key, default)
    
    def __setitem__(self, key, value):
        self.session_state[key] = value
    
    def __getitem__(self, key):
        return self.session_state[key]
    
    def __contains__(self, key):
        return key in self.session_state
    
    def __delitem__(self, key):
        del self.session_state[key]
        
    def pop(self, key, default=None):
        return self.session_state.pop(key, default)
    
    def keys(self):
        return self.session_state.keys()

# Create mock streamlit
mock_st = MagicMock()
mock_st.session_state = MockStreamlit()

class TestDataAnalysisV2SessionState(unittest.TestCase):
    
    def setUp(self):
        """Reset session state before each test"""
        mock_st.session_state.session_state = {}
        
    @patch('dashboard_pages.data_analysis_v2.st', mock_st)
    @patch('dashboard_pages.data_analysis_v2.SessionManager')
    @patch('dashboard_pages.data_analysis_v2.setup_page')
    @patch('dashboard_pages.data_analysis_v2.get_dashboard_logger')
    @patch('dashboard_pages.data_analysis_v2.DataValidator')
    def test_first_load_clears_state(self, mock_validator, mock_logger, mock_setup, mock_session_manager):
        """Test that session state is cleared only on first load"""
        
        # Mock the SessionManager
        mock_sm_instance = MagicMock()
        mock_sm_instance.namespace = "data_analysis_v2_stable"
        mock_session_manager.return_value = mock_sm_instance
        
        # Set up some existing state (simulating a previous session)
        mock_st.session_state['uploaded_dataframe'] = pd.DataFrame({'test': [1, 2, 3]})
        mock_st.session_state['data_analysis_v2_summary'] = "old summary"
        mock_st.session_state['validation_result'] = "old validation"
        
        # Import and run the initialization function
        from dashboard_pages.data_analysis_v2 import _init_data_analysis_v2_state
        
        # First call should clear the state
        _init_data_analysis_v2_state()
        
        # Check that the first load flag is set
        self.assertTrue(mock_st.session_state.get('data_analysis_v2_stable_first_load_done'))
        
        # Check that old state was cleared
        self.assertNotIn('uploaded_dataframe', mock_st.session_state.session_state)
        self.assertEqual(mock_st.session_state.get('data_analysis_v2_summary'), '')
        self.assertIsNone(mock_st.session_state.get('validation_result'))
        
        # Set some new state
        mock_st.session_state['data_analysis_v2_summary'] = "new summary"
        
        # Second call should NOT clear the state
        _init_data_analysis_v2_state()
        
        # State should be preserved
        self.assertEqual(mock_st.session_state.get('data_analysis_v2_summary'), "new summary")
        
    @patch('dashboard_pages.data_analysis_v2.st', mock_st)
    @patch('dashboard_pages.data_analysis_v2.SessionManager')
    @patch('dashboard_pages.data_analysis_v2.setup_page')
    @patch('dashboard_pages.data_analysis_v2.get_dashboard_logger')
    @patch('dashboard_pages.data_analysis_v2.DataValidator')
    def test_defaults_are_set(self, mock_validator, mock_logger, mock_setup, mock_session_manager):
        """Test that default values are set correctly"""
        
        # Mock the SessionManager
        mock_sm_instance = MagicMock()
        mock_sm_instance.namespace = "data_analysis_v2_stable"
        mock_session_manager.return_value = mock_sm_instance
        
        # Import and run the initialization function
        from dashboard_pages.data_analysis_v2 import _init_data_analysis_v2_state
        
        _init_data_analysis_v2_state()
        
        # Check that defaults are set
        expected_defaults = {
            'data_analysis_v2_detected_patterns': [],
            'data_analysis_v2_summary': '',
            'data_analysis_v2_gpt_response': '',
            'data_analysis_v2_pattern_detection_attempted': False,
            'data_analysis_v2_new_file_uploaded_this_run': False,
            'data_analysis_v2_send_to_gpt_requested': False,
            'data_analysis_v2_summary_for_gpt': '',
            'validation_result': None,
            'data_analysis_v2_tech_summary': '',
        }
        
        for key, expected_value in expected_defaults.items():
            self.assertEqual(mock_st.session_state.get(key), expected_value, 
                           f"Default value for {key} should be {expected_value}")

def run_tests():
    """Run the test suite"""
    unittest.main(argv=[''], exit=False, verbosity=2)

if __name__ == "__main__":
    print("Testing Data Analysis V2 Session State Management...")
    run_tests()
    print("Session state tests completed!")
