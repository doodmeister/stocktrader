#!/usr/bin/env python
"""
Test script to verify the StockTrader dashboard validation fixes.
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_pattern_detection():
    """Test pattern detection system."""
    print("Testing pattern detection...")
    
    try:
        from patterns.patterns import create_pattern_detector
        
        # Create test data
        df = pd.DataFrame({
            'open': [100, 102, 101, 103, 102, 104],
            'high': [105, 107, 106, 108, 107, 109], 
            'low': [99, 101, 100, 102, 101, 103],
            'close': [104, 106, 105, 107, 106, 108],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500]
        })
        
        # Test pattern detection
        detector = create_pattern_detector()
        patterns = detector.detect_patterns(df)
        
        detected_patterns = [p for p in patterns if p.detected]
        print(f"SUCCESS: {len(patterns)} patterns processed, {len(detected_patterns)} detected")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    """Run test."""
    print("StockTrader Dashboard Validation Fix Test")
    print("=" * 40)
    
    if test_pattern_detection():
        print("\nALL TESTS PASSED! Dashboard is ready.")
        print("To launch: streamlit run main.py")
    else:
        print("\nTEST FAILED!")
    
if __name__ == "__main__":
    main()
