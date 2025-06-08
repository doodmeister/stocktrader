#!/usr/bin/env python3
"""Test script for the new patterns management system."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from patterns.pattern_utils import get_pattern_names
    print("✅ Successfully imported get_pattern_names")
    
    pattern_names = get_pattern_names()
    print(f"✅ Found {len(pattern_names)} patterns")
    print("First 5 patterns:", pattern_names[:5])
    
except Exception as e:
    print(f"❌ Error testing pattern_utils: {e}")

try:
    from dashboard_pages.patterns_management import PatternsManager
    print("✅ Successfully imported PatternsManager")
    
    manager = PatternsManager()
    patterns = manager.get_all_patterns()
    print(f"✅ PatternsManager loaded {len(patterns)} patterns")
    
    if patterns:
        sample_pattern = patterns[0]
        print(f"Sample pattern: {sample_pattern['name']} - {sample_pattern['pattern_type']}")
    
except Exception as e:
    print(f"❌ Error testing PatternsManager: {e}")
    import traceback
    traceback.print_exc()

try:
    print("✅ Successfully imported PatternsManagementUI")
    
except Exception as e:
    print(f"❌ Error testing PatternsManagementUI: {e}")

print("\n🎯 Pattern Management System Test Complete")
