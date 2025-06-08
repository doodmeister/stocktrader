#!/usr/bin/env python3
import sys
import os

print("Starting test...")
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

try:
    print("Attempting import...")
    sys.path.insert(0, '.')
    from dashboard_pages.patterns_management import PatternsManager
    print("✅ Import successful!")
    
    print("Creating manager...")
    manager = PatternsManager()
    print("✅ Manager created!")
    
    print("Getting patterns...")
    patterns = manager.get_all_patterns()
    print(f"✅ Found {len(patterns)} patterns")
    
    if patterns:
        print("Sample patterns:")
        for i, pattern in enumerate(patterns[:3]):
            print(f"  {i+1}. {pattern.get('name', 'Unknown')}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("Test completed.")
