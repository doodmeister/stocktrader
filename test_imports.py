#!/usr/bin/env python3
"""Test script to verify dashboard page imports."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_import(module_name):
    """Test importing a dashboard page module."""
    try:
        __import__(module_name)
        print(f"✅ {module_name} imports successfully")
        return True
    except Exception as e:
        print(f"❌ {module_name} import error: {e}")
        return False

def main():
    """Test all dashboard page imports."""
    print("🧪 Testing dashboard page imports...")
    
    # Test key dashboard pages
    pages_to_test = [
        "dashboard_pages.simple_trade",
        "dashboard_pages.advanced_ai_trade", 
        "dashboard_pages.realtime_dashboard_v3",
        "dashboard_pages.data_dashboard_v2",
        "dashboard_pages.classic_strategy_backtest"
    ]
    
    success_count = 0
    total_count = len(pages_to_test)
    
    for page in pages_to_test:
        if test_import(page):
            success_count += 1
    
    print(f"\n📊 Results: {success_count}/{total_count} pages imported successfully")
    
    if success_count == total_count:
        print("🎉 All dashboard pages imported successfully!")
    else:
        print("⚠️ Some dashboard pages have import issues")

if __name__ == "__main__":
    main()
