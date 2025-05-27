#!/usr/bin/env python3
"""
Comprehensive Dashboard Test Script
Tests that all dashboard components work correctly
"""

import sys
import os
import importlib
import traceback

def test_main_dashboard():
    """Test the main dashboard module"""
    print("🎯 Testing main dashboard...")
    try:
        import streamlit_dashboard
        print("✅ Main dashboard module imports successfully")
        
        # Test dashboard discovery
        dashboard = streamlit_dashboard.StockTradingDashboard()
        pages = dashboard._discover_pages()
        print(f"✅ Discovered {len(pages)} dashboard pages: {list(pages.keys())}")
        
        return True, pages
    except Exception as e:
        print(f"❌ Main dashboard test failed: {e}")
        traceback.print_exc()
        return False, {}

def test_page_loading(pages):
    """Test that all discovered pages can be loaded"""
    print("\n📄 Testing individual page loading...")
    
    success_count = 0
    total_pages = len(pages)
    
    for page_name, page_info in pages.items():
        try:
            # Import the module
            module_path = page_info['module']
            module = importlib.import_module(module_path)
            
            # Check if it has the expected function
            if hasattr(module, page_info['function']):
                print(f"✅ {page_name}: Module and function available")
                success_count += 1
            else:
                print(f"⚠️  {page_name}: Module loads but function '{page_info['function']}' not found")
        except Exception as e:
            print(f"❌ {page_name}: Failed to load - {e}")
    
    print(f"\n📊 Page Loading Results: {success_count}/{total_pages} pages loaded successfully")
    return success_count == total_pages

def test_configuration():
    """Test configuration and dependencies"""
    print("\n⚙️  Testing configuration and dependencies...")
    
    try:
        # Test security module
        from utils.security import get_api_credentials, validate_credentials, get_sandbox_mode
        print("✅ Security utilities available")
        
        # Test pattern utilities
        from patterns.pattern_utils import add_candlestick_pattern_features
        print("✅ Pattern utilities available")
        
        # Test optional dependencies
        try:
            import talib
            print("✅ TA-Lib available")
        except ImportError:
            print("ℹ️  TA-Lib not available (optional)")
            
        try:
            import ta
            print("✅ Technical Analysis library available")
        except ImportError:
            print("ℹ️  Technical Analysis library not available (optional)")
            
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run comprehensive dashboard tests"""
    print("🧪 Running Comprehensive Dashboard Tests\n")
    print("=" * 60)
    
    # Test main dashboard
    dashboard_success, pages = test_main_dashboard()
    
    if not dashboard_success:
        print("\n❌ Dashboard tests failed - cannot continue")
        sys.exit(1)
    
    # Test page loading
    pages_success = test_page_loading(pages)
    
    # Test configuration
    config_success = test_configuration()
    
    # Final results
    print("\n" + "=" * 60)
    print("🎉 COMPREHENSIVE TEST RESULTS:")
    print(f"   📱 Main Dashboard: {'✅ PASS' if dashboard_success else '❌ FAIL'}")
    print(f"   📄 Page Loading: {'✅ PASS' if pages_success else '❌ FAIL'}")
    print(f"   ⚙️  Configuration: {'✅ PASS' if config_success else '❌ FAIL'}")
    
    all_tests_passed = dashboard_success and pages_success and config_success
    
    if all_tests_passed:
        print("\n🎉 ALL TESTS PASSED! Dashboard is ready for use.")
        print("🌐 Access your dashboard at: http://localhost:8502")
    else:
        print("\n⚠️  Some tests failed. Please review the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
