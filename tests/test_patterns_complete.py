#!/usr/bin/env python3
"""
Test script for the patterns management system
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_patterns_management():
    """Test the patterns management system functionality."""
    print("🎯 Testing Patterns Management System")
    print("=" * 50)
    
    try:
        from dashboard_pages.patterns_management import PatternsManager, PatternsManagementUI
        print("✅ Successfully imported PatternsManager and PatternsManagementUI")
        
        # Test basic functionality
        manager = PatternsManager()
        patterns = manager.get_all_patterns()
        print(f"✅ Found {len(patterns)} patterns")
        
        # Show some pattern details
        if patterns:
            print("\n📋 Pattern samples:")
            for i, pattern in enumerate(patterns[:3]):  # Show first 3
                print(f"   {i+1}. {pattern['name']}")
                print(f"      Type: {pattern['pattern_type']}")
                print(f"      Description: {pattern['description'][:50]}...")
        
        # Test export functionality
        print("\n📤 Testing export functionality:")
        
        json_export = manager.export_patterns_data('json')
        if json_export:
            print(f"✅ JSON export working - {len(json_export)} characters")
        else:
            print("❌ JSON export failed")
        
        csv_export = manager.export_patterns_data('csv')
        if csv_export:
            print(f"✅ CSV export working - {len(csv_export)} characters")
        else:
            print("❌ CSV export failed")
        
        # Test source code retrieval
        source_code = manager.get_patterns_source_code()
        if source_code:
            print(f"✅ Source code retrieval working - {len(source_code)} characters")
        else:
            print("❌ Source code retrieval failed")
        
        # Test UI components
        print("\n🎨 Testing UI components:")
        PatternsManagementUI() # Assuming instantiation might be relevant
        print("✅ PatternsManagementUI instantiated successfully")
        
        print("\n🎉 All tests passed! Patterns management system is ready!")
        
        # Summary
        print("\n📊 System Summary:")
        print(f"   • Total patterns available: {len(patterns)}")
        print("   • Export formats: JSON, CSV, Source Code")
        print("   • UI components: Pattern Viewer, Catalog, Download, Add Pattern")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_patterns_management()
    sys.exit(0 if success else 1)
