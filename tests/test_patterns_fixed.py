#!/usr/bin/env python3
"""Test the fixed patterns management module."""

def test_patterns_management():
    """Test patterns management import and basic functionality."""
    print("Testing patterns management fixes...")
    
    try:
        # Test import
        from dashboard_pages.patterns_management import PatternsManager, PatternsManagementUI
        print("✅ Import successful")
        
        # Test class instantiation
        manager = PatternsManager()
        print("✅ PatternsManager created successfully")

        PatternsManagementUI() # Assuming instantiation might be relevant
        print("✅ PatternsManagementUI created successfully")
        
        # Test basic pattern loading
        patterns = manager.get_all_patterns()
        print(f"✅ Loaded {len(patterns)} patterns")
        
        if patterns:
            print("Sample pattern names:")
            for i, pattern in enumerate(patterns[:3]):
                print(f"  {i+1}. {pattern['name']}")
        
        print("\n🎉 All tests passed! The fixes appear to be working.")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_patterns_management()
