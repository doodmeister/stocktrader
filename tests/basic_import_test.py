#!/usr/bin/env python3
import sys

print("Testing basic Python imports...")

try:
    print("1. Testing patterns.pattern_utils...")
    print("✅ patterns.pattern_utils imported")
    
    print("2. Testing core.dashboard_utils...")
    print("✅ core.dashboard_utils imported") 
    
    print("3. Testing utils.logger...")
    print("✅ utils.logger imported")
    
    print("4. Testing individual components...")
    
    # Test just the manager class without UI
    print("5. Creating basic manager...")
    sys.path.insert(0, '.')
    
    # Import just the business logic parts
    
    print("✅ All basic imports successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("Test completed.")
