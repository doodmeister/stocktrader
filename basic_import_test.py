#!/usr/bin/env python3
import sys
import os

print("Testing basic Python imports...")

try:
    print("1. Testing patterns.pattern_utils...")
    from patterns.pattern_utils import get_pattern_names
    print("✅ patterns.pattern_utils imported")
    
    print("2. Testing core.dashboard_utils...")
    from core.dashboard_utils import safe_file_write
    print("✅ core.dashboard_utils imported") 
    
    print("3. Testing utils.logger...")
    from utils.logger import get_dashboard_logger
    print("✅ utils.logger imported")
    
    print("4. Testing individual components...")
    
    # Test just the manager class without UI
    print("5. Creating basic manager...")
    sys.path.insert(0, '.')
    
    # Import just the business logic parts
    import json
    import inspect
    from pathlib import Path
    from typing import Dict, List, Optional, Any
    
    print("✅ All basic imports successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("Test completed.")
