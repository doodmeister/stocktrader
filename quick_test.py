import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import dashboard_pages.patterns_management as pm
    print("✅ Import successful")
    
    manager = pm.PatternsManager()
    patterns = manager.get_all_patterns()
    print(f"✅ Found {len(patterns)} patterns")
    
    ui = pm.PatternsManagementUI()
    print("✅ UI instantiated successfully")
    
    # Test export functionality
    json_data = manager.export_patterns_data('json')
    print(f"✅ JSON export: {len(json_data)} characters")
    
    csv_data = manager.export_patterns_data('csv')
    print(f"✅ CSV export: {len(csv_data)} characters")
    
    print("🎉 All core functionality is working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
