#!/usr/bin/env python3
"""
Test script to isolate plotly import issues
"""

print("Testing basic plotly import...")
try:
    import plotly
    print(f"✅ plotly imported successfully, version: {plotly.__version__}")
except ImportError as e:
    print(f"❌ Failed to import plotly: {e}")
    exit(1)

print("\nTesting plotly.graph_objects import...")
try:
    import plotly.graph_objects as go
    print("✅ plotly.graph_objects imported successfully")
except ImportError as e:
    print(f"❌ Failed to import plotly.graph_objects: {e}")

print("\nTesting plotly.subplots import...")
try:
    from plotly.subplots import make_subplots
    print("✅ make_subplots imported successfully")
    print(f"make_subplots function: {make_subplots}")
except ImportError as e:
    print(f"❌ Failed to import make_subplots: {e}")

print("\nTesting dashboard_utils import...")
try:
    from core.streamlit.dashboard_utils import safe_streamlit_metric
    print("✅ dashboard_utils imported successfully")
except ImportError as e:
    print(f"❌ Failed to import dashboard_utils: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting realtime_dashboard import...")
try:
    import dashboard_pages.realtime_dashboard
    print("✅ realtime_dashboard imported successfully")
except ImportError as e:
    print(f"❌ Failed to import realtime_dashboard: {e}")
    import traceback
    traceback.print_exc()

print("\nAll tests completed!")
