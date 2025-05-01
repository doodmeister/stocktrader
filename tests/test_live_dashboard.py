"""
Tests for the live dashboard functionality.
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the components to test
from pages.live_dashboard import (
    DashboardConfig,
    load_price_data,
    render_price_chart,
    render_metrics
)


class TestDashboardConfig(unittest.TestCase):
    """Tests for the DashboardConfig model."""
    
    # filepath: c:\dev\stocktrader\tests\test_live_dashboard.py
"""
Tests for the live dashboard functionality.
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the components to test
from pages.live_dashboard import (
    DashboardConfig,
    load_price_data,
    render_price_chart,
    render_metrics
)


class TestDashboardConfig(unittest.TestCase):
    """Tests for the DashboardConfig model."""
    
    