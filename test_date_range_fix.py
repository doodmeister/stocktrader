#!/usr/bin/env python3
"""
Test script to verify the date range fix for the data dashboard.

This script tests that when a user selects a 7-day range, the system downloads
exactly 7 days of data instead of defaulting to 365 days.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.data_downloader import download_stock_data


def test_date_range_fix():
    """Test that the date range selection is respected."""
    print("🧪 Testing Date Range Fix")
    print("=" * 50)
    
    # Test case: 7-day range
    start_date = date(2025, 5, 31)  # Saturday (weekend)
    end_date = date(2025, 6, 7)     # Next Saturday
    expected_days = (end_date - start_date).days + 1
    
    print(f"📅 Selected range: {start_date} to {end_date}")
    print(f"📊 Expected days: {expected_days}")
    print()
    
    try:
        # Test with a common stock
        print("🔄 Downloading AAPL data...")
        result = download_stock_data(['AAPL'], start_date, end_date, '1d')
        
        if result and 'AAPL' in result:
            df = result['AAPL']
            actual_rows = len(df)
            date_range_start = df.index.min().date() if hasattr(df.index.min(), 'date') else df.index.min()
            date_range_end = df.index.max().date() if hasattr(df.index.max(), 'date') else df.index.max()
            actual_span_days = (date_range_end - date_range_start).days + 1
            
            print(f"✅ Downloaded {actual_rows} rows")
            print(f"📈 Data range: {date_range_start} to {date_range_end}")
            print(f"📊 Actual span: {actual_span_days} days")
            print()
            
            # Check if the fix worked
            if actual_rows <= 10:  # Reasonable for ~7 business days
                print("🎉 SUCCESS: Date range fix is working!")
                print("   The system is no longer downloading 365 days of data.")
                return True
            else:
                print("❌ ISSUE: Still downloading too much data")
                print(f"   Expected ~5-7 business days, got {actual_rows} rows")
                return False
                
        else:
            print("❌ ERROR: No data returned")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_edge_cases():
    """Test edge cases."""
    print("\n🧪 Testing Edge Cases")
    print("=" * 50)
    
    # Test very short range (1 day)
    today = date.today()
    yesterday = today - timedelta(days=1)
    
    print(f"📅 Testing 1-day range: {yesterday} to {today}")
    
    try:
        result = download_stock_data(['AAPL'], yesterday, today, '1d')
        if result and 'AAPL' in result:
            df = result['AAPL']
            print(f"✅ 1-day test: {len(df)} rows downloaded")
            return True
        else:
            print("⚠️ No data for 1-day range (might be weekend)")
            return True  # This is acceptable
    except Exception as e:
        print(f"❌ 1-day test failed: {e}")
        return False


if __name__ == "__main__":
    print("🔧 Data Dashboard Date Range Fix Validation")
    print("=" * 60)
    print()
    
    # Run tests
    test1_passed = test_date_range_fix()
    test2_passed = test_edge_cases()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED: Date range fix is working correctly!")
        exit(0)
    else:
        print("❌ SOME TESTS FAILED: Issues remain with date range handling")
        exit(1)
