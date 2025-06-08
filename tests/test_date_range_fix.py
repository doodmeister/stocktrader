#!/usr/bin/env python3
"""
Test script to verify the date range fix for the data dashboard.

This script tests that when a user selects a specific date range, the system downloads
data for that range instead of defaulting to 365 days. Uses 30+ day ranges to ensure
sufficient data points (>10) for validation.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_downloader import download_stock_data


def test_date_range_fix():
    """Test that the date range selection is respected."""
    print("🧪 Testing Date Range Fix")
    print("=" * 50)
    
    # Test case: 30-day range (sufficient for >10 data points)
    end_date = date(2025, 6, 7)     # Today
    start_date = end_date - timedelta(days=30)  # 30 days ago
    expected_days = (end_date - start_date).days + 1
    
    print(f"📅 Selected range: {start_date} to {end_date}")
    print(f"📊 Expected days: {expected_days}")
    print("📋 Using 30-day range to ensure >10 data points for validation")
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
              # Check if the fix worked (should be around 20-22 business days for 30 calendar days)
            if 15 <= actual_rows <= 35:  # Reasonable for ~30 days (accounting for weekends/holidays)
                print("🎉 SUCCESS: Date range fix is working!")
                print("   The system is downloading the correct amount of data for the selected range.")
                print(f"   Expected ~20-22 business days for 30-day period, got {actual_rows} rows")
                return True
            else:
                print("❌ ISSUE: Data count doesn't match expected range")
                print(f"   Expected 15-35 rows for 30-day period, got {actual_rows} rows")
                # Check if it's still downloading 365 days worth of data (would be ~250+ rows)
                if actual_rows > 200:
                    print("   ⚠️ This looks like the old 365-day bug is still present!")
                return False
                
        else:
            print("❌ ERROR: No data returned")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_edge_cases():
    """Test edge cases with longer periods to avoid validation errors."""
    print("\n🧪 Testing Edge Cases")
    print("=" * 50)
    
    # Test shorter range (15 days) - still sufficient for validation
    today = date.today()
    start_date = today - timedelta(days=15)
    
    print(f"📅 Testing 15-day range: {start_date} to {today}")
    print("📋 Using 15-day range to ensure >10 data points for validation")
    
    try:
        result = download_stock_data(['AAPL'], start_date, today, '1d')
        if result and 'AAPL' in result:
            df = result['AAPL']
            rows = len(df)
            print(f"✅ 15-day test: {rows} rows downloaded")
            
            # Should have at least 10 business days worth of data
            if rows >= 10:
                print("🎉 SUCCESS: Sufficient data points for validation")
                return True
            else:
                print(f"⚠️ Only {rows} rows - may not meet 10+ data point requirement")
                return False
        else:
            print("⚠️ No data for 15-day range")
            return False
    except Exception as e:
        print(f"❌ 15-day test failed: {e}")
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
