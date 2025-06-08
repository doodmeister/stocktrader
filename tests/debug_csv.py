#!/usr/bin/env python3
"""
Debug script to test CSV loading exactly like Streamlit does it
"""

import io
import pandas as pd
from deprecated.data_analysis import load_df

def test_csv_loading():
    """Test CSV loading with both files"""
    
    files_to_test = [
        'test_stock_data.csv',
        'data/AAPL_1d.csv'
    ]
    
    for file_path in files_to_test:
        print(f"\n{'='*60}")
        print(f"Testing: {file_path}")
        print(f"{'='*60}")
        
        try:
            # Read file as bytes (like Streamlit does)
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            print(f"File size: {len(file_content)} bytes")
            
            # Create a BytesIO object (like Streamlit UploadedFile)
            uploaded_file = io.BytesIO(file_content)
            uploaded_file.name = file_path.split('/')[-1]  # Set filename
            
            # Test direct pandas read first
            uploaded_file.seek(0)
            df_direct = pd.read_csv(uploaded_file)
            print(f"Direct pandas read: {len(df_direct)} rows, {len(df_direct.columns)} columns")
            print(f"Columns: {list(df_direct.columns)}")
            
            # Test our load_df function
            uploaded_file.seek(0)
            df_our_function = load_df(uploaded_file)
            print(f"Our load_df function: {len(df_our_function)} rows, {len(df_our_function.columns)} columns")
            print(f"Columns: {list(df_our_function.columns)}")
            
            # Compare results
            if len(df_direct) == len(df_our_function):
                print("✅ Row counts match!")
            else:
                print(f"❌ Row count mismatch: direct={len(df_direct)}, our_function={len(df_our_function)}")
            
            print("Sample data from our function:")
            if len(df_our_function) > 0:
                print(df_our_function.head(2))
            else:
                print("No data!")
                
        except Exception as e:
            print(f"❌ Error testing {file_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_csv_loading()
