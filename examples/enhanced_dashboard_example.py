#!/usr/bin/env python3
"""
Example Dashboard Using Enhanced Dashboard Utils

This example demonstrates how to use the new features in dashboard_utils.py
while maintaining compatibility with existing code.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Import enhanced dashboard utilities
from core.dashboard_utils import (
    # Original functions (still work exactly the same)
    initialize_dashboard_session_state,
    setup_page,
    handle_streamlit_error,
    create_candlestick_chart,
    
    # New enhanced functions
    monitor_performance,
    sanitize_user_input,
    create_advanced_notification,
    create_advanced_candlestick_chart,
    safe_json_save,
    safe_json_load,
    create_tiered_cache_key,
    cache_context
)

# Page configuration
st.set_page_config(
    page_title="Enhanced Dashboard Example",
    page_icon="📈",
    layout="wide"
)

# Initialize session state with enhanced features
initialize_dashboard_session_state()

# Enable debug mode for demonstration
st.session_state.debug_mode = True
st.session_state.enable_performance_monitoring = True

# Setup page with enhanced features
setup_page(
    title="Enhanced Dashboard Example",
    icon="📈",
    description="Demonstrating enhanced dashboard utilities"
)

def generate_sample_data():
    """Generate sample OHLC data for demonstration."""
    dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='D')
    np.random.seed(42)
    
    # Generate realistic stock data
    base_price = 100
    prices = [base_price]
    
    for _ in range(len(dates) - 1):
        change = np.random.normal(0, 2)
        new_price = max(prices[-1] + change, 10)  # Minimum price of 10
        prices.append(new_price)
    
    # Create OHLC data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        high = close + np.random.uniform(0, 3)
        low = close - np.random.uniform(0, 3)
        open_price = prices[i-1] if i > 0 else close
        volume = np.random.randint(100000, 1000000)
        
        data.append({
            'Date': date,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df

@monitor_performance
def load_and_process_data():
    """Example of monitored function that loads and processes data."""
    time.sleep(0.5)  # Simulate processing time
    return generate_sample_data()

def main():
    """Main dashboard function."""
    
    st.header("🚀 Enhanced Dashboard Features Demo")
    
    # Sidebar for controls
    st.sidebar.header("Dashboard Controls")
    
    # Security demo: User input sanitization
    st.sidebar.subheader("🔒 Security Features")
    user_input = st.sidebar.text_input("Enter some text (will be sanitized):")
    if user_input:
        sanitized = sanitize_user_input(user_input, max_length=100)
        st.sidebar.text(f"Sanitized: {sanitized}")
    
    # Performance monitoring demo
    st.sidebar.subheader("⚡ Performance Monitoring")
    if st.sidebar.button("Load Data (Monitored)"):
        with st.spinner("Loading data..."):
            df = load_and_process_data()
            
        # Save to cache with tiered key
        cache_key = create_tiered_cache_key("sample_data", cache_level="session")
        st.session_state[cache_key] = df
        
        create_advanced_notification(
            "Data loaded successfully with performance monitoring!",
            "success",
            auto_close=3
        )
    
    # Get cached data
    cache_key = create_tiered_cache_key("sample_data", cache_level="session")
    df = st.session_state.get(cache_key)
    
    if df is None:
        st.info("👆 Click 'Load Data' in the sidebar to start")
        return
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📊 Enhanced Chart Features")
        
        # Chart type selection
        chart_type = st.selectbox(
            "Select Chart Type:",
            ["Original Chart", "Advanced Chart with Indicators"]
        )
        
        if chart_type == "Original Chart":
            # Use original function (backward compatibility)
            fig = create_candlestick_chart(
                df,
                title="Original Candlestick Chart",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Use enhanced chart with technical indicators
            # Calculate some simple technical indicators
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # Create RSI indicator (simplified)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            technical_indicators = {
                'RSI': rsi
            }
            
            fig = create_advanced_candlestick_chart(
                df,
                title="Advanced Chart with Technical Indicators",
                height=600,
                volume_subplot=True,
                technical_indicators=technical_indicators,
                custom_styling={
                    'increasing_color': '#00ff88',
                    'decreasing_color': '#ff4444',
                    'background_color': '#0e1117'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📋 Data Summary")
        
        # Display basic statistics
        st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
        st.metric("Daily Change", f"${df['Close'].iloc[-1] - df['Close'].iloc[-2]:.2f}")
        st.metric("Volume", f"{df['Volume'].iloc[-1]:,}")
        
        # Advanced notifications demo
        st.subheader("🔔 Advanced Notifications")
        
        if st.button("Success Notification"):
            create_advanced_notification(
                "Operation completed successfully!",
                "success",
                auto_close=5
            )
        
        if st.button("Warning Notification"):
            create_advanced_notification(
                "This is a warning message.",
                "warning",
                dismissible=True
            )
        
        if st.button("Notification with Actions"):
            actions = [
                {
                    'label': 'View Details',
                    'callback': lambda: st.session_state.update({'show_details': True})
                },
                {
                    'label': 'Download',
                    'callback': lambda: create_advanced_notification("Download started!", "info")
                }
            ]
            create_advanced_notification(
                "Task completed! Choose an action:",
                "success",
                actions=actions
            )
    
    # File operations demo
    st.subheader("💾 Enhanced File Operations")
    
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("Save Data as JSON"):
            data_dict = {
                'summary': {
                    'total_rows': len(df),
                    'date_range': {
                        'start': df.index.min().isoformat(),
                        'end': df.index.max().isoformat()
                    },
                    'price_range': {
                        'min': df['Close'].min(),
                        'max': df['Close'].max()
                    }
                },
                'last_update': datetime.now().isoformat()
            }
            
            success = safe_json_save(data_dict, "temp/dashboard_summary.json")
            if success:
                create_advanced_notification(
                    "Data summary saved successfully!",
                    "success"
                )
    
    with col4:
        if st.button("Load Saved JSON"):
            data = safe_json_load("temp/dashboard_summary.json", default={})
            if data:
                st.json(data)
                create_advanced_notification(
                    "Data loaded from JSON file!",
                    "info"
                )
    
    # Debug information (only shown when debug mode is enabled)
    if st.session_state.get('debug_mode', False):
        with st.expander("🐛 Debug Information"):
            st.subheader("Session State")
            debug_state = {k: str(v)[:100] + "..." if len(str(v)) > 100 else v 
                          for k, v in st.session_state.items()}
            st.json(debug_state)
            
            st.subheader("Performance Metrics")
            if hasattr(st.session_state, 'performance_metrics'):
                st.json(st.session_state.performance_metrics)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tip**: All enhancements are backward compatible. "
        "Existing code continues to work without changes!"
    )

if __name__ == "__main__":
    # Error handling with enhanced features
    try:
        main()
    except Exception as e:
        handle_streamlit_error(
            e, 
            context="Enhanced Dashboard Example",
            show_recovery_suggestions=True
        )
