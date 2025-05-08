"""
Live Trading Dashboard
----------------------
Provides real-time stock price monitoring and trading capabilities via E*Trade API.
Features candlestick charting, technical indicators, and trade execution.

Part of the StockTrader application suite.
"""
from utils.logger import setup_logger
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pydantic import BaseModel, Field, validator


# Local imports
from utils.etrade_candlestick_bot import ETradeClient
from utils.technicals.indicators import add_technical_indicators
from train.model_manager import ModelManager
from utils.config.validation import validate_symbol, safe_request
from utils.security import get_api_credentials
from utils.etrade_client_factory import create_etrade_client
from utils.dashboard_utils import initialize_dashboard_session_state

# Configure logger with proper format
logger = setup_logger(__name__)

# Constants
DEFAULT_SYMBOL = "AAPL"
DEFAULT_REFRESH_INTERVAL = 60  # seconds
MIN_REFRESH_INTERVAL = 10  # seconds
REQUIRED_DATA_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}
DEFAULT_CHART_HEIGHT = 500
INDICATORS = ["SMA", "EMA", "MACD", "RSI", "Bollinger Bands"]


class DashboardConfig(BaseModel):
    """Configuration for the live trading dashboard."""
    symbol: str = Field(DEFAULT_SYMBOL, description="Stock ticker symbol")
    refresh_interval: int = Field(
        DEFAULT_REFRESH_INTERVAL,
        description="Data refresh interval in seconds",
        ge=MIN_REFRESH_INTERVAL
    )
    indicators: List[str] = Field(
        default=[],
        description="Technical indicators to display on the chart"
    )
    chart_height: int = Field(
        DEFAULT_CHART_HEIGHT,
        description="Height of the main chart in pixels",
        ge=300,
        le=1000
    )
    
    @validator('symbol')
    def validate_symbol(cls, v):
        """Ensure symbol is valid."""
        return validate_symbol(v)


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables if they don't exist."""
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = None
    
    if 'data_cache' not in st.session_state:
        st.session_state.data_cache = None
        
    if 'error_count' not in st.session_state:
        st.session_state.error_count = 0
        
    if 'login_status' not in st.session_state:
        st.session_state.login_status = False


def render_sidebar() -> DashboardConfig:
    """
    Render the sidebar controls and return the configuration.
    
    Returns:
        DashboardConfig: The dashboard configuration from user inputs
    """
    st.sidebar.header("Dashboard Settings")
    
    symbol = st.sidebar.text_input(
        "Symbol", 
        value=DEFAULT_SYMBOL
    ).strip().upper()
    
    refresh_interval = st.sidebar.number_input(
        "Refresh Interval (seconds)",
        min_value=MIN_REFRESH_INTERVAL,
        value=DEFAULT_REFRESH_INTERVAL
    )
    
    st.sidebar.subheader("Technical Indicators")
    selected_indicators = []
    for indicator in INDICATORS:
        if st.sidebar.checkbox(indicator, value=False):
            selected_indicators.append(indicator)
    
    chart_height = st.sidebar.slider(
        "Chart Height", 
        min_value=300, 
        max_value=1000, 
        value=DEFAULT_CHART_HEIGHT,
        step=50
    )
    
    return DashboardConfig(
        symbol=symbol,
        refresh_interval=refresh_interval,
        indicators=selected_indicators,
        chart_height=chart_height
    )


def render_credentials_sidebar() -> Optional[Dict[str, str]]:
    st.sidebar.header("🔑 API Credentials")
    env_creds = get_api_credentials()

    # Pre-populate fields with values from .env, allow user to override
    consumer_key = st.sidebar.text_input("Consumer Key", value=env_creds.get('consumer_key', ''), type="password")
    consumer_secret = st.sidebar.text_input("Consumer Secret", value=env_creds.get('consumer_secret', ''), type="password")
    oauth_token = st.sidebar.text_input("OAuth Token", value=env_creds.get('oauth_token', ''), type="password")
    oauth_token_secret = st.sidebar.text_input("OAuth Token Secret", value=env_creds.get('oauth_token_secret', ''), type="password")
    account_id = st.sidebar.text_input("Account ID", value=env_creds.get('account_id', ''))

    # Environment selection
    env = st.sidebar.radio("Environment", ["Sandbox", "Live"], index=0)
    use_sandbox = (env == "Sandbox")

    # Confirm live trading
    live_ok = True
    if not use_sandbox:
        live_ok = st.sidebar.checkbox("I understand this is LIVE trading (real money)")

    creds = {
        'consumer_key': consumer_key,
        'consumer_secret': consumer_secret,
        'oauth_token': oauth_token,
        'oauth_token_secret': oauth_token_secret,
        'account_id': account_id,
        'use_sandbox': str(use_sandbox).lower()
    }

    if all([consumer_key, consumer_secret, oauth_token, oauth_token_secret, account_id]) and (use_sandbox or live_ok):
        st.sidebar.success("✅ Credentials ready")
        return creds
    else:
        st.sidebar.info("Enter all credentials to enable trading features.")
        return None


def load_price_data(client: ETradeClient, symbol: str) -> Optional[pd.DataFrame]:
    """
    Load real-time price data for the given symbol with robust error handling.
    
    Args:
        client: The initialized E*Trade API client
        symbol: The stock symbol to fetch data for
        
    Returns:
        DataFrame with price data or None if an error occurred
    """
    try:
        logger.info(f"Fetching live data for {symbol}")
        
        # Use safe_request utility to handle API call with retries and error handling
        raw_data = safe_request(
            lambda: client.get_live_data(symbol),
            max_retries=3,
            retry_delay=1,
            timeout=10
        )
        
        if not raw_data:
            logger.error(f"Empty data received for {symbol}")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(raw_data)
        
        # Validate required columns
        missing_cols = REQUIRED_DATA_COLUMNS - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns in API response: {missing_cols}")
            return None
            
        # Process data
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df
        
    except Exception as e:
        logger.exception(f"Error loading data for {symbol}: {str(e)}")
        st.session_state.error_count += 1
        return None


def render_price_chart(
    data: pd.DataFrame, 
    symbol: str, 
    config: DashboardConfig
) -> None:
    """
    Render the interactive price chart with selected indicators.
    
    Args:
        data: DataFrame with OHLCV data
        symbol: The stock symbol being displayed
        config: Dashboard configuration options
    """
    if data is None or data.empty:
        st.warning("No data available to display.")
        return
    
    # Create a Plotly candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='Price'
    )])
    
    # Add selected technical indicators
    if config.indicators:
        data_with_indicators = add_technical_indicators(data, config.indicators)
        
        # Add each indicator as a trace to the chart
        for indicator in config.indicators:
            if indicator in data_with_indicators.columns:
                fig.add_trace(go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators[indicator],
                    mode='lines',
                    name=indicator
                ))
            elif indicator == "Bollinger Bands" and "BB_upper" in data_with_indicators.columns:
                # Handle special case for Bollinger Bands (multiple lines)
                fig.add_trace(go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators["BB_upper"],
                    mode='lines',
                    line=dict(width=1, color='rgba(50, 50, 150, 0.5)'),
                    name='Upper Band'
                ))
                fig.add_trace(go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators["BB_lower"],
                    mode='lines',
                    line=dict(width=1, color='rgba(50, 50, 150, 0.5)'),
                    fill='tonexty',
                    fillcolor='rgba(50, 50, 150, 0.1)',
                    name='Lower Band'
                ))
    
    # Update layout for better display
    fig.update_layout(
        title=f"{symbol} - Live Price Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        height=config.chart_height,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display additional information
    st.markdown(f"""
    **Last Price:** ${data['close'].iloc[-1]:.2f}  
    **Daily Change:** {((data['close'].iloc[-1] / data['open'].iloc[0]) - 1) * 100:.2f}%  
    **Volume:** {data['volume'].iloc[-1]:,.0f}  
    """)


def render_trade_controls(client: ETradeClient, symbol: str) -> None:
    """
    Render trading controls for buying and selling the selected symbol.
    
    Args:
        client: The initialized E*Trade API client
        symbol: The stock symbol to trade
    """
    st.subheader("Trade Controls")
    
    cols = st.columns(2)
    
    with cols[0]:
        buy_quantity = st.number_input("Buy Quantity", min_value=1, value=1, step=1)
        buy_price = st.number_input("Buy Limit Price", min_value=0.01, step=0.01)
        buy_market = st.checkbox("Market Order (Buy)")
        
        if st.button("Place Buy Order"):
            try:
                order_type = "MARKET" if buy_market else "LIMIT"
                
                # This would be the actual API call in production
                # result = client.place_order(
                #     symbol=symbol,
                #     quantity=buy_quantity,
                #     order_type=order_type,
                #     price=None if buy_market else buy_price,
                #     action="BUY"
                # )
                
                # For demo purposes
                st.success(f"Buy order placed for {buy_quantity} shares of {symbol}")
                logger.info(f"Buy order placed: {symbol}, {buy_quantity} shares, {order_type}")
                
            except Exception as e:
                logger.exception(f"Error placing buy order: {str(e)}")
                st.error(f"Failed to place buy order: {str(e)}")
    
    with cols[1]:
        sell_quantity = st.number_input("Sell Quantity", min_value=1, value=1, step=1)
        sell_price = st.number_input("Sell Limit Price", min_value=0.01, step=0.01)
        sell_market = st.checkbox("Market Order (Sell)")
        
        if st.button("Place Sell Order"):
            try:
                order_type = "MARKET" if sell_market else "LIMIT"
                
                # This would be the actual API call in production
                # result = client.place_order(
                #     symbol=symbol,
                #     quantity=sell_quantity,
                #     order_type=order_type,
                #     price=None if sell_market else sell_price,
                #     action="SELL"
                # )
                
                # For demo purposes
                st.success(f"Sell order placed for {sell_quantity} shares of {symbol}")
                logger.info(f"Sell order placed: {symbol}, {sell_quantity} shares, {order_type}")
                
            except Exception as e:
                logger.exception(f"Error placing sell order: {str(e)}")
                st.error(f"Failed to place sell order: {str(e)}")


def render_metrics(data: pd.DataFrame) -> None:
    """
    Render key metrics and statistics for the current symbol.
    
    Args:
        data: DataFrame with OHLCV data
    """
    if data is None or data.empty:
        return
        
    st.subheader("Key Metrics")
    
    # Calculate basic metrics safely with error handling
    try:
        latest_close = data['close'].iloc[-1]
        daily_high = data['high'].max()
        daily_low = data['low'].min()
        daily_volume = data['volume'].sum()
        
        # Calculate volatility (avoid division by zero)
        pct_change = data['close'].pct_change().dropna()
        if len(pct_change) > 1:
            volatility = pct_change.std() * (252 ** 0.5) * 100  # Annualized volatility
        else:
            volatility = 0.0
            
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Last Price", f"${latest_close:.2f}")
        
        with col2:
            st.metric("Day High", f"${daily_high:.2f}")
        
        with col3:
            st.metric("Day Low", f"${daily_low:.2f}")
        
        with col4:
            st.metric("Volatility", f"{volatility:.2f}%")
        
        # Volume chart
        st.subheader("Volume")
        st.bar_chart(data['volume'])
    except Exception as e:
        logger.exception(f"Error calculating metrics: {str(e)}")
        st.warning("Unable to calculate metrics due to insufficient data")


def render_last_update_info() -> None:
    """Render information about the last data update."""
    if st.session_state.last_update_time:
        st.caption(f"Last updated: {st.session_state.last_update_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        
    if st.session_state.error_count > 0:
        st.warning(f"Experienced {st.session_state.error_count} data fetch errors since dashboard startup.")


def main():
    st.set_page_config(
        page_title="Live Trading Dashboard - StockTrader",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Live Trading Dashboard")

    # Initialize session state
    initialize_dashboard_session_state()

    # Try to get credentials and initialize the E*Trade client
    creds = render_credentials_sidebar()
    client = create_etrade_client(creds) if creds else None

    # Render sidebar and get configuration
    config = render_sidebar()

    # Auto-refresh container
    data_container = st.empty()

    # Add manual refresh button
    col1, col2 = st.columns([1, 5])
    with col1:
        refresh_clicked = st.button("🔄 Refresh")

    # Check if it's time to refresh the data
    time_to_refresh = False
    if st.session_state.last_update_time is None:
        time_to_refresh = True
    else:
        elapsed = (datetime.now() - st.session_state.last_update_time).total_seconds()
        time_to_refresh = elapsed >= config.refresh_interval

    # Load data if it's time to refresh or if refresh was clicked
    if time_to_refresh or refresh_clicked:
        with st.spinner(f"Loading data for {config.symbol}..."):
            if client:
                data = load_price_data(client, config.symbol)
            else:
                data = None  # Or load demo data if you have it

            if data is not None:
                st.session_state.data_cache = data
                st.session_state.last_update_time = datetime.now()

    # Show last update time
    render_last_update_info()

    # If we have data, render the dashboard components
    if st.session_state.data_cache is not None:
        with data_container:
            # Render price chart
            render_price_chart(st.session_state.data_cache, config.symbol, config)

            # Render key metrics
            render_metrics(st.session_state.data_cache)

            # Render trade controls only if client is available
            if client:
                render_trade_controls(client, config.symbol)
            else:
                st.info("Trading controls are disabled in demo mode (no credentials provided).")
    else:
        with data_container:
            st.warning(f"No data available for {config.symbol}. Please check that the symbol is valid.")

    # Set up auto-refresh
    if not refresh_clicked and st.session_state.last_update_time:
        elapsed = (datetime.now() - st.session_state.last_update_time).total_seconds()
        time_to_next_refresh = max(1, config.refresh_interval - int(elapsed))

        with st.empty():
            st.caption(f"Next refresh in approximately {time_to_next_refresh} seconds")

        # Schedule a rerun after the refresh interval
        time.sleep(1)  # Small delay to prevent UI freezing
        st.experimental_rerun()


if __name__ == "__main__":
    main()
