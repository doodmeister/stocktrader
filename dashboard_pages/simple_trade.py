"""
Live Trading Dashboard
----------------------
Provides real-time stock price monitoring and trading capabilities via E*Trade API.
Features candlestick charting, technical indicators, and trade execution.

This module implements the main trading interface with:
- Real-time price data fetching with caching
- Interactive candlestick charts with technical indicators
- Trade execution controls with risk management
- ML-based trading signals
- Comprehensive error handling and logging

Part of the StockTrader application suite.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import threading

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pydantic import BaseModel, Field, validator, ValidationError

# Local imports
from core.etrade_client import ETradeClient
from core.etrade_auth_ui import (
    render_etrade_authentication
)
from utils.technicals.analysis import add_technical_indicators
from core.safe_requests import safe_request
from core.dashboard_utils import (
    initialize_dashboard_session_state,
    setup_page,
    handle_streamlit_error
)
from utils.live_inference import make_trade_decision
from patterns.pattern_utils import add_candlestick_pattern_features
from core.data_validator import validate_dataframe, validate_symbols

# Import SessionManager to solve button key conflicts and session state issues
from core.session_manager import create_session_manager

# Dashboard logger setup
from utils.logger import get_dashboard_logger
logger = get_dashboard_logger(__name__)

# Initialize the page (setup_page returns a logger, but we already have one)
setup_page(
    title="💼 Live Trading Dashboard",
    logger_name=__name__,
    sidebar_title="Trading Controls"
)
logger = get_dashboard_logger(__name__)

# Initialize SessionManager at module level to prevent button conflicts
_session_manager = create_session_manager("simple_trade")

# Constants
DEFAULT_SYMBOL = "AAPL"
DEFAULT_REFRESH_INTERVAL = 60  # seconds
MIN_REFRESH_INTERVAL = 10  # seconds
MAX_REFRESH_INTERVAL = 300  # 5 minutes
REQUIRED_DATA_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}
DEFAULT_CHART_HEIGHT = 500
MIN_CHART_HEIGHT = 300
MAX_CHART_HEIGHT = 1000
INDICATORS = ["SMA", "EMA", "MACD", "RSI", "Bollinger Bands"]
MAX_ERROR_COUNT = 10
CACHE_EXPIRY_MINUTES = 5


class DashboardConfig(BaseModel):
    """Configuration for the live trading dashboard with comprehensive validation."""
    
    symbol: str = Field(DEFAULT_SYMBOL, description="Stock ticker symbol")
    refresh_interval: int = Field(
        DEFAULT_REFRESH_INTERVAL,
        description="Data refresh interval in seconds",
        ge=MIN_REFRESH_INTERVAL,
        le=MAX_REFRESH_INTERVAL
    )
    indicators: List[str] = Field(
        default=[],
        description="Technical indicators to display on the chart"
    )
    chart_height: int = Field(
        DEFAULT_CHART_HEIGHT,
        description="Height of the main chart in pixels",
        ge=MIN_CHART_HEIGHT,
        le=MAX_CHART_HEIGHT
    )
    enable_trading: bool = Field(
        False,
        description="Whether trading controls are enabled"
    )
    max_order_value: float = Field(
        10000.0,
        description="Maximum allowed order value in USD",
        gt=0
    )
    
    @validator('symbol')
    def validate_symbol_format(cls, v):
        """Ensure symbol is valid using the centralized validator."""
        try:
            result = validate_symbols(v)
            if not result.is_valid:
                raise ValueError(f"Invalid symbol format: {result.errors}")
            return v
        except Exception as e:
            raise ValueError(f"Validation error: {e}")
    
    @validator('indicators')
    def validate_indicators(cls, v):
        """Ensure all indicators are supported."""
        invalid = [ind for ind in v if ind not in INDICATORS]
        if invalid:
            raise ValueError(f"Unsupported indicators: {invalid}")
        return v


class TradingSession:
    """Manages trading session state and data caching."""
    
    def __init__(self):
        self.data_cache: Optional[pd.DataFrame] = None
        self.last_update_time: Optional[datetime] = None
        self.error_count: int = 0
        self.client: Optional[ETradeClient] = None
        self._lock = threading.Lock()
    
    def is_cache_valid(self) -> bool:
        """Check if cached data is still valid."""
        if self.data_cache is None or self.last_update_time is None:
            return False
        
        cache_age = datetime.now() - self.last_update_time
        return cache_age < timedelta(minutes=CACHE_EXPIRY_MINUTES)
    
    def update_data(self, data: pd.DataFrame) -> None:
        """Thread-safely update cached data."""
        with self._lock:
            self.data_cache = data.copy()
            self.last_update_time = datetime.now()
    
    def increment_error_count(self) -> None:
        """Thread-safely increment error count."""
        with self._lock:
            self.error_count += 1
    
    def reset_error_count(self) -> None:
        """Thread-safely reset error count."""
        with self._lock:
            self.error_count = 0


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables with comprehensive setup."""
    if 'trading_session' not in st.session_state:
        st.session_state.trading_session = TradingSession()
    
    if 'dashboard_initialized' not in st.session_state:
        initialize_dashboard_session_state()
        st.session_state.dashboard_initialized = True
    
    # Initialize other required state variables
    state_defaults = {
        'login_status': False,
        'last_symbol': DEFAULT_SYMBOL,
        'demo_mode': True,
        'trading_enabled': False,
        'risk_warnings_acknowledged': False
    }
    
    for key, default_value in state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def render_sidebar() -> Optional[DashboardConfig]:
    """
    Render the sidebar controls and return the configuration.
    
    Returns:
        DashboardConfig: The dashboard configuration from user inputs, or None if invalid
    """
    st.sidebar.header("📊 Dashboard Settings")
    
    try:
        # Symbol input with validation
        symbol_input = st.sidebar.text_input(
            "Symbol", 
            value=st.session_state.get('last_symbol', DEFAULT_SYMBOL),
            help="Enter a valid stock ticker symbol (e.g., AAPL, MSFT)"
        ).strip().upper()
        
        # Refresh interval
        refresh_interval = st.sidebar.number_input(
            "Refresh Interval (seconds)",
            min_value=MIN_REFRESH_INTERVAL,
            max_value=MAX_REFRESH_INTERVAL,
            value=DEFAULT_REFRESH_INTERVAL,
            help=f"Data refresh interval between {MIN_REFRESH_INTERVAL}-{MAX_REFRESH_INTERVAL} seconds"
        )
        
        # Technical indicators section
        st.sidebar.subheader("📈 Technical Indicators")
        selected_indicators = []
        
        indicator_cols = st.sidebar.columns(2)
        for i, indicator in enumerate(INDICATORS):
            col = indicator_cols[i % 2]
            if col.checkbox(indicator, value=False, key=f"indicator_{indicator}"):
                selected_indicators.append(indicator)
        
        # Chart settings
        st.sidebar.subheader("🎨 Chart Settings")
        chart_height = st.sidebar.slider(
            "Chart Height", 
            min_value=MIN_CHART_HEIGHT, 
            max_value=MAX_CHART_HEIGHT, 
            value=DEFAULT_CHART_HEIGHT,
            step=50,
            help="Adjust the height of the price chart"
        )
        
        # Trading settings
        st.sidebar.subheader("💰 Trading Settings")
        enable_trading = st.sidebar.checkbox(
            "Enable Trading Controls",
            value=False,
            help="Enable actual trade execution (requires valid credentials)"
        )
        
        max_order_value = st.sidebar.number_input(
            "Max Order Value ($)",
            min_value=100.0,
            max_value=100000.0,
            value=10000.0,
            step=100.0,
            help="Maximum allowed value per trade order"
        )
        
        # Create and validate configuration
        config = DashboardConfig(
            symbol=symbol_input,
            refresh_interval=int(refresh_interval),
            indicators=selected_indicators,
            chart_height=int(chart_height),
            enable_trading=enable_trading,
            max_order_value=max_order_value
        )
        
        # Update last symbol in session state
        st.session_state.last_symbol = config.symbol
        
        return config
        
    except ValidationError as e:
        st.sidebar.error(f"Configuration error: {e}")
        logger.error(f"Dashboard configuration validation failed: {e}")
        return None
    except Exception as e:
        st.sidebar.error(f"Unexpected error in sidebar: {e}")
        logger.exception("Unexpected error in render_sidebar")
        return None


def render_credentials_sidebar() -> Optional[ETradeClient]:
    """
    Render E*Trade authentication using the secure authentication UI.
    
    Returns:
        ETradeClient if authenticated, None if not authenticated
    """
    # Use the secure E*Trade authentication UI following the same pattern as advanced_ai_trade.py
    return render_etrade_authentication()


def load_price_data(client: ETradeClient, symbol: str, session: TradingSession) -> Optional[pd.DataFrame]:
    """Load real-time price data with comprehensive error handling and caching."""
    try:
        logger.info(f"Fetching live data for {symbol}")
        
        # Check if cached data is still valid to avoid unnecessary API calls
        if session.is_cache_valid() and session.data_cache is not None:
            if len(session.data_cache) > 0:
                last_symbol = session.data_cache.get('symbol', {}).get(0, '')
                if last_symbol == symbol:
                    logger.debug(f"Using cached data for {symbol}")
                    return session.data_cache
        
        # Fetch new data with retry mechanism
        raw_data = safe_request(
            lambda: client.get_live_data(symbol),
            max_retries=3,
            retry_delay=2,
            timeout=15
        )
        
        if not raw_data:
            logger.warning(f"Empty data received for {symbol}")
            session.increment_error_count()
            return None
        
        # Convert to DataFrame with validation
        df = pd.DataFrame(raw_data)
        
        # Run centralized validation
        validation_result = validate_dataframe(
            df,
            required_cols=list(REQUIRED_DATA_COLUMNS),
            validate_ohlc=True,
            check_statistical_anomalies=True
        )

        if not validation_result.is_valid:
            error_message = "; ".join(validation_result.errors)
            logger.error(f"Data validation failed for {symbol}: {error_message}")
            session.increment_error_count()
            return None
            
        # Log any warnings
        if validation_result.warnings:
            for warning in validation_result.warnings:
                logger.warning(f"Data warning: {warning}")

        # Process the data
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        df.set_index('timestamp', inplace=True)
        
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove invalid rows
        df = df.dropna(subset=numeric_cols)
        
        if df.empty:
            logger.error(f"No valid data rows after processing for {symbol}")
            session.increment_error_count()
            return None
        
        # Remove redundant OHLC validation since it's already handled by validate_dataframe
        
        # Add symbol for caching validation
        df['symbol'] = symbol
        
        # Update session cache
        session.update_data(df)
        session.reset_error_count()
        
        logger.info(f"Successfully loaded {len(df)} data points for {symbol}")
        return df
            
    except Exception as e:
        logger.exception(f"Critical error loading data for {symbol}: {e}")
        session.increment_error_count()
        return None


def render_price_chart(data: pd.DataFrame, symbol: str, config: DashboardConfig) -> None:
    """
    Render an interactive price chart with technical indicators and enhanced features.
    
    Args:
        data: DataFrame with OHLCV data
        symbol: The stock symbol being displayed
        config: Dashboard configuration options
    """
    if data is None or data.empty:
        st.warning("📊 No price data available to display")
        return
    
    try:
        # Create base candlestick chart
        fig = go.Figure()
        
        # Add candlestick trace
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ))
        
        # Add technical indicators if selected
        if config.indicators:
            try:
                data_with_indicators = add_technical_indicators(data, config.indicators)
                
                # Color scheme for indicators
                indicator_colors = {
                    'SMA': '#ff9500',
                    'EMA': '#0099ff',
                    'MACD': '#9d4edd',
                    'RSI': '#f72585'
                }
                
                for indicator in config.indicators:
                    color = indicator_colors.get(indicator, '#888888')
                    
                    if indicator == "Bollinger Bands":
                        # Handle Bollinger Bands specially
                        if all(col in data_with_indicators.columns for col in ["BB_upper", "BB_lower", "BB_middle"]):
                            fig.add_trace(go.Scatter(
                                x=data_with_indicators.index,
                                y=data_with_indicators["BB_upper"],
                                mode='lines',
                                line=dict(width=1, color='rgba(100, 100, 200, 0.8)'),
                                name='BB Upper',
                                hovertemplate='BB Upper: $%{y:.2f}<extra></extra>'
                            ))
                            fig.add_trace(go.Scatter(
                                x=data_with_indicators.index,
                                y=data_with_indicators["BB_lower"],
                                mode='lines',
                                line=dict(width=1, color='rgba(100, 100, 200, 0.8)'),
                                fill='tonexty',
                                fillcolor='rgba(100, 100, 200, 0.1)',
                                name='BB Lower',
                                hovertemplate='BB Lower: $%{y:.2f}<extra></extra>'
                            ))
                            fig.add_trace(go.Scatter(
                                x=data_with_indicators.index,
                                y=data_with_indicators["BB_middle"],
                                mode='lines',
                                line=dict(width=1, color='rgba(100, 100, 200, 1.0)', dash='dash'),
                                name='BB Middle',
                                hovertemplate='BB Middle: $%{y:.2f}<extra></extra>'
                            ))
                    else:
                        # Handle regular indicators
                        if indicator in data_with_indicators.columns:
                            fig.add_trace(go.Scatter(
                                x=data_with_indicators.index,
                                y=data_with_indicators[indicator],
                                mode='lines',
                                line=dict(width=2, color=color),
                                name=indicator,
                                hovertemplate=f'{indicator}: %{{y:.2f}}<extra></extra>'
                            ))
            except Exception as e:
                logger.warning(f"Error adding technical indicators: {e}")
                st.warning("⚠️ Some technical indicators could not be displayed")
        
        # Enhanced layout configuration
        fig.update_layout(
            title={
                'text': f"{symbol} - Live Price Chart",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#ffffff'}
            },
            xaxis_title="Time",
            yaxis_title="Price ($)",
            height=config.chart_height,
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='x unified',
            template='plotly_dark',
            showlegend=True
        )
        
        # Enhanced axes formatting
        fig.update_xaxes(
            gridcolor='rgba(128, 128, 128, 0.3)',
            showgrid=True
        )
        fig.update_yaxes(
            gridcolor='rgba(128, 128, 128, 0.3)',
            showgrid=True,
            tickformat='$.2f'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display summary statistics
        render_price_summary(data, symbol)
        
    except Exception as e:
        logger.exception(f"Error rendering price chart: {e}")
        st.error("❌ Failed to render price chart")


def render_price_summary(data: pd.DataFrame, symbol: str) -> None:
    """Render price summary statistics."""
    try:
        latest_close = data['close'].iloc[-1]
        prev_close = data['close'].iloc[-2] if len(data) > 1 else latest_close
        daily_change = latest_close - prev_close
        daily_change_pct = (daily_change / prev_close) * 100 if prev_close != 0 else 0
        
        daily_high = data['high'].max()
        daily_low = data['low'].min()
        volume = data['volume'].iloc[-1] if not data['volume'].empty else 0
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Last Price", 
                f"${latest_close:.2f}",
                delta=f"{daily_change:+.2f} ({daily_change_pct:+.2f}%)"
            )
        
        with col2:
            st.metric("Day High", f"${daily_high:.2f}")
        
        with col3:
            st.metric("Day Low", f"${daily_low:.2f}")
        
        with col4:
            st.metric("Volume", f"{volume:,.0f}")
            
    except Exception as e:
        logger.warning(f"Error rendering price summary: {e}")


def render_trade_controls(client: ETradeClient, symbol: str, config: DashboardConfig) -> None:
    """
    Render trading controls with enhanced risk management and validation.
    
    Args:
        client: The initialized E*Trade API client
        symbol: The stock symbol to trade
        config: Dashboard configuration for limits
    """
    st.subheader("💼 Trade Controls")
    
    if not config.enable_trading:
        st.info("🔒 Trading controls are disabled. Enable in sidebar to trade.")
        return
    
    if client is None:
        st.warning("⚠️ Trading requires valid API credentials")
        return
    
    try:        # Risk warning
        if not st.session_state.get('risk_warnings_acknowledged', False):
            st.warning("⚠️ **RISK WARNING**: Trading involves substantial risk of loss")
            if _session_manager.create_checkbox("I acknowledge the risks of trading", "risk_acknowledge"):
                st.session_state.risk_warnings_acknowledged = True
            else:
                return
          # Trading interface
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Buy Order")
            buy_quantity = _session_manager.create_number_input(
                "Quantity", 
                min_value=1, 
                max_value=10000, 
                value=1, 
                step=1,
                number_input_name="buy_qty"
            )
            
            buy_order_type = _session_manager.create_radio(
                "Order Type", 
                options=["Market", "Limit"], 
                radio_name="buy_type"
            )
            
            buy_price = None
            if buy_order_type == "Limit":
                buy_price = _session_manager.create_number_input(
                    "Limit Price ($)", 
                    min_value=0.01, 
                    step=0.01,
                    number_input_name="buy_price"
                )
            
            # Calculate order value
            estimated_price = buy_price if buy_price else (
                st.session_state.trading_session.data_cache['close'].iloc[-1] 
                if st.session_state.trading_session.data_cache is not None else 0
            )
            order_value = buy_quantity * estimated_price
            st.info(f"Estimated Order Value: ${order_value:,.2f}")
            
            if order_value > config.max_order_value:
                st.error(f"❌ Order value exceeds limit of ${config.max_order_value:,.2f}")
            else:
                if _session_manager.create_button("🛒 Place Buy Order", type="primary"):
                    place_order(client, symbol, buy_quantity, "BUY", buy_order_type, buy_price)
        
        with col2:
            st.markdown("#### 📉 Sell Order")
            sell_quantity = _session_manager.create_number_input(
                "Quantity", 
                min_value=1, 
                max_value=10000, 
                value=1, 
                step=1,
                number_input_name="sell_qty"
            )
            
            sell_order_type = _session_manager.create_radio(
                "Order Type", 
                options=["Market", "Limit"], 
                radio_name="sell_type"
            )
            
            sell_price = None
            if sell_order_type == "Limit":
                sell_price = _session_manager.create_number_input(
                    "Limit Price ($)", 
                    min_value=0.01, 
                    step=0.01,
                    number_input_name="sell_price"
                )
            
            # Calculate order value
            estimated_price = sell_price if sell_price else (
                st.session_state.trading_session.data_cache['close'].iloc[-1] 
                if st.session_state.trading_session.data_cache is not None else 0
            )
            order_value = sell_quantity * estimated_price
            st.info(f"Estimated Order Value: ${order_value:,.2f}")
            
            if order_value > config.max_order_value:
                st.error(f"❌ Order value exceeds limit of ${config.max_order_value:,.2f}")
            else:
                if _session_manager.create_button("💰 Place Sell Order", type="primary"):
                    place_order(client, symbol, sell_quantity, "SELL", sell_order_type, sell_price)
        
    except Exception as e:
        logger.exception(f"Error rendering trade controls: {e}")
        st.error("❌ Error in trade controls interface")


def place_order(
    client: ETradeClient, 
    symbol: str, 
    quantity: int, 
    action: str, 
    order_type: str, 
    price: Optional[float] = None
) -> None:
    """
    Place a trading order with comprehensive error handling and production safeguards.
    
    IMPORTANT: This function currently contains simulated order placement.
    Remove the DEMO_MODE check and implement actual E*Trade API calls before production use.
    """
    # Production safety check - prevent accidental real trades during development
    DEMO_MODE = True  # TODO: Set to False and implement real API calls for production
    
    try:
        # Validate inputs before proceeding
        if quantity <= 0:
            st.error("❌ Quantity must be positive")
            return
        
        if order_type == "Limit" and (price is None or price <= 0):
            st.error("❌ Limit orders require a valid price")
            return
          # Additional safety check for large orders
        if quantity > 1000:  # Configurable threshold
            st.warning("⚠️ Large order detected - please confirm")
            if not _session_manager.create_checkbox(f"I confirm this {quantity} share order", f"large_order_{action}"):
                return
        
        # Show detailed order confirmation to prevent accidental trades
        order_summary = f"""
        **Order Summary:**
        - Symbol: {symbol}
        - Action: {action}
        - Quantity: {quantity:,} shares
        - Type: {order_type}
        - Price: {"Market" if order_type == "Market" else f"${price:.2f}"}        """
        
        st.info(order_summary)
        
        if _session_manager.create_button(f"Confirm {action} Order"):
            with st.spinner("Placing order..."):
                if DEMO_MODE:
                    # Simulated order placement - safe for development/testing
                    st.warning("🚧 DEMO MODE: No real orders are being placed")
                    result = {
                        'order_id': f'DEMO_ORD_{int(time.time())}',
                        'status': 'SIMULATED',
                        'symbol': symbol,
                        'quantity': quantity,
                        'action': action,
                        'type': order_type,
                        'price': price,
                        'note': 'This is a simulated order for testing purposes'
                    }
                else:
                    # TODO: Implement actual E*Trade API order placement
                    # Example: result = client.place_order(symbol, quantity, action, order_type, price)
                    raise NotImplementedError(
                        "Real order placement not implemented. "
                        "Implement E*Trade API integration before production use."
                    )
                
                st.success(f"✅ {action} order {'simulated' if DEMO_MODE else 'placed'} successfully!")
                st.json(result)
                
                logger.info(f"Order {'simulated' if DEMO_MODE else 'placed'}: {action} {quantity} shares of {symbol} at {order_type}")
                
    except Exception as e:
        logger.exception(f"Error placing {action} order: {e}")
        st.error(f"❌ Failed to place {action} order: {str(e)}")


def render_model_inference(data: pd.DataFrame, symbol: str) -> None:
    """
    Render ML model inference with enhanced error handling and display.
    
    Args:
        data: OHLCV DataFrame
        symbol: Stock symbol
    """
    st.subheader("🧠 AI Trading Signal")
    
    if data is None or data.empty:
        st.warning("📊 No data available for model inference")
        return
    
    try:
        with st.spinner("Computing AI signal..."):
            # Find latest model
            model_dir = Path("models/")
            if not model_dir.exists():
                st.warning("📁 Models directory not found")
                return
            
            model_files = list(model_dir.glob("pattern_nn_*.pth"))
            if not model_files:
                st.warning("🤖 No trained models found")
                return
            
            # Get the most recent model
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            preproc_path = latest_model.with_name(latest_model.stem + "_preprocessing.json")
            
            if not preproc_path.exists():
                st.warning("⚙️ Model preprocessing config not found")
                return
            
            # Prepare data for inference
            df_for_inference = data.reset_index().copy()
            df_enhanced = add_candlestick_pattern_features(df_for_inference)
            
            # Get prediction
            decision = make_trade_decision(
                df=df_enhanced,
                preprocessing_path=str(preproc_path),
                model_dir=str(model_dir),
                seq_len=10
            )
            
            # Display result with styling
            decision_emoji = {
                'buy': '📈',
                'sell': '📉', 
                'hold': '⏸️'
            }
            
            decision_color = {
                'buy': '#00ff88',
                'sell': '#ff4444',
                'hold': '#ffaa00'
            }
            
            emoji = decision_emoji.get(decision.lower(), '❓')
            color = decision_color.get(decision.lower(), '#888888')
            
            st.markdown(f"""
            <div style="
                padding: 1rem;
                border-radius: 0.5rem;
                background-color: {color}22;
                border-left: 4px solid {color};
                margin: 1rem 0;
            ">
                <h4 style="color: {color}; margin: 0;">
                    {emoji} Model Recommendation: <strong>{decision.upper()}</strong>
                </h4>
                <p style="margin: 0.5rem 0 0 0; color: #888;">
                    Based on pattern analysis and technical indicators
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Model metadata
            model_info = {
                'Model': latest_model.name,
                'Last Modified': datetime.fromtimestamp(latest_model.stat().st_mtime).strftime('%Y-%m-%d %H:%M'),
                'Data Points': len(df_enhanced),
                'Symbol': symbol
            }
            
            with st.expander("🔍 Model Details"):
                for key, value in model_info.items():
                    st.text(f"{key}: {value}")
            
    except Exception as e:
        logger.exception(f"Error in model inference: {e}")
        st.error("❌ Unable to compute AI trading signal")
        st.text(f"Error details: {str(e)}")


def render_system_status(session: TradingSession) -> None:
    """
    Render system status and health information.
    
    Args:
        session: Trading session for status information
    """
    with st.expander("🔧 System Status"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if session.last_update_time:
                time_since_update = datetime.now() - session.last_update_time
                st.metric("Last Update", f"{time_since_update.seconds}s ago")
            else:
                st.metric("Last Update", "Never")
        
        with col2:
            error_status = "🟢 Good" if session.error_count == 0 else f"🔴 {session.error_count} errors"
            st.metric("Error Count", str(session.error_count))
        
        with col3:
            cache_status = "🟢 Valid" if session.is_cache_valid() else "🔴 Expired"
            st.metric("Cache Status", cache_status)
        
        if session.error_count > MAX_ERROR_COUNT:
            st.error("⚠️ High error count detected. Consider restarting the application.")


class SimpleTradeDashboard:
    def __init__(self):
        # Initialize SessionManager to prevent button conflicts and state issues
        self.session_manager = create_session_manager("simple_trade")
    
    def run(self):
        """Main dashboard application entry point."""
        try:
            # Initialize application
            # st.set_page_config(  # Handled by main dashboard
            #     page_title="Live Trading Dashboard - StockTrader",
            #     page_icon="📊",
            #     layout="wide",
            #     initial_sidebar_state="expanded"
            # )
            
            # Initialize session state and get configuration
            initialize_session_state()
            session = st.session_state.trading_session
            
            # Render main header
            st.title("📊 Live Trading Dashboard")
            st.markdown("Real-time market data, technical analysis, and AI-powered trading signals")
              # Get user configuration from sidebar
            client = render_credentials_sidebar()
            config = render_sidebar()
            
            if config is None:
                st.error("❌ Invalid dashboard configuration")
                return
            
            # Handle data refresh logic
            auto_refresh_container = st.empty()
            was_manual_refresh = handle_refresh_logic(session, config, client)
            
            # Render main dashboard content
            render_main_content(session, config, client)
            
            # Display system status information
            render_system_status(session)
            
            # Handle auto-refresh countdown and scheduling
            with auto_refresh_container:
                setup_auto_refresh_display(session, config, was_manual_refresh)
            
        except Exception as e:
            _handle_critical_error(e)


def _handle_critical_error(error: Exception) -> None:
    """Handle critical application errors with recovery options."""
    logger.exception("Critical error in main application")
    st.error("❌ Critical application error occurred")
    st.text(f"Error: {str(error)}")
      # Provide recovery options for users
    st.markdown("### 🔧 Recovery Options")
    if _session_manager.create_button("🔄 Reset Application State"):
        # Clear all session state to recover from corrupted state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


def handle_refresh_logic(session: TradingSession, config: DashboardConfig, client: Optional[ETradeClient]) -> bool:
    """
    Handle data refresh logic and determine if refresh is needed.
    
    Args:
        session: Trading session for state management
        config: Dashboard configuration
        client: E*Trade client (optional)
        
    Returns:
        bool: True if data was refreshed, False otherwise
    """    # Manual refresh button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        manual_refresh = _session_manager.create_button("🔄 Refresh Data", use_container_width=True)
    
    # Determine if refresh is needed based on cache age
    should_refresh = False
    if session.last_update_time is None:
        # No previous data - force refresh
        should_refresh = True
    else:
        # Check if cache has expired based on configured interval
        elapsed = (datetime.now() - session.last_update_time).total_seconds()
        should_refresh = elapsed >= config.refresh_interval
    
    # Load data if refresh is needed or manually requested
    if should_refresh or manual_refresh:
        if client:
            with st.spinner(f"Loading data for {config.symbol}..."):
                data = load_price_data(client, config.symbol, session)
                if data is not None:
                    st.success(f"✅ Data loaded for {config.symbol}")
                    return True
                else:
                    st.warning(f"⚠️ Failed to load data for {config.symbol}")
        else:
            st.info("🔗 Demo mode: Connect API credentials for live data")
    
    return manual_refresh


def render_main_content(session: TradingSession, config: DashboardConfig, client: Optional[ETradeClient]) -> None:
    """
    Render the main dashboard content including charts, AI signals, and trading controls.
    
    Args:
        session: Trading session for cached data
        config: Dashboard configuration
        client: E*Trade client (optional)
    """
    if session.data_cache is not None and not session.data_cache.empty:
        # Render price chart with technical indicators
        render_price_chart(session.data_cache, config.symbol, config)
        
        # Display AI-powered trading signals
        render_model_inference(session.data_cache, config.symbol)
        
        # Show trading controls if enabled and client available
        if client and config.enable_trading:
            render_trade_controls(client, config.symbol, config)
        elif config.enable_trading:
            st.info("🔐 Connect API credentials to enable trading")
    else:
        st.warning(f"📊 No data available for {config.symbol}")
        if client is None:
            st.info("💡 Connect to E*Trade API to fetch live market data")


def setup_auto_refresh_display(session: TradingSession, config: DashboardConfig, was_manual_refresh: bool) -> None:
    """
    Handle auto-refresh countdown display and scheduling.
    
    Args:
        session: Trading session for timing information
        config: Dashboard configuration for refresh interval
        was_manual_refresh: Whether the last action was a manual refresh
    """
    # Only show countdown if we have a last update time and it wasn't a manual refresh
    if session.last_update_time and not was_manual_refresh:
        elapsed = (datetime.now() - session.last_update_time).total_seconds()
        remaining = max(0, config.refresh_interval - elapsed)
        
        if remaining > 0:
            # Display countdown to next auto-refresh
            st.caption(f"⏱️ Next refresh in {int(remaining)} seconds")
              # Schedule rerun for auto-refresh (Streamlit-specific)
            time.sleep(1)
            st.rerun()


# Execute the main function
if __name__ == "__main__":
    try:
        dashboard = SimpleTradeDashboard()
        dashboard.run()
    except Exception as e:
        handle_streamlit_error(e, "Simple Trade Dashboard")
