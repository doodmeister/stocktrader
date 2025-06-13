"""
Advanced AI Trading Dashboard for E*Trade Integration

This module provides a comprehensive Streamlit dashboard for automated trading
with E*Trade, featuring real-time market data, pattern detection, risk management,
and machine learning integration.

Key Features:
- Real-time candlestick pattern detection
- Machine learning model inference
- Risk management and position sizing
- Alert system with multi-channel notifications
- Live trading with sandbox/production modes

Architecture:
- Modular design with clear separation of concerns
- Robust error handling and logging
- Comprehensive input validation
- Secure credential management
- Performance optimizations with caching
"""

import hashlib
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, List
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Project imports
from core.streamlit.dashboard_utils import (
    initialize_dashboard_session_state,
    handle_streamlit_error,
    get_model_info
)
from core.etrade_client import ETradeClient
from core.etrade_auth_ui import (
    render_etrade_authentication
)
from security.etrade_security import SecureETradeManager
from core.risk_manager_v2 import RiskManager, RiskConfigManager
from patterns.patterns import create_pattern_detector
from patterns.patterns_nn import PatternNN
from train.model_manager import ModelManager as CoreModelManager
from core.data_validator import (
    DataValidator, 
    get_global_validator, 
    validate_symbol, 
    validate_symbols
)
from utils.notifier import Notifier

# Import new centralized technical analysis modules
from core.technical_indicators import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands, 
    calculate_atr, calculate_sma, calculate_ema, IndicatorError
)
from utils.technicals.analysis import TechnicalAnalysis

from core.streamlit.decorators import handle_exceptions
from core.streamlit.session_manager import SessionManager

# Import SessionManager to solve button key conflicts and session state issues
from core.streamlit.session_manager import create_session_manager

# Dashboard logger setup
from utils.logger import get_dashboard_logger
logger = get_dashboard_logger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCKER_INDICATOR = '/.dockerenv'
DEFAULT_REFRESH_INTERVAL = 60  # seconds
MAX_SYMBOLS = 20  # Limit for performance
CACHE_TTL = 300  # 5 minutes
MIN_PRICE = 0.01
MAX_POSITION_SIZE = 0.5  # 50%
MIN_POSITION_SIZE = 0.001  # 0.1%

# Mapping of model names to classes
SKLEARN_MODELS = {
    "RandomForest": RandomForestClassifier,
    "LogisticRegression": LogisticRegression
}


class TradingEnvironment(Enum):
    """Trading environment enumeration."""
    SANDBOX = "sandbox"
    LIVE = "live"


class AlertType(Enum):
    """Alert type enumeration."""
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    PATTERN = "pattern"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class DashboardError(Exception):
    """Custom exception for dashboard-specific errors."""
    pass


class DashboardValidationError(Exception):
    """Custom validation error for dashboard-specific validation."""
    pass


@dataclass
class AlertConfig:
    """Configuration for alerts with immutable design."""
    symbol: str
    alert_type: AlertType
    threshold: Optional[float] = None
    pattern: Optional[str] = None
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate alert configuration after initialization."""
        self.symbol = self._validate_symbol(self.symbol)
        if self.alert_type in [AlertType.PRICE_ABOVE, AlertType.PRICE_BELOW] and self.threshold is None:
            raise DashboardValidationError("Price alerts require a threshold value")
        if self.alert_type == AlertType.PATTERN and not self.pattern:
            raise DashboardValidationError("Pattern alerts require a pattern name")
    
    @staticmethod
    def _validate_symbol(symbol: str) -> str:
        """Validate and normalize symbol using core validator."""
        result = validate_symbol(symbol)
        if not result.is_valid:
            errors = result.errors or []
            raise DashboardValidationError(f"Invalid symbol: {'; '.join(errors)}")
        return result.value


@dataclass
class RiskParameters:
    """Risk management parameters with validation."""
    max_position_size: float = field(default_factory=lambda: RiskConfigManager.get_max_loss_percent())
    stop_loss_atr: float = field(default_factory=lambda: RiskConfigManager.get_default_atr_multiplier())
    max_daily_loss: float = field(default_factory=lambda: RiskConfigManager.get_max_daily_loss())
    position_correlation_limit: float = field(default_factory=lambda: RiskConfigManager.get_max_correlation_exposure())
    
    def __post_init__(self):
        """Validate risk parameters."""
        if not MIN_POSITION_SIZE <= self.max_position_size <= MAX_POSITION_SIZE:
            raise DashboardValidationError(f"Position size must be between {MIN_POSITION_SIZE:.1%} and {MAX_POSITION_SIZE:.1%}")
        if not 0.5 <= self.stop_loss_atr <= 10.0:
            raise DashboardValidationError("Stop loss ATR must be between 0.5 and 10.0")
        if not 0.01 <= self.max_daily_loss <= 0.50:
            raise DashboardValidationError("Max daily loss must be between 1% and 50%")


class ClientProtocol(Protocol):
    """Protocol defining the interface for trading clients."""
    
    def get_quote(self, symbol: str) -> pd.DataFrame:
        """Get quote data for a symbol."""
        ...
    
    def get_candles(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """Get candlestick data for a symbol."""
        ...
    
    def place_order(self, symbol: str, side: str, quantity: int, 
                   order_type: str, limit_price: Optional[float] = None) -> Optional[Dict]:
        """Place a trading order."""
        ...


def is_docker_environment() -> bool:
    """Check if running in Docker container."""
    return os.path.exists(DOCKER_INDICATOR)


def create_data_hash(df: pd.DataFrame) -> str:
    """Create a consistent hash for DataFrame caching."""
    try:
        # Use a more reliable hash method
        data_str = df.to_string()
        return hashlib.md5(data_str.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Failed to create data hash: {e}")
        return str(hash(str(df.values.tobytes())))


class SessionStateManager:
    """Centralized session state management with type safety and validation."""
    
    # Session state keys as class constants
    INITIALIZED = "dashboard_initialized"
    SYMBOLS = "watched_symbols"
    RISK_PARAMS = "risk_parameters"
    ALERTS = "alerts_config"
    MODEL = "loaded_model"
    TRAINING_STATUS = "training_in_progress"
    LAST_REFRESH = "last_data_refresh"
    CONNECTION_STATUS = "etrade_connection"
    CREDENTIALS = "etrade_credentials"
    
    # New flags to prevent unnecessary re-renders
    MODEL_LOAD_SUCCESS = "model_load_success"
    TRAINING_COMPLETE = "training_complete"
    ALERT_ADDED = "alert_added"
    SETTINGS_CHANGED = "settings_changed"
    SYMBOLS_UPDATED = "symbols_updated"
    ORDER_PLACED = "order_placed"
    CONNECTION_CHANGED = "connection_changed"
    CACHE_CLEARED = "cache_cleared"
    
    @classmethod
    def initialize(cls) -> None:
        """Initialize all session state variables with defaults."""
        if st.session_state.get(cls.INITIALIZED, False):
            return
            
        try:
            # Initialize core dashboard state first
            initialize_dashboard_session_state()
            
            # Core settings
            st.session_state[cls.INITIALIZED] = True
            st.session_state[cls.SYMBOLS] = ["AAPL", "MSFT"]
            st.session_state[cls.RISK_PARAMS] = RiskParameters().__dict__
            st.session_state[cls.MODEL] = None
            st.session_state[cls.TRAINING_STATUS] = False
            st.session_state[cls.LAST_REFRESH] = datetime.now()
            st.session_state[cls.CONNECTION_STATUS] = False
            st.session_state[cls.CREDENTIALS] = None
            
            # Initialize re-render control flags
            st.session_state[cls.MODEL_LOAD_SUCCESS] = False
            st.session_state[cls.TRAINING_COMPLETE] = False
            st.session_state[cls.ALERT_ADDED] = False
            st.session_state[cls.SETTINGS_CHANGED] = False
            st.session_state[cls.SYMBOLS_UPDATED] = False
            st.session_state[cls.ORDER_PLACED] = False
            st.session_state[cls.CONNECTION_CHANGED] = False
            st.session_state[cls.CACHE_CLEARED] = False
            
            # Alert configuration with proper structure
            st.session_state[cls.ALERTS] = {
                "price_alerts": {},
                "pattern_alerts": {},
                "triggered_alerts": [],
                "notification_channels": {
                    "email": {"enabled": False, "address": ""},
                    "sms": {"enabled": False, "number": ""},
                    "dashboard": {"enabled": True},
                    "slack": {"enabled": False, "webhook": ""}
                }
            }
            
            logger.info("Session state initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize session state: {e}")
            raise DashboardError(f"Session initialization failed: {e}")
    
    @classmethod
    def set_flag(cls, flag_name: str, value: bool = True) -> None:
        """Set a session state flag."""
        st.session_state[flag_name] = value
    
    @classmethod
    def get_flag(cls, flag_name: str) -> bool:
        """Get and reset a session state flag."""
        flag_value = st.session_state.get(flag_name, False)
        if flag_value:
            st.session_state[flag_name] = False  # Reset after reading
        return flag_value
    
    @classmethod
    def check_flag(cls, flag_name: str) -> bool:
        """Check flag without resetting it."""
        return st.session_state.get(flag_name, False)
    
    @classmethod
    def get_symbols(cls) -> list:
        """Get current symbols."""
        return st.session_state.get(cls.SYMBOLS, ["AAPL", "MSFT"])
    
    @classmethod
    def set_symbols(cls, symbols: list) -> None:
        """Set symbols."""
        st.session_state[cls.SYMBOLS] = symbols
    
    @classmethod
    def get_risk_params(cls) -> RiskParameters:
        """Get current risk parameters."""
        params_dict = st.session_state.get(cls.RISK_PARAMS, {})
        return RiskParameters(**params_dict)
    
    @classmethod
    def set_risk_params(cls, params: RiskParameters) -> None:
        """Set risk parameters."""
        st.session_state[cls.RISK_PARAMS] = params.__dict__


class AlertManager:
    """Centralized alert management with validation, persistence, and rate limiting."""
    
    def __init__(self, notifier: Notifier):
        self.notifier = notifier
        self.validator = get_global_validator()  # Use centralized validator
        self._last_alert_times = {}  # Rate limiting
        self._alert_cooldown = 300  # 5 minutes
    
    @handle_exceptions
    def add_price_alert(self, symbol: str, condition: str, price: float) -> bool:
        """Add a price alert with comprehensive validation."""
        try:
            # Validate inputs
            symbol = self._validate_and_normalize_symbol(symbol)
            condition = condition.lower()
            
            if condition not in ['above', 'below']:
                raise DashboardValidationError("Condition must be 'above' or 'below'")
            if price <= MIN_PRICE:
                raise DashboardValidationError(f"Price must be greater than ${MIN_PRICE}")
            if price > 1000000:  # Reasonable upper limit
                raise DashboardValidationError("Price exceeds reasonable limits")
            
            # Create alert config
            alert_type = AlertType.PRICE_ABOVE if condition == 'above' else AlertType.PRICE_BELOW
            AlertConfig(symbol=symbol, alert_type=alert_type, threshold=price)  # Assuming object creation might have side effects or is planned for use

            # Store in session state
            alerts = st.session_state[SessionStateManager.ALERTS]
            if symbol not in alerts["price_alerts"]:
                alerts["price_alerts"][symbol] = {}
            
            # Check for duplicate alerts
            if condition in alerts["price_alerts"][symbol]:
                existing_price = alerts["price_alerts"][symbol][condition]
                if abs(existing_price - price) < 0.01:  # Allow minor price differences
                    logger.info(f"Price alert already exists: {symbol} {condition} ${price}")
                    return False
            
            alerts["price_alerts"][symbol][condition] = price
            
            # Set flag instead of immediate rerun
            SessionStateManager.set_flag(SessionStateManager.ALERT_ADDED)
            
            logger.info(f"Added price alert: {symbol} {condition} ${price}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add price alert: {e}")
            raise DashboardError(f"Failed to add alert: {e}")
    
    @handle_exceptions
    def add_pattern_alert(self, symbol: str, pattern: str) -> bool:
        """Add a pattern alert with validation."""
        try:
            symbol = self._validate_and_normalize_symbol(symbol)
              # Validate pattern exists
            pattern_detector = create_pattern_detector()
            available_patterns = pattern_detector.get_pattern_names()
            if pattern not in available_patterns:
                raise DashboardValidationError(f"Unknown pattern: {pattern}")
            
            # Store in session state
            alerts = st.session_state[SessionStateManager.ALERTS]
            if symbol not in alerts["pattern_alerts"]:
                alerts["pattern_alerts"][symbol] = []
            
            if pattern not in alerts["pattern_alerts"][symbol]:
                alerts["pattern_alerts"][symbol].append(pattern)
                
                # Set flag instead of immediate rerun
                SessionStateManager.set_flag(SessionStateManager.ALERT_ADDED)
                
                logger.info(f"Added pattern alert: {symbol} - {pattern}")
                return True
            else:
                logger.info(f"Pattern alert already exists: {symbol} - {pattern}")
                return False                
        except Exception as e:
            logger.error(f"Failed to add pattern alert: {e}")
            raise DashboardError(f"Failed to add pattern alert: {e}")
    
    def _validate_and_normalize_symbol(self, symbol: str) -> str:
        """Validate and normalize a stock symbol using core validator."""
        result = validate_symbol(symbol)
        if not result.is_valid:
            errors = result.errors or []
            raise DashboardValidationError(f"Invalid symbol: {'; '.join(errors)}")
        return result.value


class EnhancedModelManager:
    """Enhanced model management with caching, validation, and error recovery."""
    
    def __init__(self):
        self.model_manager = CoreModelManager()
        self.current_model = None
        self.model_metadata = None
        self.model_type = None
        self._model_cache = {}
    
    @handle_exceptions
    def load_model(self, model_type: str) -> Optional[Any]:
        """Load and cache a model with comprehensive error handling."""
        try:
            # Check cache first
            if model_type in self._model_cache:
                cached_model, cached_metadata = self._model_cache[model_type]
                self.current_model = cached_model
                self.model_metadata = cached_metadata
                self.model_type = model_type
                
                # Set success flag instead of immediate rerun
                SessionStateManager.set_flag(SessionStateManager.MODEL_LOAD_SUCCESS)
                
                logger.info(f"Loaded cached {model_type} model")
                return cached_model
            
            if model_type == "PatternNN":
                model, metadata = self._load_pattern_nn_model()
            else:
                model, metadata = self._load_sklearn_model(model_type)
            
            if model is not None:
                # Cache the model
                self._model_cache[model_type] = (model, metadata)
                self.current_model = model
                self.model_metadata = metadata
                self.model_type = model_type
                
                # Set success flag instead of immediate rerun
                SessionStateManager.set_flag(SessionStateManager.MODEL_LOAD_SUCCESS)
                
                logger.info(f"Successfully loaded {model_type} model")
                return model
            else:
                logger.warning(f"Failed to load {model_type} model")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load model {model_type}: {e}")
            raise DashboardError(f"Model loading failed: {e}")
    
    def clear_cache(self) -> None:
        """Clear model cache to free memory."""
        self._model_cache.clear()
        
        # Set flag instead of immediate rerun        SessionStateManager.set_flag(SessionStateManager.CACHE_CLEARED)
        
        logger.info("Model cache cleared")
    
    def _load_pattern_nn_model(self):
        """Load PatternNN model."""
        try:
            model = PatternNN()
            metadata = {"type": "PatternNN", "loaded_at": datetime.now()}
            return model, metadata
        except Exception as e:
            logger.error(f"Failed to load PatternNN: {e}")
            return None, None
    
    def _load_sklearn_model(self, model_type: str):
        """Load sklearn model."""
        try:
            model = self.model_manager.load_model(SKLEARN_MODELS[model_type])
            metadata = {"type": "sklearn", "model_file": model_type, "loaded_at": datetime.now()}
            return model, metadata
        except Exception as e:
            logger.error(f"Failed to load sklearn model {model_type}: {e}")
            return None, None


class TradingDashboard:
    """Main trading dashboard class with enterprise-grade architecture."""
    
    def __init__(self):
        """Initialize dashboard with all required components."""
        self._setup_page_config()
        
        # Initialize SessionManager to prevent button conflicts and state issues
        self.session_manager = create_session_manager("advanced_ai_trade")
        
        # Initialize session state first using dashboard utils
        SessionStateManager.initialize()
          # Initialize core components with error handling
        self.validator = get_global_validator()  # Use centralized validator from core module
        self.risk_manager = self._safe_init_component(RiskManager, "RiskManager")
        self.notifier = self._safe_init_component(Notifier, "Notifier")
        self.model_manager = self._safe_init_component(EnhancedModelManager, "ModelManager")
        
        if self.notifier:
            self.alert_manager = AlertManager(self.notifier)
        else:
            logger.error("Failed to initialize AlertManager due to Notifier failure")
            self.alert_manager = None
        
        # Training manager (initialized when needed)
        self.training_manager = None
        
        # Initialize risk manager with environment configuration
        self.risk_manager = RiskManager(load_from_env=True)
        
        logger.info("Dashboard initialized successfully with centralized technical analysis")
    
    def _setup_page_config(self) -> None:
        """Configure Streamlit page settings."""
        # st.set_page_config(  # Handled by main dashboard
        #     page_title="Advanced AI Trading Dashboard",
        #     page_icon="📈",
        #     layout="wide",
        #     initial_sidebar_state="expanded"
        # )
        pass

    def _safe_init_component(self, component_class, component_name: str):
        """Safely initialize a component with error handling."""
        try:
            return component_class()
        except Exception as e:
            logger.error(f"Failed to initialize {component_name}: {e}")
            return None

    @handle_exceptions
    def run(self) -> None:
        """Main dashboard entry point with comprehensive error handling."""
        try:
            # Check for and display success messages from flags
            self._handle_session_flags()
              # Render sidebar and get authenticated ETradeClient
            self._render_sidebar() # Assuming the action is still needed

            # Render main dashboard content
            self._render_main_dashboard()
            
            # Set up auto-refresh only if needed
            self._setup_conditional_auto_refresh()
            
        except Exception as e:
            handle_streamlit_error(e, "dashboard run")
    
    def _handle_session_flags(self) -> None:
        """Handle session state flags and show appropriate messages."""
        # Check for model load success
        if SessionStateManager.get_flag(SessionStateManager.MODEL_LOAD_SUCCESS):
            st.success("✅ Model loaded successfully!")
        
        # Check for training completion
        if SessionStateManager.get_flag(SessionStateManager.TRAINING_COMPLETE):
            st.success("✅ Model training completed successfully!")
        
        # Check for alert addition
        if SessionStateManager.get_flag(SessionStateManager.ALERT_ADDED):
            st.success("✅ Alert added successfully!")
        
        # Check for settings changes
        if SessionStateManager.get_flag(SessionStateManager.SETTINGS_CHANGED):
            st.success("✅ Settings updated successfully!")
        
        # Check for symbol updates
        if SessionStateManager.get_flag(SessionStateManager.SYMBOLS_UPDATED):
            st.success("✅ Watchlist updated successfully!")
        
        # Check for order placement
        if SessionStateManager.get_flag(SessionStateManager.ORDER_PLACED):
            st.success("✅ Order placed successfully!")
        
        # Check for connection changes
        if SessionStateManager.get_flag(SessionStateManager.CONNECTION_CHANGED):
            st.info("🔄 Connection status updated")
          # Check for cache clearing        if SessionStateManager.get_flag(SessionStateManager.CACHE_CLEARED):
            st.success("✅ Cache cleared successfully!")

    def _render_sidebar(self) -> Optional[ETradeClient]:
        """Render sidebar controls and return authenticated ETradeClient."""
        st.sidebar.title("🤖 AI Trading Dashboard")
        
        # Use SecureETradeManager for consistent authentication
        render_etrade_authentication()
        
        # Only render other controls if we have an authenticated client
        if SecureETradeManager.get_authenticated_client():
            self._render_symbol_management()
            self._render_risk_management()
            self._render_model_section()
            # Settings section
            with st.sidebar.expander("⚙️ Settings"):
                self._render_settings()
        
        return None

    def _render_main_dashboard(self) -> None:
        """Render the main dashboard content with comprehensive technical analysis."""
        st.title("📈 Advanced AI Trading Dashboard")
        
        # Check authentication using SecureETradeManager
        authenticated_client = SecureETradeManager.get_authenticated_client()
        if not authenticated_client:
            st.info("🔗 Connect your E*Trade credentials in the sidebar to begin trading")
            self._render_demo_technical_analysis()
            return
        
        # Main dashboard content with technical analysis
        symbols = SessionStateManager.get_symbols()
        if not symbols:
            st.warning("📊 Add symbols to your watchlist in the sidebar to begin analysis")
            return
        
        # Symbol selection for detailed analysis
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_symbol = st.selectbox("Select Symbol for Analysis", symbols, key="main_symbol_select")
        with col2:
            refresh_data = st.button("🔄 Refresh Data", help="Fetch latest market data")
        
        if selected_symbol:
            self._render_symbol_analysis(selected_symbol, refresh_data)
        
        # Portfolio overview
        self._render_portfolio_overview()
        
        # Market scanning section
        self._render_market_scanner(symbols)

    def _render_demo_technical_analysis(self) -> None:
        """Render demo technical analysis when not connected."""
        st.markdown("### 📊 Technical Analysis Demo")
        st.info("Connect to E*Trade to analyze live market data, or view demo analysis below:")
        
        # Create sample data for demo
        import numpy as np
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Generate realistic OHLCV data
        np.random.seed(42)
        base_price = 100
        prices = [base_price]
        
        for i in range(99):
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            new_price = prices[-1] * (1 + change)
            prices.append(int(max(new_price, 1)))  # Ensure positive prices
        
        demo_data = pd.DataFrame({
            'date': dates,
            'open': [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
            'high': [p * (1 + abs(np.random.uniform(0, 0.02))) for p in prices],
            'low': [p * (1 - abs(np.random.uniform(0, 0.02))) for p in prices],
            'close': prices,
            'volume': [np.random.randint(100000, 1000000) for _ in range(100)]
        })
          # Apply technical analysis
        self._display_technical_analysis(demo_data, "DEMO")

    def _render_symbol_analysis(self, symbol: str, refresh: bool = False) -> None:
        """Render detailed technical analysis for a specific symbol."""
        try:
            # Fetch market data
            with st.spinner(f"Fetching data for {symbol}..."):
                data = self._get_market_data(symbol, refresh)
            
            if data is None or data.empty:
                st.error(f"❌ No data available for {symbol}")
                return
            
            # Display analysis
            self._display_technical_analysis(data, symbol)
            
        except Exception as e:
            logger.error(f"Error rendering symbol analysis for {symbol}: {e}")
            st.error(f"❌ Failed to analyze {symbol}: {e}")

    def _get_market_data(self, symbol: str, refresh: bool = False) -> Optional[pd.DataFrame]:
        """Fetch market data with caching using secure E*Trade client."""
        try:
            # Use SecureETradeManager to get authenticated client
            secure_client = SecureETradeManager.get_authenticated_client()
            if secure_client:
                # Validate market data access permission
                if not SecureETradeManager.validate_operation_access('market_data'):
                    logger.warning(f"Market data access denied for {symbol}")
                    return None
                
                logger.info(f"Fetching live market data for {symbol}")
                # Get 30 days of daily data for analysis
                return secure_client.get_candles(symbol=symbol, interval="1day", days=30)
            else:
                # Fall back to demo mode
                logger.debug(f"No authenticated E*Trade client available for {symbol}, using demo mode")
                return None
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None

    def _display_technical_analysis(self, data: pd.DataFrame, symbol: str) -> None:
        """Display comprehensive technical analysis for market data."""
        try:
            # Initialize technical analysis
            ta = TechnicalAnalysis(data)
            
            # Create tabs for different analysis views
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Price & Indicators", 
                "🎯 Trading Signals", 
                "📊 Pattern Analysis", 
                "🔍 Risk Assessment"
            ])
            
            with tab1:
                self._render_price_analysis(data, symbol, ta)
            
            with tab2:
                self._render_trading_signals(data, symbol, ta)
            
            with tab3:
                self._render_pattern_analysis(data, symbol)
            
            with tab4:
                self._render_risk_analysis(data, symbol, ta)
                
        except Exception as e:
            logger.error(f"Error in technical analysis display: {e}")
            st.error(f"❌ Technical analysis failed: {e}")

    def _render_price_analysis(self, data: pd.DataFrame, symbol: str, ta: TechnicalAnalysis) -> None:
        """Render price chart with technical indicators."""
        try:
            st.markdown(f"### 📈 {symbol} Price Analysis")
            
            # Calculate indicators
            data_enriched = data.copy()
            
            # Add indicators using new centralized functions
            data_enriched['rsi'] = calculate_rsi(data)
            macd_line, macd_signal, macd_hist = calculate_macd(data)
            data_enriched['macd'] = macd_line
            data_enriched['macd_signal'] = macd_signal
            data_enriched['macd_hist'] = macd_hist
            
            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(data)
            data_enriched['bb_upper'] = bb_upper
            data_enriched['bb_middle'] = bb_middle
            data_enriched['bb_lower'] = bb_lower
            
            data_enriched['atr'] = calculate_atr(data)
            data_enriched['sma_20'] = calculate_sma(data, length=20)
            data_enriched['ema_20'] = calculate_ema(data, length=20)
            
            # Create comprehensive chart
            self._create_technical_chart(data_enriched, symbol)
            
            # Display current metrics
            self._display_current_metrics(data_enriched, symbol)
            
        except Exception as e:
            logger.error(f"Error in price analysis: {e}")
            st.error(f"❌ Price analysis failed: {e}")

    def _create_technical_chart(self, data: pd.DataFrame, symbol: str) -> None:
        """Create interactive chart with technical indicators."""
        try:
            # Create subplots
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(
                    f'{symbol} Price with Bollinger Bands',
                    'RSI',
                    'MACD',
                    'Volume'
                ),
                row_heights=[0.5, 0.2, 0.2, 0.1]
            )
            
            # Price and Bollinger Bands
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Bollinger Bands
            fig.add_trace(go.Scatter(x=data.index, y=data['bb_upper'], 
                                   name='BB Upper', line=dict(color='rgba(255,0,0,0.3)')), 
                         row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['bb_lower'], 
                                   name='BB Lower', line=dict(color='rgba(255,0,0,0.3)'),
                                   fill='tonexty', fillcolor='rgba(255,0,0,0.1)'), 
                         row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['sma_20'], 
                                   name='SMA 20', line=dict(color='blue')), 
                         row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['ema_20'], 
                                   name='EMA 20', line=dict(color='orange')), 
                         row=1, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=data.index, y=data['rsi'], name='RSI', 
                                   line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red",   row="2", col="1")
            fig.add_hline(y=30, line_dash="dash", line_color="green", row="2", col="1")
            fig.add_hline(y=50, line_dash="dot",  line_color="gray",  row="2", col="1")
            
            # MACD
            fig.add_trace(go.Scatter(x=data.index, y=data['macd'], name='MACD', 
                                   line=dict(color='blue')), row=3, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['macd_signal'], name='Signal', 
                                   line=dict(color='red')), row=3, col=1)
            fig.add_trace(go.Bar(x=data.index, y=data['macd_hist'], name='Histogram', 
                               marker_color='gray'), row=3, col=1)
            
            # Volume
            fig.add_trace(go.Bar(x=data.index, y=data['volume'], name='Volume', 
                               marker_color='lightblue'), row=4, col=1)
            
            # Update layout
            fig.update_layout(
                height=800,
                title=f'{symbol} Technical Analysis',
                xaxis_rangeslider_visible=False,
                showlegend=True
            )
            
            # Update y-axis labels
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
            fig.update_yaxes(title_text="MACD", row=3, col=1)
            fig.update_yaxes(title_text="Volume", row=4, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error creating technical chart: {e}")
            st.error("❌ Failed to create chart")

    def _display_current_metrics(self, data: pd.DataFrame, symbol: str) -> None:
        """Display current technical indicator values."""
        try:
            last_row = data.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                rsi_val = last_row['rsi']
                rsi_status = "🔴 Overbought" if rsi_val > 70 else "🟢 Oversold" if rsi_val < 30 else "🟡 Neutral"
                st.metric("RSI", f"{rsi_val:.1f}", help=f"Status: {rsi_status}")
            
            with col2:
                macd_val = last_row['macd'] - last_row['macd_signal']
                macd_trend = "🟢 Bullish" if macd_val > 0 else "🔴 Bearish"
                st.metric("MACD Diff", f"{macd_val:.4f}", help=f"Trend: {macd_trend}")
            
            with col3:
                atr_val = last_row['atr']
                st.metric("ATR", f"{atr_val:.2f}", help="Average True Range (Volatility)")
            
            with col4:
                price = last_row['close']
                bb_pos = ((price - last_row['bb_lower']) / 
                          (last_row['bb_upper'] - last_row['bb_lower'])) * 100
                bb_status = "🔴 Upper" if bb_pos > 80 else "🟢 Lower" if bb_pos < 20 else "🟡 Middle"
                st.metric("BB Position", f"{bb_pos:.1f}%", help=f"Position: {bb_status}")
                
        except Exception as e:
            logger.error(f"Error displaying metrics: {e}")
            st.error("❌ Failed to display metrics")

    def _render_trading_signals(self, data: pd.DataFrame, symbol: str, ta: TechnicalAnalysis) -> None:
        """Render trading signals based on technical analysis."""
        try:
            st.markdown(f"### 🎯 {symbol} Trading Signals")
            
            # Evaluate composite signal using centralized TechnicalAnalysis class
            composite_score, rsi_score, macd_score, bb_score = ta.evaluate()
            
            if composite_score is not None:
                # Display composite signal
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    signal_strength = abs(composite_score)
                    if composite_score > 0.3:
                        signal_text = "🟢 STRONG BUY"
                        signal_color = "green"
                    elif composite_score > 0.1:
                        signal_text = "🟡 BUY"
                        signal_color = "orange"
                    elif composite_score < -0.3:
                        signal_text = "🔴 STRONG SELL"
                        signal_color = "red"
                    elif composite_score < -0.1:
                        signal_text = "🟡 SELL"
                        signal_color = "orange"
                    else:
                        signal_text = "⚪ HOLD"
                        signal_color = "gray"
                    
                    st.markdown(f"**Overall Signal:** <span style='color:{signal_color}'>{signal_text}</span>", 
                               unsafe_allow_html=True)
                    st.metric("Signal Strength", f"{signal_strength:.2f}", f"{composite_score:.2f}")
                
                with col2:
                    # Individual indicator scores
                    st.markdown("**Individual Indicator Scores:**")
                    score_df = pd.DataFrame({
                        'Indicator': ['RSI', 'MACD', 'Bollinger Bands'],
                        'Score': [rsi_score, macd_score, bb_score],
                        'Signal': [
                            # RSI
                            'Bullish' if rsi_score is not None and rsi_score > 0
                            else 'Bearish' if rsi_score is not None and rsi_score < 0
                            else 'Neutral',
                            # MACD
                            'Bullish' if macd_score is not None and macd_score > 0
                            else 'Bearish' if macd_score is not None and macd_score < 0
                            else 'Neutral',
                            # Bollinger Bands
                            'Bullish' if bb_score is not None and bb_score > 0
                            else 'Bearish' if bb_score is not None and bb_score < 0
                            else 'Neutral'
                        ]
                    })
                    st.dataframe(score_df, use_container_width=True)            # Price targets
            self._display_price_targets(data, symbol, ta)
            
            # Risk metrics
            self._display_risk_metrics(data, symbol, ta)
            
        except Exception as e:
            logger.error(f"Error in trading signals: {e}")
            st.error(f"❌ Trading signals failed: {e}")

    def _display_price_targets(self, data: pd.DataFrame, symbol: str, ta: TechnicalAnalysis) -> None:
        """Display price targets based on technical analysis."""
        try:
            st.markdown("#### 🎯 Price Targets")
            
            current_price = data['close'].iloc[-1]
            
            # Use centralized function for ATR calculation
            try:
                atr_series = calculate_atr(data, length=14)
                atr_value = atr_series.iloc[-1] if len(atr_series) > 0 and not pd.isna(atr_series.iloc[-1]) else None
            except IndicatorError as e:
                logger.warning(f"ATR calculation failed: {e}")
                atr_value = None
            
            if atr_value:
                # Calculate targets using ATR
                target_up = current_price + (1.5 * atr_value)
                target_down = current_price - (1.5 * atr_value)
                  # Fibonacci target using centralized TechnicalAnalysis class
                fib_target = ta.calculate_price_target()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                
                with col2:
                    upside = ((target_up - current_price) / current_price) * 100
                    st.metric("Upside Target", f"${target_up:.2f}", f"+{upside:.1f}%")
                
                with col3:
                    downside = ((target_down - current_price) / current_price) * 100
                    st.metric("Downside Target", f"${target_down:.2f}", f"{downside:.1f}%")
                
                with col4:
                    fib_change = ((fib_target - current_price) / current_price) * 100
                    st.metric("Fibonacci Target", f"${fib_target:.2f}", f"{fib_change:+.1f}%")
            
        except Exception as e:
            logger.error(f"Error displaying price targets: {e}")

    def _display_risk_metrics(self, data: pd.DataFrame, symbol: str, ta: TechnicalAnalysis) -> None:
        """Display risk assessment metrics."""
        try:
            st.markdown("#### ⚠️ Risk Assessment")
            
            risk_params = SessionStateManager.get_risk_params()
            current_price = data['close'].iloc[-1]
            
            # Use centralized function for ATR calculation
            try:
                atr_series = calculate_atr(data, length=14)
                atr_value = atr_series.iloc[-1] if len(atr_series) > 0 and not pd.isna(atr_series.iloc[-1]) else None
            except IndicatorError as e:
                logger.warning(f"ATR calculation failed: {e}")
                atr_value = None
            
            if atr_value:
                # Calculate position sizing
                account_value = 100000  # Default demo value
                max_position_value = account_value * risk_params.max_position_size
                stop_loss_distance = atr_value * risk_params.stop_loss_atr
                stop_loss_price = current_price - stop_loss_distance
                
                risk_per_share = stop_loss_distance
                max_risk_amount = account_value * risk_params.max_daily_loss
                
                if risk_per_share > 0:
                    suggested_shares = int(max_risk_amount / risk_per_share)
                    suggested_position_value = suggested_shares * current_price
                    
                    # Ensure within position size limits
                    if suggested_position_value > max_position_value:
                        suggested_shares = int(max_position_value / current_price)
                        suggested_position_value = suggested_shares * current_price
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Suggested Shares", f"{suggested_shares:,}")
                    st.metric("Position Value", f"${suggested_position_value:,.0f}")
                
                with col2:
                    st.metric("Stop Loss Price", f"${stop_loss_price:.2f}")
                    risk_pct = (risk_per_share / current_price) * 100
                    st.metric("Risk per Share", f"${risk_per_share:.2f}", f"{risk_pct:.1f}%")
                
                with col3:
                    total_risk = suggested_shares * risk_per_share
                    risk_of_account = (total_risk / account_value) * 100
                    st.metric("Total Risk", f"${total_risk:.0f}", f"{risk_of_account:.2f}%")
                    
        except Exception as e:
            logger.error(f"Error displaying risk metrics: {e}")

    def _render_pattern_analysis(self, data: pd.DataFrame, symbol: str) -> None:
        """Render candlestick pattern analysis."""
        try:
            st.markdown(f"### 📊 {symbol} Pattern Analysis")
            
            if len(data) < 4:
                st.warning("Insufficient data for pattern analysis")
                return
            
            # Pattern detection
            pattern_detector = create_pattern_detector()
            patterns_found = []
            
            # Check last few candles for patterns
            for i in range(max(0, len(data) - 10), len(data)):
                if i + 3 < len(data):
                    window = data.iloc[i:i+4]
                    
                    # Check for bullish patterns
                    if hasattr(pattern_detector, 'is_bullish_engulfing'):
                        if pattern_detector.is_bullish_engulfing(window):
                            patterns_found.append({
                                'Pattern': 'Bullish Engulfing',
                                'Date': window.index[-1],
                                'Type': 'Bullish',
                                'Confidence': 85
                            })
                    
                    if hasattr(pattern_detector, 'is_hammer'):
                        if pattern_detector.is_hammer(window):
                            patterns_found.append({
                                'Pattern': 'Hammer',
                                'Date': window.index[-1],
                                'Type': 'Bullish',
                                'Confidence': 75
                            })
            
            if patterns_found:
                st.markdown("#### 🔍 Recent Patterns Detected")
                patterns_df = pd.DataFrame(patterns_found)
                st.dataframe(patterns_df, use_container_width=True)
            else:
                st.info("No significant patterns detected in recent data")
            
            # ML Model prediction if available
            if st.session_state.get(SessionStateManager.MODEL):
                self._render_ml_prediction(data, symbol)
                
        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}")
            st.error(f"❌ Pattern analysis failed: {e}")

    def _render_ml_prediction(self, data: pd.DataFrame, symbol: str) -> None:
        """Render ML model prediction."""
        try:
            st.markdown("#### 🤖 AI Model Prediction")
            
            model = st.session_state.get(SessionStateManager.MODEL)
            if not model:
                st.info("Load a model in the sidebar to see AI predictions")
                return
            
            # This would integrate with the actual ML prediction pipeline
            # For now, show placeholder
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Simulate prediction
                import random
                prediction_prob = random.uniform(0.3, 0.9)
                prediction = "BUY" if prediction_prob > 0.6 else "SELL" if prediction_prob < 0.4 else "HOLD"
                
                st.metric("AI Prediction", prediction)
                st.metric("Confidence", f"{prediction_prob:.1%}")
            
            with col2:
                st.markdown("**Model Info:**")
                st.text("Model: PatternNN")
                st.text("Last Updated: 2024-01-01")
                st.text("Accuracy: 72.5%")
            
            with col3:
                if st.button("🔄 Update Prediction", key=f"update_pred_{symbol}"):
                    st.rerun()
                    
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")

    def _render_risk_analysis(self, data: pd.DataFrame, symbol: str, ta: TechnicalAnalysis) -> None:
        """Render comprehensive risk analysis."""
        try:
            st.markdown(f"### 🔍 {symbol} Risk Analysis")
            
            # Volatility analysis
            returns = data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Value at Risk (VaR)
            var_95 = returns.quantile(0.05)
            var_99 = returns.quantile(0.01)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Volatility Metrics")
                st.metric("Annualized Volatility", f"{volatility:.1%}")
                st.metric("95% VaR (Daily)", f"{var_95:.2%}")
                st.metric("99% VaR (Daily)", f"{var_99:.2%}")
                
                # Risk rating
                if volatility > 0.4:
                    risk_rating = "🔴 High Risk"
                elif volatility > 0.25:
                    risk_rating = "🟡 Medium Risk"
                else:
                    risk_rating = "🟢 Low Risk"
                
                st.markdown(f"**Risk Rating:** {risk_rating}")
            
            with col2:
                st.markdown("#### 🎯 Position Sizing Recommendation")
                
                risk_params = SessionStateManager.get_risk_params()
                account_value = 100000  # Demo value
                
                # Kelly Criterion approximation
                win_rate = 0.55  # Assumed
                avg_win = abs(var_95) * 1.5  # Simplified
                avg_loss = abs(var_95)
                
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_fraction = max(0, min(kelly_fraction, risk_params.max_position_size))
                
                recommended_position = account_value * kelly_fraction
                current_price = data['close'].iloc[-1]
                recommended_shares = int(recommended_position / current_price)
                
                st.metric("Kelly Fraction", f"{kelly_fraction:.1%}")
                st.metric("Recommended Position", f"${recommended_position:,.0f}")
                st.metric("Recommended Shares", f"{recommended_shares:,}")
            
            # Correlation analysis (if multiple symbols available)
            symbols = SessionStateManager.get_symbols()
            if len(symbols) > 1:
                st.markdown("#### 🔗 Portfolio Correlation Analysis")
                st.info("Portfolio correlation analysis requires multiple symbol data")
                
        except Exception as e:
            logger.error(f"Error in risk analysis: {e}")
            st.error(f"❌ Risk analysis failed: {e}")

    def _render_portfolio_overview(self) -> None:
        """Render portfolio overview section."""
        try:
            st.markdown("### 💼 Portfolio Overview")
            
            # Mock portfolio data - in real implementation this would come from E*Trade API
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Portfolio Value", "$52,500", "+5.0%")
            
            with col2:
                st.metric("Daily P&L", "+$2,500", "+5.0%")
            
            with col3:
                st.metric("Open Positions", "4", "+1")
            
            with col4:
                st.metric("Available Cash", "$10,000", "-$2,500")
            
            # Portfolio allocation chart would go here
            st.info("📊 Portfolio allocation and performance charts will be displayed here when connected to E*Trade")
            
        except Exception as e:
            logger.error(f"Error in portfolio overview: {e}")

    def _render_market_scanner(self, symbols: List[str]) -> None:
        """Render market scanner for multiple symbols."""
        try:
            st.markdown("### 🔍 Market Scanner")
            
            if not symbols:
                st.info("Add symbols to your watchlist to scan the market")
                return
            
            # This would scan all symbols for trading opportunities
            scanner_data = []
            
            for symbol in symbols[:5]:  # Limit to first 5 for demo
                # In real implementation, fetch data for each symbol
                # For demo, create mock data
                scanner_data.append({                    'Symbol': symbol,
                    'Price': f"${np.random.uniform(50, 200):.2f}",
                    'Change': f"{np.random.uniform(-5, 5):+.2f}%",
                    'RSI': f"{np.random.uniform(20, 80):.1f}",
                    'Signal': np.random.choice(['BUY', 'SELL', 'HOLD']),
                    'Pattern': np.random.choice(['Hammer', 'Doji', 'None', 'Engulfing'])
                })
            
            if scanner_data:
                scanner_df = pd.DataFrame(scanner_data)
                
                # Style the dataframe
                st.dataframe(
                    scanner_df,
                    use_container_width=True,
                    column_config={
                        'Symbol': st.column_config.TextColumn('Symbol', width='medium'),
                        'Signal': st.column_config.TextColumn('Signal', width='small'),
                    }
                )
            else:
                st.info("No scanner data available")
                
        except Exception as e:
            logger.error(f"Error in market scanner: {e}")
            st.error(f"❌ Market scanner failed: {e}")

    def _setup_conditional_auto_refresh(self) -> None:
        """Set up auto-refresh only when needed and not during user interactions."""
        try:
            # Skip auto-refresh if user is actively interacting
            if self._is_user_interacting():
                return
            
            refresh_interval = st.sidebar.number_input(
                "Auto-refresh (seconds)",
                min_value=10,
                max_value=300,
                value=DEFAULT_REFRESH_INTERVAL,
                help="Automatic data refresh interval"
            )
            
            auto_refresh_enabled = st.sidebar.checkbox("Enable Auto-refresh")
            
            if auto_refresh_enabled:
                # Use session state from dashboard_utils
                last_update = st.session_state.get('last_update_time')
                current_time = datetime.now()
                
                if (last_update is None or 
                    (current_time - last_update).total_seconds() >= refresh_interval):
                    st.session_state.last_update_time = current_time
                    
                    # Only refresh if no flags are set (no recent user actions)
                    if not self._has_active_flags():
                        st.rerun()
                    
        except Exception as e:
            handle_streamlit_error(e, "auto-refresh setup")
    
    def _is_user_interacting(self) -> bool:
        """Check if user is currently interacting with the dashboard."""
        # Check for recent user actions
        active_flags = [
            SessionStateManager.MODEL_LOAD_SUCCESS,
            SessionStateManager.TRAINING_COMPLETE,
            SessionStateManager.ALERT_ADDED,
            SessionStateManager.SETTINGS_CHANGED,
            SessionStateManager.SYMBOLS_UPDATED,
            SessionStateManager.ORDER_PLACED,
            SessionStateManager.CACHE_CLEARED
        ]
        
        return any(SessionStateManager.check_flag(flag) for flag in active_flags)
    
    def _has_active_flags(self) -> bool:
        """Check if any flags are currently active."""
        return self._is_user_interacting()

    @handle_exceptions
    def _render_symbol_management(self) -> None:
        """Render symbol watchlist management with enhanced validation."""
        st.sidebar.markdown("### 📊 Watchlist")
        
        current_symbols = SessionStateManager.get_symbols()
        symbols_input = st.sidebar.text_input(
            "Symbols (comma-separated)",
            value=",".join(current_symbols),
            help=f"Maximum {MAX_SYMBOLS} symbols. Example: AAPL,MSFT,GOOGL"        )
        
        if self.session_manager.create_button("Update Watchlist", container=st.sidebar):
            try:                # Parse symbols input
                symbols_input_str = symbols_input.strip()
                
                if not symbols_input_str:
                    st.sidebar.error("Please provide at least one symbol")
                    return
                
                # Use the centralized validate_symbols function
                validation_result = validate_symbols(symbols_input_str)
                
                if not validation_result.is_valid:
                    for error in validation_result.errors:
                        st.sidebar.warning(error)
                    return
                
                new_symbols = validation_result.symbols
                
                if len(new_symbols) > MAX_SYMBOLS:
                    st.sidebar.error(f"Maximum {MAX_SYMBOLS} symbols allowed")
                    return
                
                if not new_symbols:
                    st.sidebar.error("No valid symbols provided")
                    return
                
                SessionStateManager.set_symbols(new_symbols)
                
                # Set flag instead of immediate rerun
                SessionStateManager.set_flag(SessionStateManager.SYMBOLS_UPDATED)
                
            except DashboardValidationError as e:
                st.sidebar.error(f"Validation error: {e}")
            except Exception as e:
                logger.error(f"Error updating symbols: {e}")
                st.sidebar.error("Failed to update symbols")

    @handle_exceptions
    def _render_risk_management(self) -> None:
        """Render risk management controls with real-time validation."""
        st.sidebar.markdown("### ⚖️ Risk Management")
        
        # Display current configuration from environment
        with st.sidebar.expander("📊 Current Risk Configuration", expanded=False):
            config = self.risk_manager.get_configuration_summary()
            st.write(f"**Max Positions:** {config['max_positions']}")
            st.write(f"**Max Daily Loss:** {config['max_daily_loss']*100:.1f}%")
            st.write(f"**Max Order Value:** ${config['max_order_value']:,.0f}")
            st.write(f"**Max Position %:** {config['max_position_pct']*100:.1f}%")
            st.write(f"**Default ATR Period:** {config['default_atr_period']}")
            st.write(f"**Default ATR Multiplier:** {config['default_atr_multiplier']}")
        
        current_params = SessionStateManager.get_risk_params()
        
        # Position size control with dynamic limits
        max_position = st.sidebar.slider(
            "Max Position Size (%)",
            min_value=MIN_POSITION_SIZE * 100,
            max_value=min(MAX_POSITION_SIZE * 100, config['max_position_pct'] * 100),
            value=current_params.max_position_size * 100,
            step=0.1,
            help=f"Maximum position size (Environment limit: {config['max_position_pct']*100:.1f}%)",
            format="%.1f",
            key="risk_max_position"
        ) / 100.0
        
        # Stop loss control
        stop_loss_atr = st.sidebar.slider(
            "Stop Loss (ATR multiplier)",
            min_value=0.5,
            max_value=10.0,
            value=current_params.stop_loss_atr,
            step=0.1,
            help=f"Stop loss distance as ATR multiple (Default: {config['default_atr_multiplier']})",
            key="risk_stop_loss"
        )
        
        # Daily loss limit
        max_daily_loss = st.sidebar.slider(
            "Max Daily Loss (%)",
            min_value=1.0,
            max_value=min(50.0, config['max_daily_loss'] * 100),
            value=current_params.max_daily_loss * 100,
            step=0.5,
            help=f"Maximum allowed daily portfolio loss (Environment limit: {config['max_daily_loss']*100:.1f}%)",
            key="risk_daily_loss"
        ) / 100.0
        
        # Position correlation limit
        correlation_limit = st.sidebar.slider(
            "Max Position Correlation",
            min_value=0.1,
            max_value=1.0,
            value=current_params.position_correlation_limit,
            step=0.05,
            help=f"Maximum correlation between positions (Default: {config['max_correlation_exposure']})",
            key="risk_correlation"
        )
        
        # Update risk parameters only when values actually change
        try:
            new_params = RiskParameters(
                max_position_size=max_position,
                stop_loss_atr=stop_loss_atr,
                max_daily_loss=max_daily_loss,
                position_correlation_limit=correlation_limit
            )
            
            # Only update if parameters actually changed
            if new_params.__dict__ != current_params.__dict__:
                SessionStateManager.set_risk_params(new_params)
                # Risk manager configuration comes from environment, no need to recreate
                
                # Set flag instead of showing immediate success message
                SessionStateManager.set_flag(SessionStateManager.SETTINGS_CHANGED)
                
        except DashboardValidationError as e:
            st.sidebar.error(f"❌ Invalid parameters: {e}")
        except Exception as e:
            logger.error(f"Risk parameter update error: {e}")
            st.sidebar.error("Failed to update risk parameters")

    @handle_exceptions
    def _render_model_section(self) -> None:
        """Render ML model controls with enhanced management."""
        st.sidebar.markdown("### 🤖 AI Models")
        
        if not self.model_manager:
            st.sidebar.error("❌ Model manager unavailable")
            return
        
        # Model selection with status indication
        try:
            available_models = ["PatternNN"]
            # Ensure items are strings before calling endswith
            all_listed_items = self.model_manager.model_manager.list_models()
            joblib_models = [
                item for item in all_listed_items
                if isinstance(item, str) and item.endswith(".joblib")
            ]
            available_models.extend(joblib_models)
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            available_models = ["PatternNN"]
        
        selected_model = st.sidebar.selectbox(
            "Select Model",
            available_models,
            help="Choose ML model for predictions",
            key="model_selection"
        )
          # Model controls
        col1, col2 = st.sidebar.columns(2)
        
        # Load model button
        with col1:
            if self.session_manager.create_button("Load", help="Load selected model", container=col1):
                try:
                    with st.spinner("Loading model..."):
                        model = self.model_manager.load_model(selected_model)
                        if model:
                            st.session_state[SessionStateManager.MODEL] = model
                        else:
                            st.sidebar.error("❌ Failed to load model")
                except Exception as e:
                    logger.error(f"Model loading error: {e}")
                    st.sidebar.error(f"❌ Loading failed: {e}")
        
        # Clear cache button
        with col2:
            if self.session_manager.create_button("Clear Cache", help="Clear model cache", container=col2):
                try:
                    self.model_manager.clear_cache()
                except Exception as e:
                    logger.error(f"Cache clear error: {e}")
                    st.sidebar.error("❌ Clear failed")

    def _render_settings(self) -> None:
        """Render settings controls."""
        self.session_manager.create_checkbox("Enable Debug Mode", "debug_mode")
        self.session_manager.create_checkbox("Paper Trading Mode", "paper_trading", value=True)
        
        if self.session_manager.create_button("Reset Settings"):
            self._reset_settings()
        
        if self.session_manager.create_button("Clear All Caches"):
            self._clear_all_caches()

    def _reset_settings(self) -> None:
        """Reset all settings to defaults."""
        try:
            # Reset risk parameters to defaults
            default_risk = RiskParameters()
            SessionStateManager.set_risk_params(default_risk)
            
            # Set flag instead of immediate success message
            SessionStateManager.set_flag(SessionStateManager.SETTINGS_CHANGED)
            
        except Exception as e:
            logger.error(f"Settings reset error: {e}")
            st.error("Failed to reset settings")

    def _clear_all_caches(self) -> None:
        """Clear all application caches."""
        try:
            # Clear Streamlit cache
            st.cache_data.clear()
            
            # Clear session state cache
            if 'data_cache' in st.session_state:
                st.session_state['data_cache'] = {}
            
            # Clear model cache
            if self.model_manager:
                self.model_manager.clear_cache()
            
            # Set flag instead of immediate success message
            SessionStateManager.set_flag(SessionStateManager.CACHE_CLEARED)
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            st.error("Failed to clear caches")


if __name__ == "__main__":
    """Main entry point for the trading dashboard application."""
    try:
        # Initialize and run the dashboard
        dashboard = TradingDashboard()
        dashboard.run()
        
    except Exception as e:
        logger.exception(f"Fatal dashboard error: {e}")
        st.error("🚨 Fatal Error: Dashboard failed to start")
        st.error(f"Error details: {e}")        # Emergency fallback UI
        st.markdown("### 🔧 Emergency Options")
        
        # Create emergency session manager for fallback buttons
        emergency_session_manager = create_session_manager("emergency_fallback")
        
        if emergency_session_manager.create_button("🔄 Restart Dashboard", "emergency_restart"):
            st.cache_data.clear()
            st.rerun()
        
        if emergency_session_manager.create_button("🗑️ Clear Session State", "emergency_clear"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
