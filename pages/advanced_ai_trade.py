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

import functools
import hashlib
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Protocol
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from pydantic import ValidationError as PydanticValidationError

# Project imports
from core.dashboard_utils import (
    initialize_dashboard_session_state,
    safe_streamlit_metric,
    handle_streamlit_error,
    cache_key_builder
)
from core.etrade_client_factory import create_etrade_client
from core.risk_manager_v2 import RiskManager
from patterns.patterns import CandlestickPatterns
from patterns.patterns_nn import PatternNN
from train.deeplearning_config import TrainingConfig
from train.deeplearning_trainer import PatternModelTrainer
from train.model_manager import ModelManager as CoreModelManager
from utils.config.notification_settings_ui import render_notification_settings
from utils.data_validator import DataValidator
from utils.logger import setup_logger
from utils.notifier import Notifier
from utils.security import get_api_credentials
from utils.technicals.indicators import TechnicalIndicators
from utils.decorators import handle_exceptions, handle_dashboard_exceptions
from utils.training_manager import TrainingManager
from train.ml_config import MLConfig
import torch

# Configure logging
logger = setup_logger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCKER_INDICATOR = '/.dockerenv'
DEFAULT_REFRESH_INTERVAL = 60  # seconds
MAX_SYMBOLS = 20  # Limit for performance
CACHE_TTL = 300  # 5 minutes
MIN_PRICE = 0.01
MAX_POSITION_SIZE = 0.5  # 50%
MIN_POSITION_SIZE = 0.001  # 0.1%


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
        """Validate and normalize symbol."""
        if not symbol or not symbol.strip():
            raise DashboardValidationError("Symbol cannot be empty")
        symbol = symbol.upper().strip()
        if not symbol.isalpha() or len(symbol) > 10:
            raise DashboardValidationError("Invalid symbol format")
        return symbol


@dataclass
class RiskParameters:
    """Risk management parameters with validation."""
    max_position_size: float = 0.02  # 2% default
    stop_loss_atr: float = 2.0
    max_daily_loss: float = 0.05  # 5% max daily loss
    position_correlation_limit: float = 0.3
    
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
    

class AlertManager:
    """Centralized alert management with validation, persistence, and rate limiting."""
    
    def __init__(self, notifier: Notifier):
        self.notifier = notifier
        self.validator = DataValidator()
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
            alert = AlertConfig(symbol=symbol, alert_type=alert_type, threshold=price)
            
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
            available_patterns = CandlestickPatterns.get_pattern_names()
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
        
        # Set flag instead of immediate rerun
        SessionStateManager.set_flag(SessionStateManager.CACHE_CLEARED)
        
        logger.info("Model cache cleared")


class TradingDashboard:
    """Main trading dashboard class with enterprise-grade architecture."""
    
    def __init__(self):
        """Initialize dashboard with all required components."""
        self._setup_page_config()
        
        # Initialize session state first using dashboard utils
        SessionStateManager.initialize()
        
        # Initialize core components with error handling
        self.validator = self._safe_init_component(DataValidator, "DataValidator")
        self.risk_manager = self._safe_init_component(RiskManager, "RiskManager")
        self.indicators = self._safe_init_component(TechnicalIndicators, "TechnicalIndicators")
        self.notifier = self._safe_init_component(Notifier, "Notifier")
        self.model_manager = self._safe_init_component(EnhancedModelManager, "ModelManager")
        
        if self.notifier:
            self.alert_manager = AlertManager(self.notifier)
        else:
            logger.error("Failed to initialize AlertManager due to Notifier failure")
            self.alert_manager = None
        
        # E*Trade client (initialized on credential validation)
        self.etrade_client: Optional[ClientProtocol] = None
        
        logger.info("Dashboard initialized successfully")

    @handle_exceptions
    def run(self) -> None:
        """Main dashboard entry point with comprehensive error handling."""
        try:
            # Check for and display success messages from flags
            self._handle_session_flags()
            
            # Render sidebar and get credentials
            credentials = self._render_sidebar()
            
            # Initialize E*Trade client if credentials provided
            if credentials:
                self._initialize_etrade_client(credentials)
            
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
        
        # Check for cache clearing
        if SessionStateManager.get_flag(SessionStateManager.CACHE_CLEARED):
            st.success("✅ Cache cleared successfully!")

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
            help=f"Maximum {MAX_SYMBOLS} symbols. Example: AAPL,MSFT,GOOGL"
        )
        
        if st.sidebar.button("Update Watchlist"):
            try:
                # Parse and validate symbols
                new_symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
                
                if not new_symbols:
                    st.sidebar.error("Please provide at least one symbol")
                    return
                
                if len(new_symbols) > MAX_SYMBOLS:
                    st.sidebar.error(f"Maximum {MAX_SYMBOLS} symbols allowed")
                    return
                
                # Additional validation using DataValidator if available
                if self.validator:
                    validated_symbols = []
                    for symbol in new_symbols:
                        try:
                            validated_symbol = self.validator.validate_symbol(symbol)
                            validated_symbols.append(validated_symbol)
                        except Exception as e:
                            st.sidebar.warning(f"Invalid symbol {symbol}: {e}")
                    new_symbols = validated_symbols
                
                if not new_symbols:
                    st.sidebar.error("No valid symbols provided")
                    return
                
                SessionStateManager.set_symbols(new_symbols)
                
                # Set flag instead of immediate rerun
                SessionStateManager.set_flag(SessionStateManager.SYMBOLS_UPDATED)
                
                # Don't call st.rerun() here - let the flag handler show the message
                
            except DashboardValidationError as e:
                st.sidebar.error(f"Validation error: {e}")
            except Exception as e:
                logger.error(f"Error updating symbols: {e}")
                st.sidebar.error("Failed to update symbols")

    @handle_exceptions
    def _render_risk_management(self) -> None:
        """Render risk management controls with real-time validation."""
        st.sidebar.markdown("### ⚖️ Risk Management")
        
        current_params = SessionStateManager.get_risk_params()
        
        # Position size control with dynamic limits
        max_position = st.sidebar.slider(
            "Max Position Size (%)",
            min_value=MIN_POSITION_SIZE * 100,
            max_value=MAX_POSITION_SIZE * 100,
            value=current_params.max_position_size * 100,
            step=0.1,
            help="Maximum position size as percentage of portfolio",
            format="%.1f",
            key="risk_max_position"  # Add key to prevent conflicts
        ) / 100.0
        
        # Stop loss control
        stop_loss_atr = st.sidebar.slider(
            "Stop Loss (ATR multiplier)",
            min_value=0.5,
            max_value=10.0,
            value=current_params.stop_loss_atr,
            step=0.1,
            help="Stop loss distance as ATR multiple",
            key="risk_stop_loss"
        )
        
        # Daily loss limit
        max_daily_loss = st.sidebar.slider(
            "Max Daily Loss (%)",
            min_value=1.0,
            max_value=50.0,
            value=current_params.max_daily_loss * 100,
            step=0.5,
            help="Maximum allowed daily portfolio loss",
            key="risk_daily_loss"
        ) / 100.0
        
        # Position correlation limit
        correlation_limit = st.sidebar.slider(
            "Max Position Correlation",
            min_value=0.1,
            max_value=1.0,
            value=current_params.position_correlation_limit,
            step=0.05,
            help="Maximum correlation between positions",
            key="risk_correlation"
        )
        
        # Real-time risk assessment
        self._display_risk_assessment(max_position, stop_loss_atr, max_daily_loss)
        
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
                if self.risk_manager:
                    self.risk_manager = RiskManager(max_position_pct=max_position)
                
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
            joblib_models = [f for f in self.model_manager.model_manager.list_models() 
                           if f.endswith(".joblib")]
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
            if st.button("Load", help="Load selected model", key="load_model_btn"):
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
            if st.button("Clear Cache", help="Clear model cache", key="clear_cache_btn"):
                try:
                    self.model_manager.clear_cache()
                except Exception as e:
                    logger.error(f"Cache clear error: {e}")
                    st.sidebar.error("❌ Clear failed")
        
        # Training controls
        st.sidebar.markdown("#### 🎯 Training")
        
        # Training backend selection
        training_backend = st.sidebar.selectbox(
            "Training Method",
            ["MLPipeline (Complete)", "Direct Trainer"],
            help="Choose training approach",
            key="training_backend"
        )
        
        training_enabled = bool(self.etrade_client and SessionStateManager.get_symbols())
        training_in_progress = st.session_state.get(SessionStateManager.TRAINING_STATUS, False)
        
        if st.sidebar.button("Train New Model", 
                           disabled=not training_enabled or training_in_progress,
                           help="Train new model with current data",
                           key="train_model_btn"):
            if training_enabled:
                self._handle_model_training_with_backend(training_backend)
            else:
                st.sidebar.warning("⚠️ Connect to E*Trade and add symbols first")
        
        if training_in_progress:
            st.sidebar.info("🔄 Training in progress...")
    
    def _handle_model_training_with_backend(self, backend: str) -> None:
        """Handle training with backend selection."""
        if not self.training_manager:
            self.training_manager = TrainingManager(
                etrade_client=self.etrade_client,
                model_dir="models/"
            )
        
        symbols = SessionStateManager.get_symbols()
        
        try:
            st.session_state[SessionStateManager.TRAINING_STATUS] = True
            
            with st.spinner(f"Training with {backend}... This may take several minutes"):
                if backend.startswith("MLPipeline"):
                    # Use complete MLPipeline
                    config = MLConfig(
                        seq_len=10,
                        epochs=20,
                        batch_size=32,
                        learning_rate=0.001,
                        test_size=0.2,
                        random_state=42,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        model_dir=Path("models/"),
                        symbols=symbols
                    )
                    result = self.training_manager.train_model(
                        symbols=symbols,
                        backend="MLPipeline",
                        config=config
                    )
                else:
                    # Use direct trainer
                    config = TrainingConfig(
                        epochs=20,
                        seq_len=10,
                        learning_rate=0.001,
                        batch_size=32,
                        validation_split=0.2,
                        early_stopping_patience=5,
                        min_patterns=100
                    )
                    result = self.training_manager.train_model(
                        symbols=symbols,
                        backend="PatternModelTrainer",
                        config=config
                    )
                
                if result.success:
                    st.session_state[SessionStateManager.MODEL] = result.model
                    self.model_manager.current_model = result.model
                    SessionStateManager.set_flag(SessionStateManager.TRAINING_COMPLETE)
                    st.session_state['training_metrics'] = result.metrics
                    logger.info(f"{result.backend} training completed with {result.data_points} data points")
                else:
                    st.sidebar.error(f"❌ {result.backend} training failed: {result.error_message}")
                    
        except Exception as e:
            logger.exception(f"Training error: {e}")
            st.sidebar.error(f"❌ Training failed: Unexpected error")
        finally:
            st.session_state[SessionStateManager.TRAINING_STATUS] = False

    @handle_exceptions
    def _initialize_etrade_client(self, credentials: Dict[str, str]) -> None:
        """Initialize E*Trade client with comprehensive error handling."""
        try:
            # Check if connection status actually changed
            current_status = st.session_state.get(SessionStateManager.CONNECTION_STATUS, False)
            
            # Validate credentials before creating client
            if not self._validate_credential_format(credentials):
                raise DashboardError("Invalid credential format")
            
            # Create client with timeout and retry logic
            self.etrade_client = create_etrade_client(credentials)
            
            if self.etrade_client:
                # Test connection
                try:
                    # Simple test call to verify connection
                    test_symbols = SessionStateManager.get_symbols()
                    if test_symbols:
                        test_quote = self.etrade_client.get_quote(test_symbols[0])
                        if test_quote.empty:
                            logger.warning("Connection test returned empty data")
                except Exception as e:
                    logger.warning(f"Connection test failed: {e}")
                
                new_status = True
                st.session_state[SessionStateManager.CONNECTION_STATUS] = new_status
                
                # Only set flag if status actually changed
                if new_status != current_status:
                    SessionStateManager.set_flag(SessionStateManager.CONNECTION_CHANGED)
                
                logger.info("E*Trade client initialized successfully")
            else:
                new_status = False
                st.session_state[SessionStateManager.CONNECTION_STATUS] = new_status
                
                # Only set flag if status actually changed
                if new_status != current_status:
                    SessionStateManager.set_flag(SessionStateManager.CONNECTION_CHANGED)
                
                st.error("❌ Failed to initialize E*Trade client")
                
        except DashboardError as e:
            logger.error(f"E*Trade client initialization failed: {e}")
            new_status = False
            st.session_state[SessionStateManager.CONNECTION_STATUS] = new_status
            
            # Only set flag if status actually changed
            if new_status != current_status:
                SessionStateManager.set_flag(SessionStateManager.CONNECTION_CHANGED)
            
            st.error(f"❌ Connection failed: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error initializing E*Trade client: {e}")
            new_status = False
            st.session_state[SessionStateManager.CONNECTION_STATUS] = new_status
            
            # Only set flag if status actually changed
            if new_status != current_status:
                SessionStateManager.set_flag(SessionStateManager.CONNECTION_CHANGED)
            
            st.error("❌ Connection failed: Unexpected error")

    @handle_exceptions
    def _place_order(self, symbol: str, side: str, quantity: int, order_type: str, 
                     limit_price: Optional[float], risk_params: RiskParameters) -> None:
        """Place an order with comprehensive risk validation."""
        try:
            # Risk management validation
            if not self.risk_manager.validate_order(symbol, quantity, side):
                st.error("❌ Order rejected by risk management")
                return
            
            # Calculate position size validation
            account_value = self._get_account_value()
            if account_value:
                position_value = quantity * (limit_price or self._get_current_price(symbol))
                position_pct = position_value / account_value
                
                if position_pct > risk_params.max_position_size:
                    st.error(f"❌ Position size {position_pct:.1%} exceeds limit {risk_params.max_position_size:.1%}")
                    return
            
            # Place order
            with st.spinner("Placing order..."):
                order_result = self.etrade_client.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type=order_type,
                    limit_price=limit_price
                )
                
                if order_result:
                    # Set flag instead of immediate success message
                    SessionStateManager.set_flag(SessionStateManager.ORDER_PLACED)
                    
                    # Store order result for display
                    st.session_state['last_order_result'] = order_result
                    
                    # Send notification
                    self.notifier.send_order_notification({
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "order_type": order_type,
                        "limit_price": limit_price,
                        "timestamp": datetime.now().isoformat(),
                        "result": order_result
                    })
                else:
                    st.error("❌ Order placement failed")
                    
        except Exception as e:
            logger.error(f"Order placement error: {e}")
            st.error(f"❌ Order failed: {e}")

    def _render_price_alerts(self) -> None:
        """Render price alert configuration."""
        alert_cols = st.columns([2, 1, 1, 1])
        
        symbols = SessionStateManager.get_symbols()
        symbol = alert_cols[0].selectbox("Symbol", symbols, key="price_alert_symbol")
        condition = alert_cols[1].selectbox("Condition", ["Above", "Below"], key="price_alert_condition")
        
        # Get current price for reference
        current_price = self._get_current_price(symbol) if symbol else 0.0
        
        price_threshold = alert_cols[2].number_input(
            f"Price (Current: ${current_price:.2f})",
            min_value=0.01,
            value=current_price if current_price > 0 else 100.0,
            step=0.01,
            key="price_alert_threshold"
        )
        
        if alert_cols[3].button("Add Price Alert", key="add_price_alert_btn"):
            try:
                success = self.alert_manager.add_price_alert(
                    symbol, condition.lower(), price_threshold
                )
                if success:
                    # Flag is set in alert_manager.add_price_alert()
                    # Don't call st.rerun() here
                    pass
            except Exception as e:
                st.error(f"❌ Failed to add alert: {e}")

    def _render_pattern_alerts(self) -> None:
        """Render pattern alert configuration."""
        pattern_cols = st.columns([2, 2, 1])
        
        symbols = SessionStateManager.get_symbols()
        symbol = pattern_cols[0].selectbox("Symbol", symbols, key="pattern_alert_symbol")
        
        available_patterns = CandlestickPatterns.get_pattern_names()
        pattern = pattern_cols[1].selectbox("Pattern", available_patterns, key="pattern_alert_pattern")
        
        if pattern_cols[2].button("Add Pattern Alert", key="add_pattern_alert_btn"):
            try:
                success = self.alert_manager.add_pattern_alert(symbol, pattern)
                if success:
                    # Flag is set in alert_manager.add_pattern_alert()
                    # Don't call st.rerun() here
                    pass
                else:
                    st.info(f"ℹ️ Alert already exists for {pattern} on {symbol}")
            except Exception as e:
                st.error(f"❌ Failed to add pattern alert: {e}")

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

    def _export_settings(self) -> None:
        """Export current settings as JSON."""
        try:
            # Collect exportable settings (exclude sensitive data)
            settings = {
                "refresh_interval": st.session_state.get('refresh_interval', DEFAULT_REFRESH_INTERVAL),
                "chart_periods": st.session_state.get('chart_periods', "1 day"),
                "enable_data_validation": st.session_state.get('enable_data_validation', True),
                "require_order_confirmation": st.session_state.get('require_order_confirmation', True),
                "paper_trading": st.session_state.get('paper_trading', True),
                "model_confidence_threshold": st.session_state.get('model_confidence_threshold', 0.7),
                "pattern_sensitivity": st.session_state.get('pattern_sensitivity', "Medium"),
                "auto_retrain": st.session_state.get('auto_retrain', False),
                "session_timeout": st.session_state.get('session_timeout', 60),
                "api_rate_limit": st.session_state.get('api_rate_limit', 60),
                "log_level": st.session_state.get('log_level', "INFO"),
                "enable_trade_logging": st.session_state.get('enable_trade_logging', True),
                "log_retention_days": st.session_state.get('log_retention_days', 30),
                "debug_mode": st.session_state.get('debug_mode', False),
                "exported_at": datetime.now().isoformat(),
                "symbols": SessionStateManager.get_symbols(),
                "risk_params": st.session_state.get(SessionStateManager.RISK_PARAMS, {})
            }
            
            import json
            settings_json = json.dumps(settings, indent=2)
            
            st.download_button(
                label="📥 Download Settings",
                data=settings_json,
                file_name=f"etrade_dashboard_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="export_settings_btn"
            )
            
        except Exception as e:
            logger.error(f"Settings export error: {e}")
            st.error("Failed to export settings")

    def _import_settings(self, uploaded_file) -> None:
        """Import settings from uploaded JSON file."""
        try:
            import json
            
            settings = json.load(uploaded_file)
            
            # Validate settings structure
            if not isinstance(settings, dict):
                raise ValueError("Invalid settings format")
            
            # Apply non-sensitive settings
            safe_settings = [
                'refresh_interval', 'chart_periods', 'enable_data_validation',
                'require_order_confirmation', 'paper_trading', 'model_confidence_threshold',
                'pattern_sensitivity', 'auto_retrain', 'session_timeout', 'api_rate_limit',
                'log_level', 'enable_trade_logging', 'log_retention_days', 'debug_mode'
            ]
            
            applied_count = 0
            for key in safe_settings:
                if key in settings:
                    st.session_state[key] = settings[key]
                    applied_count += 1
            
            # Import symbols if available
            if 'symbols' in settings and settings['symbols']:
                try:
                    SessionStateManager.set_symbols(settings['symbols'])
                    applied_count += 1
                except Exception as e:
                    logger.warning(f"Failed to import symbols: {e}")
            
            # Import risk parameters if available
            if 'risk_params' in settings and settings['risk_params']:
                try:
                    risk_params = RiskParameters(**settings['risk_params'])
                    SessionStateManager.set_risk_params(risk_params)
                    applied_count += 1
                except Exception as e:
                    logger.warning(f"Failed to import risk parameters: {e}")
            
            # Set flag instead of immediate success message and rerun
            SessionStateManager.set_flag(SessionStateManager.SETTINGS_CHANGED)
            st.session_state['settings_import_count'] = applied_count
            
        except Exception as e:
            logger.error(f"Settings import error: {e}")
            st.error(f"Failed to import settings: {e}")

    def _reset_settings(self) -> None:
        """Reset all settings to defaults."""
        try:
            # List of settings to reset
            settings_to_reset = [
                'refresh_interval', 'chart_periods', 'enable_data_validation',
                'require_order_confirmation', 'paper_trading', 'model_confidence_threshold',
                'pattern_sensitivity', 'auto_retrain', 'session_timeout', 'api_rate_limit',
                'log_level', 'enable_trade_logging', 'log_retention_days', 'debug_mode'
            ]
            
            # Reset to default values
            defaults = {
                'refresh_interval': DEFAULT_REFRESH_INTERVAL,
                'chart_periods': "1 day",
                'enable_data_validation': True,
                'require_order_confirmation': True,
                'paper_trading': True,
                'model_confidence_threshold': 0.7,
                'pattern_sensitivity': "Medium",
                'auto_retrain': False,
                'session_timeout': 60,
                'api_rate_limit': 60,
                'log_level': "INFO",
                'enable_trade_logging': True,
                'log_retention_days': 30,
                'debug_mode': False
            }
            
            for key in settings_to_reset:
                if key in defaults:
                    st.session_state[key] = defaults[key]
                elif key in st.session_state:
                    del st.session_state[key]
            
            # Reset risk parameters to defaults
            default_risk = RiskParameters()
            SessionStateManager.set_risk_params(default_risk)
            
            # Set flag instead of immediate success message
            SessionStateManager.set_flag(SessionStateManager.SETTINGS_CHANGED)
            
        except Exception as e:
            logger.error(f"Settings reset error: {e}")
            st.error("Failed to reset settings")


# ...existing code...

if __name__ == "__main__":
    """Main entry point for the trading dashboard application."""
    try:
        # Initialize and run the dashboard
        dashboard = TradingDashboard()
        dashboard.run()
        
    except Exception as e:
        logger.exception(f"Fatal dashboard error: {e}")
        st.error("🚨 Fatal Error: Dashboard failed to start")
        st.error(f"Error details: {e}")
        
        # Emergency fallback UI
        st.markdown("### 🔧 Emergency Options")
        if st.button("🔄 Restart Dashboard", key="emergency_restart"):
            st.cache_data.clear()
            st.rerun()
        
        if st.button("🗑️ Clear Session State", key="emergency_clear"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
