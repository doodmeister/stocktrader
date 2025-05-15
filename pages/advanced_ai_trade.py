import os, sys
from pathlib import Path
import functools
def in_docker() -> bool:
    return os.path.exists('/.dockerenv')
# figure out where “.` actually is
PROJECT_ROOT = Path('/app') if in_docker() else Path(__file__).resolve().parent
# add it so `import stocktrader…` works
sys.path.insert(0, str(PROJECT_ROOT))
from utils.logger import setup_logger
from typing import List, Optional, Dict
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from utils.config.notification_settings_ui import render_notification_settings
from utils.etrade_candlestick_bot import ETradeClient
from patterns.patterns import CandlestickPatterns
from patterns.patterns_nn import PatternNN
from utils.deprecated.deeplearning_trainer_v1 import train_pattern_model
from utils.technicals.risk_manager import RiskManager
from utils.notifier import Notifier
from utils.technicals.indicators import TechnicalIndicators
from utils.etrade_client_factory import create_etrade_client
from utils.dashboard_utils import initialize_dashboard_session_state
from utils.security import get_api_credentials

import os
from pathlib import Path

# --- Logging Configuration ---
logger = setup_logger(__name__)

def in_docker() -> bool:
    # Docker injects this file at container runtime
    return os.path.exists('/.dockerenv')

# Pick the right project root:
if in_docker():
    PROJECT_ROOT = Path('/app')
else:
    # On your host, assume the script lives in the project root
    PROJECT_ROOT = Path(__file__).resolve().parent

    sys.path.insert(0, str(PROJECT_ROOT))
  
# --- Constants for Session State Keys ---
SESSION_KEYS = {
    "initialized": "initialized",
    "model": "model",
    "training": "training",
    "symbols": "symbols",
    "class_names": "class_names",
    "risk_params": "risk_params",
}

# --- Utility: Cached Pattern Detection ---
@st.cache_data(show_spinner=False, ttl=300)
def detect_patterns(df: pd.DataFrame) -> List[str]:
    return CandlestickPatterns.detect_patterns(df)


def handle_streamlit_exception(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as e:
            logger.exception(f"Error in {method.__name__}: {e}")
            st.error(f"An error occurred: {str(e)}")
            return None
    return wrapper


class DashboardState:
    def __init__(self):
        if not st.session_state.get(SESSION_KEYS["initialized"], False):
            self._initialize_state()

    def _initialize_state(self):
        st.session_state[SESSION_KEYS["initialized"]] = True
        st.session_state[SESSION_KEYS["model"]] = None
        st.session_state[SESSION_KEYS["training"]] = False
        st.session_state[SESSION_KEYS["symbols"]] = ["AAPL", "MSFT"]
        st.session_state[SESSION_KEYS["class_names"]] = [
            "Hammer", "Bullish Engulfing", "Doji", "Morning Star",
            "Morning Doji Star", "Piercing Pattern", "Bullish Harami",
            "Three White Soldiers", "Inverted Hammer", "Bullish Belt Hold",
            "Bullish Abandoned Baby", "Three Inside Up", "Rising Window"
        ]
        st.session_state[SESSION_KEYS["risk_params"]] = {
            'max_position_size': 0.02,
            'stop_loss_atr': 2.0,
        }
        # Add alert configurations
        st.session_state["alerts"] = {
            "price_alerts": {},  # {symbol: {"above": price, "below": price}}
            "pattern_alerts": {},  # {symbol: [pattern_names]}
            "triggered_alerts": [],  # Store recently triggered alerts
            "notification_channels": {
                "email": {"enabled": False, "address": ""},
                "sms": {"enabled": False, "number": ""},
                "dashboard": {"enabled": True},
            }
        }

    @property
    def symbols(self) -> List[str]:
        return st.session_state[SESSION_KEYS["symbols"]]

    @property
    def is_training(self) -> bool:
        return st.session_state[SESSION_KEYS["training"]]

    def set_training(self, value: bool):
        st.session_state[SESSION_KEYS["training"]] = value

    def set_model(self, model: PatternNN):
        st.session_state[SESSION_KEYS["model"]] = model


class Dashboard:
    def __init__(self):
        st.set_page_config(
            page_title="E*Trade Candlestick Bot Dashboard",
            layout="wide"
        )
        self.state = DashboardState()
        self.risk_manager = RiskManager()
        self.indicators = TechnicalIndicators()
        self.notifier = Notifier()
        self.client: Optional[ETradeClient] = None

    def render_sidebar(self) -> Optional[Dict[str, str]]:
        st.sidebar.title("⚙️ Configuration")
        st.sidebar.markdown("### 🔒 Trading Environment")

        # Always load credentials from .env using get_api_credentials
        env_creds = get_api_credentials()

        env = st.sidebar.radio("Select Environment", ["Sandbox", "Live"], index=0)
        live_ok = False
        if env == "Live":
            live_ok = st.sidebar.checkbox("I confirm I want to use live trading")
            st.sidebar.error("⚠️ LIVE TRADING ENABLED - Using real money!")
        else:
            st.sidebar.success("🔒 Safe Mode - Using Sandbox environment")

        # Pre-populate fields with values from .env, allow user to override
        creds = {
            'consumer_key': st.sidebar.text_input("Consumer Key", value=env_creds.get('consumer_key', ''), type="password"),
            'consumer_secret': st.sidebar.text_input("Consumer Secret", value=env_creds.get('consumer_secret', ''), type="password"),
            'oauth_token': st.sidebar.text_input("OAuth Token", value=env_creds.get('oauth_token', ''), type="password"),
            'oauth_token_secret': st.sidebar.text_input("OAuth Token Secret", value=env_creds.get('oauth_token_secret', ''), type="password"),
            'account_id': st.sidebar.text_input("Account ID", value=env_creds.get('account_id', ''))
        }

        # Display confirmation of loaded values
        if all(creds.values()):
            st.sidebar.success("✅ Credentials loaded from .env file or user input")

        if all(creds.values()) and (env == "Sandbox" or live_ok):
            creds['sandbox'] = (env == "Sandbox")
        else:
            return None

        self._render_symbol_manager()
        self._render_training_controls()
        self._render_risk_controls()

        return creds

    def _render_symbol_manager(self):
        st.sidebar.markdown("### 🏷️ Symbols")
        
        # Import validation functions from utils.validation
        from utils.config.validation import validate_symbol, sanitize_input
        
        symbols_str = ",".join(self.state.symbols)
        input_str = st.sidebar.text_input("Symbols (comma-separated)", value=symbols_str)
        
        if st.sidebar.button("Update Symbols"):
            valid_symbols = []
            invalid_symbols = []
            
            # Process each symbol entry
            for raw_symbol in input_str.split(","):
                if not raw_symbol.strip():
                    continue
                    
                try:
                    # First sanitize the input
                    sanitized = sanitize_input(raw_symbol)
                    
                    # Then validate the symbol format
                    valid_symbol = validate_symbol(sanitized)
                    valid_symbols.append(valid_symbol)
                    
                except ValueError as e:
                    invalid_symbols.append(raw_symbol.strip())
                    st.sidebar.warning(f"Invalid symbol: '{raw_symbol.strip()}' - {str(e)}")
            
            # Update session state with valid symbols
            if valid_symbols:
                st.session_state[SESSION_KEYS["symbols"]] = valid_symbols
                st.sidebar.success(f"Symbols updated: {', '.join(valid_symbols)}")
            else:
                st.sidebar.error("No valid symbols provided.")
                
            # Show warning about invalid symbols if any were found
            if invalid_symbols:
                st.sidebar.info(f"Skipped {len(invalid_symbols)} invalid symbols.")

    def _render_training_controls(self):
        st.sidebar.markdown("### 🛠️ Training Hyperparameters")
        epochs = st.sidebar.number_input("Epochs", 1, 100, 10)
        seq_len = st.sidebar.number_input("Sequence Length", 2, 50, 10)
        lr = st.sidebar.number_input("Learning Rate", 1e-5, 1e-1, 1e-3, format="%.5f")

        if st.sidebar.button("Train Neural Model"):
            if self.state.is_training:
                st.sidebar.warning("Training already in progress.")
            else:
                self._handle_model_training(epochs, seq_len, lr)

    def _handle_model_training(self, epochs: int, seq_len: int, lr: float):
        self.state.set_training(True)
        with st.spinner("Training model... this may take a while"):
            try:
                model = PatternNN()
                trained = train_pattern_model(
                    client=self.client,
                    symbols=self.state.symbols,
                    model=model,
                    epochs=epochs,
                    seq_len=seq_len,
                    learning_rate=lr
                )
                self.state.set_model(trained)
                st.sidebar.success("Training complete!")
            except Exception as e:
                logger.error(f"Training failed: {e}")
                st.sidebar.error(f"Training failed: {e}")
            finally:
                self.state.set_training(False)

    def _render_risk_controls(self):
        """Render risk management controls in the sidebar."""
        st.sidebar.markdown("### ⚠️ Risk Management")
        
        # Get current risk parameters from session state or use defaults
        risk_params = st.session_state.get(SESSION_KEYS["risk_params"], {
            'max_position_size': 0.25,  # Default to 25%
            'stop_loss_atr': 1.5        # Default ATR multiplier
        })
        
        # Create input fields for each risk parameter
        new_max_position = st.sidebar.slider(
            "Max Position Size (%)", 
            min_value=1.0, 
            max_value=50.0, 
            value=risk_params['max_position_size'] * 100,
            step=1.0,
            help="Maximum position size as percentage of portfolio"
        ) / 100.0
        
        new_stop_loss_atr = st.sidebar.slider(
            "Stop Loss (ATR multiplier)",
            min_value=0.5,
            max_value=5.0,
            value=risk_params['stop_loss_atr'],
            step=0.1,
            help="Stop loss placement as a multiple of Average True Range"
        )
        
        # Update risk parameters if they've changed
        if (new_max_position != risk_params['max_position_size'] or
            new_stop_loss_atr != risk_params['stop_loss_atr']):
            
            # Update session state
            st.session_state[SESSION_KEYS["risk_params"]] = {
                'max_position_size': new_max_position,
                'stop_loss_atr': new_stop_loss_atr,
            }
            
            # Create a new RiskManager instance with updated parameters
            self.risk_manager = RiskManager(max_position_pct=new_max_position)
            
            st.sidebar.success("Risk parameters updated!")

    def render_main_content(self):
        st.title("📈 E*Trade Candlestick Strategy Dashboard")
        
        # Check for alerts on each refresh
        self._check_for_alerts()
        
        # Show tabs with the new alerts tab
        tabs = st.tabs(["Overview", "Trading", "Alerts & Notifications", "Analysis", "Settings"])
        
        with tabs[0]:
            self._render_metrics_row()
            self._render_positions_section()
            
            for symbol in self.state.symbols:
                self._render_symbol_section(symbol)
                
        # Trading tab contents would go here
        with tabs[1]:
            st.header("Trading")
            st.info("Trading functions")
        
        # Render the alerts tab
        with tabs[2]:
            self._render_alerts_tab()
        
        # Analysis tab contents would go here  
        with tabs[3]:
            st.header("Analysis")
            st.info("Analysis tools")
        
        # Settings tab contents would go here
        with tabs[4]:
            st.header("Settings")
            st.info("Additional settings")

    def _render_metrics_row(self):
        st.subheader("📊 Strategy Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Watched Symbols", len(self.state.symbols))
        col2.metric("Patterns Tracked", len(st.session_state[SESSION_KEYS['class_names']]))
        col3.metric("Last Sync", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def _render_positions_section(self):
        st.subheader("Current Positions")
        with st.spinner("Fetching positions..."):
            if not self.client:
                st.info("Demo mode: No E*Trade connection. Showing sample positions.")
                # Example mock data
                demo_positions = [
                    {"symbol": "AAPL", "market_value": 10000, "day_pl": 120},
                    {"symbol": "MSFT", "market_value": 8000, "day_pl": -50},
                ]
                df = pd.DataFrame(demo_positions)
                st.dataframe(df)
                total_val = df['market_value'].sum()
                day_pl = df['day_pl'].sum()
                st.metric("Total Position Value", f"${total_val:,.2f}", f"{day_pl:+,.2f}")
                return
            try:
                positions = self.client.get_positions()
                if positions:
                    df = pd.DataFrame(positions)
                    st.dataframe(df)
                    total_val = df['market_value'].sum()
                    day_pl = df['day_pl'].sum()
                    st.metric("Total Position Value", f"${total_val:,.2f}", f"{day_pl:+,.2f}")
                else:
                    st.info("No open positions")
            except Exception as e:
                logger.error(f"Failed to fetch positions: {e}")
                st.error(f"Failed to fetch positions: {e}")

    def _render_symbol_section(self, symbol: str):
        st.markdown(f"---\n### {symbol}")
        with st.spinner(f"Loading data for {symbol}..."):
            if not self.client:
                st.info("Demo mode: No E*Trade connection. Showing sample chart data.")
                # Generate mock candle data
                import numpy as np
                import pandas as pd
                idx = pd.date_range(end=datetime.now(), periods=30, freq="5min")
                prices = np.cumsum(np.random.randn(len(idx))) + 150
                df = pd.DataFrame({
                    "open": prices + np.random.rand(len(idx)),
                    "high": prices + np.random.rand(len(idx)) + 1,
                    "low": prices - np.random.rand(len(idx)) - 1,
                    "close": prices + np.random.rand(len(idx)),
                }, index=idx)
                self._render_chart(df, symbol)
                st.info("Pattern analysis and trading controls are disabled in demo mode.")
                return
            try:
                df = self.client.get_candles(symbol, interval="5min", days=1)
                if df.empty or not {"open", "high", "low", "close"}.issubset(df.columns):
                    st.warning(f"No valid data for {symbol}")
                    return

                self._render_chart(df, symbol)
                self._analyze_patterns(df, symbol)
                self._render_trading_controls(symbol)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                st.error(f"Error processing {symbol}: {e}")

    def _render_chart(self, df: pd.DataFrame, symbol: str):
        fig = go.Figure(data=[
            go.Candlestick(
                x=df.index,
                open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                name="Candles"
            )
        ])

        # Overlay patterns
        for i in range(2, len(df)):
            window = df.iloc[i-2:i+1]
            patterns = detect_patterns(window)
            if patterns:
                fig.add_trace(go.Scatter(
                    x=[window.index[-1]], y=[window['close'].iloc[-1]],
                    mode='markers+text',
                    marker=dict(size=10, color='red'),
                    text=[" ".join(patterns)],
                    textposition="top center",
                    name=f"{symbol} Pattern"
                ))
        col1, col2, col3 = st.columns(3)  # Define columns
        quantity = 1  # Define a default quantity or fetch it dynamically
        order_type = "LIMIT"  # Define a default order type or fetch it dynamically
        limit_price = None  # Define a default limit price or fetch it dynamically
        if col2.button("Place Sell Order", key=f"{symbol}_sell"):
            self._place_order(symbol, "SELL", quantity, order_type, limit_price)

    def _place_order(self, symbol: str, side: str, quantity: int, order_type: str, limit_price: Optional[float] = None):
        with st.spinner("Placing order..."):
            try:
                if not self.risk_manager.validate_order(symbol, quantity, side):
                    st.error("Order rejected by risk management rules")
                    return

                entry_price = self.indicators.get_entry_price(symbol)
                stop_loss = self.indicators.get_stop_loss(symbol)

                order_result = self.client.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type=order_type,
                    limit_price=limit_price,
                    stop_loss=stop_loss
                )

                st.success(f"Order placed successfully: {order_result}")
                self.notifier.send_order_notification(order_result)
            except Exception as e:
                logger.error(f"Failed to place order: {e}")
                st.error(f"Failed to place order: {e}")

    def _setup_auto_refresh(self, interval_seconds=60):
        if "last_refresh" not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        
        # Check if time to refresh
        now = datetime.now()
        elapsed = (now - st.session_state.last_refresh).total_seconds()
        
        # Auto refresh if interval has passed
        if elapsed >= interval_seconds:
            st.session_state.last_refresh = now
            st.experimental_rerun()

    def _render_alerts_tab(self):
        """Render the alerts and notifications tab."""
        st.header("⚠️ Alerts & Notifications")
        
        # Status indicator
        status_col1, status_col2 = st.columns(2)
        
        # System status indicators
        with status_col1.container():
            st.subheader("System Status")
            
            # Check API connection
            api_connected = self.client is not None
            api_status = "🟢 Connected" if api_connected else "🔴 Disconnected"
            
            # Check if model is trained
            model_trained = st.session_state.get(SESSION_KEYS["model"]) is not None
            model_status = "🟢 Trained" if model_trained else "🔴 Not Trained"
            
            # Last refresh time
            last_refresh = st.session_state.get("last_refresh", datetime.now())
            refresh_age = (datetime.now() - last_refresh).total_seconds()
            refresh_status = "🟢 Recent" if refresh_age < 300 else "🟠 Stale"
            
            # Display statuses
            status_df = pd.DataFrame({
                "Component": ["API Connection", "ML Model", "Data Refresh"],
                "Status": [api_status, model_status, refresh_status],
                "Last Updated": [datetime.now().strftime("%H:%M:%S"), 
                                "N/A" if not model_trained else "Unknown",
                                last_refresh.strftime("%H:%M:%S")]
            })
            st.dataframe(status_df, hide_index=True)
        
        # Alerts management
        with status_col2.container():
            st.subheader("Alert History")
            triggered = st.session_state["alerts"]["triggered_alerts"]
            if not triggered:
                st.info("No alerts have been triggered yet")
            else:
                alerts_df = pd.DataFrame(triggered[-10:])  # Show last 10 alerts
                st.dataframe(alerts_df, hide_index=True)
                if st.button("Clear Alert History"):
                    st.session_state["alerts"]["triggered_alerts"] = []
                    st.success("Alert history cleared")

        # Price alerts section
        st.subheader("📊 Price Alerts")
        
        # Select symbol for price alert
        alert_cols = st.columns([2, 1, 1, 1])
        symbol = alert_cols[0].selectbox("Symbol", self.state.symbols, key="price_alert_symbol")
        
        # Get current price
        current_price = 0
        try:
            if self.client:
                df = self.client.get_quote(symbol)
                if not df.empty:
                    current_price = df['last'][0]
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
        
        # Price conditions
        alert_type = alert_cols[1].radio("Condition", ["Above", "Below"], key="price_alert_condition")
        
        # Price threshold with current price as default
        price = alert_cols[2].number_input(
            f"Price (Current: ${current_price:.2f})", 
            min_value=0.01, 
            value=current_price,
            step=0.01,
            key="price_alert_value"
        )
        
        # Add alert button
        if alert_cols[3].button("Add Alert"):
            # Initialize price alerts for this symbol if needed
            if symbol not in st.session_state["alerts"]["price_alerts"]:
                st.session_state["alerts"]["price_alerts"][symbol] = {}
            
            # Add alert
            condition_key = alert_type.lower()
            st.session_state["alerts"]["price_alerts"][symbol][condition_key] = price
            st.success(f"Alert set for {symbol} {alert_type} ${price:.2f}")
        
        # Display current price alerts
        if not self.client:
            st.info("Demo mode: Price and pattern alerts are not active without E*Trade connection.")
            return
        price_alerts = st.session_state["alerts"]["price_alerts"]
        if price_alerts:
            alert_data = []
            for sym, conditions in price_alerts.items():
                for condition, threshold in conditions.items():
                    alert_data.append({
                        "Symbol": sym,
                        "Condition": condition.capitalize(),
                        "Price": f"${threshold:.2f}",
                        "Delete": False  # For deletion checkbox
                    })
            
            if alert_data:
                alert_df = pd.DataFrame(alert_data)
                edited_df = st.data_editor(alert_df, hide_index=True)
                
                # Handle deletions
                if not edited_df.equals(alert_df):
                    for i, row in edited_df.iterrows():
                        if row["Delete"]:
                            symbol = row["Symbol"]
                            condition = row["Condition"].lower()
                            if symbol in price_alerts and condition in price_alerts[symbol]:
                                del price_alerts[symbol][condition]
                                if not price_alerts[symbol]:  # If no more conditions for this symbol
                                    del price_alerts[symbol]
                                st.rerun()
        else:
            st.info("No price alerts configured")
        
        # Pattern alerts section
        st.subheader("📈 Pattern Alerts")
        
        pattern_cols = st.columns([2, 2, 1])
        pattern_symbol = pattern_cols[0].selectbox("Symbol", self.state.symbols, key="pattern_alert_symbol")
        available_patterns = st.session_state[SESSION_KEYS["class_names"]]
        selected_pattern = pattern_cols[1].selectbox("Pattern", available_patterns, key="pattern_alert_pattern")
        
        if pattern_cols[2].button("Add Pattern Alert"):
            # Initialize pattern alerts for this symbol if needed
            if pattern_symbol not in st.session_state["alerts"]["pattern_alerts"]:
                st.session_state["alerts"]["pattern_alerts"][pattern_symbol] = []
            
            # Add pattern to alerts if not already there
            if selected_pattern not in st.session_state["alerts"]["pattern_alerts"][pattern_symbol]:
                st.session_state["alerts"]["pattern_alerts"][pattern_symbol].append(selected_pattern)
                st.success(f"Alert set for {selected_pattern} pattern on {pattern_symbol}")
            else:
                st.info(f"Alert for {selected_pattern} on {pattern_symbol} already exists")
        
        # Display current pattern alerts
        pattern_alerts = st.session_state["alerts"]["pattern_alerts"]
        if pattern_alerts:
            pattern_data = []
            for sym, patterns in pattern_alerts.items():
                for pattern in patterns:
                    pattern_data.append({
                        "Symbol": sym,
                        "Pattern": pattern,
                        "Delete": False
                    })
            
            if pattern_data:
                pattern_df = pd.DataFrame(pattern_data)
                edited_pattern_df = st.data_editor(pattern_df, hide_index=True)
                
                # Handle deletions
                if not edited_pattern_df.equals(pattern_df):
                    for i, row in edited_pattern_df.iterrows():
                        if row["Delete"]:
                            symbol = row["Symbol"]
                            pattern = row["Pattern"]
                            if symbol in pattern_alerts and pattern in pattern_alerts[symbol]:
                                pattern_alerts[symbol].remove(pattern)
                                if not pattern_alerts[symbol]:  # If no more patterns for this symbol
                                    del pattern_alerts[symbol]
                                st.rerun()
        else:
            st.info("No pattern alerts configured")

        # Notification settings
    def _render_alerts_tab(self):
        # ...existing code...
        # Notification settings
        notification_channels = st.session_state["alerts"]["notification_channels"]
        render_notification_settings(
            notification_channels,
            self._update_notification_setting,
            self._send_test_notification
        )

    def _update_notification_setting(self, channel, setting, value):
        """Update a notification setting in session state."""
        st.session_state["alerts"]["notification_channels"][channel][setting] = value

    def _send_test_notification(self):
        """Send a test notification using the configured channels."""
        test_message = f"Test notification from E*Trade Bot at {datetime.now().strftime('%H:%M:%S')}"
        # Use Notifier to send to all configured channels
        try:
            self.notifier.send_order_notification({
                "symbol": "TEST",
                "side": "TEST",
                "quantity": 1,
                "filled_price": "N/A",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message": test_message
            })
            st.success("Test notification sent via all configured channels.")
        except Exception as e:
            logger.error(f"Failed to send test notification: {e}")
            st.error(f"Failed to send test notification: {e}")

    def _check_for_alerts(self):
        """Check if any alerts have been triggered and notify if needed."""
        if not self.client:
            return

        alerts = st.session_state["alerts"]
        triggered = []

        # Check price alerts
        for symbol, conditions in alerts["price_alerts"].items():
            try:
                df = self.client.get_quote(symbol)
                if df.empty:
                    continue
                    
                current_price = df['last'][0]
                
                if "above" in conditions and current_price > conditions["above"]:
                    msg = f"🔔 {symbol} price alert: ${current_price:.2f} above ${conditions['above']:.2f}"
                    triggered.append({"type": "price", "symbol": symbol, "message": msg, "time": datetime.now()})
                
                if "below" in conditions and current_price < conditions["below"]:
                    msg = f"🔔 {symbol} price alert: ${current_price:.2f} below ${conditions['below']:.2f}"
                    triggered.append({"type": "price", "symbol": symbol, "message": msg, "time": datetime.now()})
                    
            except Exception as e:
                logger.error(f"Error checking price alerts for {symbol}: {e}")
        
        # Check pattern alerts
        for symbol, patterns in alerts["pattern_alerts"].items():
            try:
                df = self.client.get_candles(symbol, interval="5min", days=1)
                if df.empty:
                    continue
                    
                detected_patterns = detect_patterns(df)
                
                for pattern in patterns:
                    if pattern in detected_patterns:
                        msg = f"🔔 {symbol} pattern alert: {pattern} detected"
                        triggered.append({"type": "pattern", "symbol": symbol, "message": msg, "time": datetime.now()})
                        
            except Exception as e:
                logger.error(f"Error checking pattern alerts for {symbol}: {e}")
        
        # Process triggered alerts
        for alert in triggered:
            # Add to alert history
            alerts["triggered_alerts"].append({
                "Time": alert["time"].strftime("%H:%M:%S"),
                "Symbol": alert["symbol"],
                "Type": alert["type"].capitalize(),
                "Message": alert["message"]
            })

            # Dashboard notification
            if alerts["notification_channels"]["dashboard"]["enabled"]:
                st.warning(alert["message"])

            # Use unified Notifier for email, SMS, Slack
            try:
                self.notifier.send_order_notification({
                    "symbol": alert["symbol"],
                    "side": alert["type"],
                    "quantity": "",
                    "filled_price": "",
                    "timestamp": alert["time"].strftime("%Y-%m-%d %H:%M:%S"),
                    "message": alert["message"]
                })
            except Exception as e:
                logger.error(f"Failed to send alert notification: {e}")

    def run(self):
        initialize_dashboard_session_state()
        creds = self.render_sidebar()
        if creds:
            try:
                self.client = create_etrade_client(creds)
                if not self.client:
                    st.error("Failed to initialize E*Trade client. Check credentials and try again.")
            except Exception as e:
                logger.error(f"Failed to initialize dashboard: {e}")
                st.error(f"Failed to initialize dashboard: {e}")
        else:
            self.client = None  # No credentials, operate in demo mode

        self.render_main_content()


if __name__ == "__main__":
    Dashboard().run()
