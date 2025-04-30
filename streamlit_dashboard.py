import logging
from typing import List, Optional, Dict
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objs as go

from etrade_candlestick_bot import ETradeClient
from patterns import CandlestickPatterns
from models.pattern_nn import PatternNN
from train.trainer import train_pattern_model
from utils.risk_manager import RiskManager
from utils.notifier import Notifier
from utils.indicators import TechnicalIndicators

# --- Constants for Session State Keys ---
SESSION_KEYS = {
    "initialized": "initialized",
    "model": "model",
    "training": "training",
    "symbols": "symbols",
    "class_names": "class_names",
    "risk_params": "risk_params",
}

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Utility: Cached Pattern Detection ---
@st.cache_data(show_spinner=False, ttl=300)
def detect_patterns(df: pd.DataFrame) -> List[str]:
    return CandlestickPatterns.detect_patterns(df)


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

        env = st.sidebar.radio("Select Environment", ["Sandbox", "Live"], index=0)
        live_ok = False
        if env == "Live":
            live_ok = st.sidebar.checkbox("I confirm I want to use live trading")
            st.sidebar.error("⚠️ LIVE TRADING ENABLED - Using real money!")
        else:
            st.sidebar.success("🔒 Safe Mode - Using Sandbox environment")

        creds = {
            'consumer_key': st.sidebar.text_input("Consumer Key", type="password"),
            'consumer_secret': st.sidebar.text_input("Consumer Secret", type="password"),
            'oauth_token': st.sidebar.text_input("OAuth Token", type="password"),
            'oauth_token_secret': st.sidebar.text_input("OAuth Token Secret", type="password"),
            'account_id': st.sidebar.text_input("Account ID")
        }
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
        symbols_str = ",".join(self.state.symbols)
        input_str = st.sidebar.text_input("Symbols (comma-separated)", value=symbols_str)
        if st.sidebar.button("Update Symbols"):
            new_syms = [s.strip().upper() for s in input_str.split(",") if s.strip()]
            st.session_state[SESSION_KEYS["symbols"]] = new_syms
            st.sidebar.success(f"Symbols updated: {', '.join(new_syms)}")

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

    def render_main_content(self):
        st.title("📈 E*Trade Candlestick Strategy Dashboard")
        self._render_metrics_row()
        self._render_positions_section()

        for symbol in self.state.symbols:
            self._render_symbol_section(symbol)

    def _render_metrics_row(self):
        st.subheader("📊 Strategy Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Watched Symbols", len(self.state.symbols))
        col2.metric("Patterns Tracked", len(st.session_state[SESSION_KEYS['class_names']]))
        col3.metric("Last Sync", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def _render_positions_section(self):
        st.subheader("Current Positions")
        with st.spinner("Fetching positions..."):
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

        fig.update_layout(title=f"{symbol} 5-min Candlestick Chart", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    def _analyze_patterns(self, df: pd.DataFrame, symbol: str):
        all_patterns = detect_patterns(df)
        st.write(f"Detected Patterns for {symbol}: {', '.join(all_patterns) or 'None'}")

    def _render_trading_controls(self, symbol: str):
        st.markdown("### Trading Controls")
        col1, col2 = st.columns(2)
        order_type = col1.selectbox("Order Type", ["MARKET", "LIMIT"], key=f"{symbol}_order_type")
        quantity = col1.number_input("Quantity", min_value=1, key=f"{symbol}_quantity")

        limit_price = None
        if order_type == "LIMIT":
            limit_price = col2.number_input("Limit Price", min_value=0.01, key=f"{symbol}_limit_price")

        if col2.button("Place Buy Order", key=f"{symbol}_buy"):
            self._place_order(symbol, "BUY", quantity, order_type, limit_price)
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

    def run(self):
        creds = self.render_sidebar()
        if not creds:
            st.info("Enter your E*Trade credentials and confirm environment above to begin.")
            return

        try:
            self.client = ETradeClient(**creds)
            self.render_main_content()
        except Exception as e:
            logger.error(f"Failed to initialize dashboard: {e}")
            st.error(f"Failed to initialize dashboard: {e}")


if __name__ == "__main__":
    Dashboard().run()
