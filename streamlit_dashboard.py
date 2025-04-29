import logging 
from typing import List, Optional, Dict
import streamlit as st
import pandas as pd
import torch
import threading
import plotly.graph_objs as go
from datetime import datetime

from etrade_candlestick_bot import ETradeClient
from patterns import CandlestickPatterns
from models.pattern_nn import PatternNN
from training.trainer import train_pattern_model
from risk_manager import RiskManager
from notifier import Notifier
from indicators import TechnicalIndicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type definitions
PatternList = List[str]
Credentials = Dict[str, str]

class DashboardState:
    def __init__(self):
        if 'initialized' not in st.session_state:
            self._initialize_state()

    def _initialize_state(self):
        st.session_state.update({
            'initialized': True,
            'model': None,
            'training': False,
            'symbols': ["AAPL", "MSFT"],
            'class_names': [
                "Hammer", "Bullish Engulfing", "Doji", "Morning Star",
                "Morning Doji Star", "Piercing Pattern", "Bullish Harami",
                "Three White Soldiers", "Inverted Hammer", "Bullish Belt Hold",
                "Bullish Abandoned Baby", "Three Inside Up", "Rising Window"
            ],
            'risk_params': {
                'max_position_size': 0.02,
                'stop_loss_atr': 2.0,
            }
        })

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

    def render_sidebar(self) -> Optional[Credentials]:
        st.sidebar.title("⚙️ Configuration")

        credentials = {
            'consumer_key': st.sidebar.text_input("Consumer Key", type="password"),
            'consumer_secret': st.sidebar.text_input("Consumer Secret", type="password"),
            'oauth_token': st.sidebar.text_input("OAuth Token", type="password"),
            'oauth_token_secret': st.sidebar.text_input("OAuth Token Secret", type="password"),
            'account_id': st.sidebar.text_input("Account ID")
        }

        use_sandbox = st.sidebar.radio("Environment", ["Sandbox", "Live"], index=0) == "Sandbox"

        self._render_training_controls()
        self._render_risk_controls()
        self._render_symbol_manager()

        if all(credentials.values()):
            return {**credentials, 'sandbox': use_sandbox}
        return None

    def _render_training_controls(self):
        st.sidebar.markdown("### 🔧 Training Hyperparameters")
        epochs = st.sidebar.number_input("Epochs", 1, 100, 10)
        seq_len = st.sidebar.number_input("Sequence Length", 2, 50, 10)
        lr = st.sidebar.number_input("Learning Rate", 1e-5, 1e-1, 1e-3, format="%.5f")

        if st.sidebar.button("Train Neural Model"):
            self._handle_model_training(epochs, seq_len, lr)

    def _handle_model_training(self, epochs: int, seq_len: int, lr: float):
        if st.session_state.training:
            st.sidebar.warning("Training already in progress.")
            return

        try:
            st.session_state.training = True
            st.sidebar.info("Training started... this may take a while.")

            def train_async():
                try:
                    model = PatternNN()
                    trained = train_pattern_model(
                        self.client,
                        st.session_state.symbols,
                        model,
                        epochs=epochs,
                        seq_len=seq_len,
                        learning_rate=lr
                    )
                    st.session_state.model = trained
                    st.sidebar.success("Training complete!")
                except Exception as e:
                    logger.error(f"Training failed: {str(e)}")
                    st.sidebar.error(f"Training failed: {str(e)}")
                finally:
                    st.session_state.training = False

            threading.Thread(target=train_async, daemon=True).start()
        except Exception as e:
            logger.error(f"Failed to start training: {str(e)}")
            st.session_state.training = False
            st.sidebar.error(f"Failed to start training: {str(e)}")

    def _render_metrics_row(self, client: ETradeClient):
        st.subheader("📈 Strategy Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Watched Symbols", len(st.session_state.symbols))
        col2.metric("Patterns Tracked", len(st.session_state.class_names))
        col3.metric("Last Sync", datetime.now().strftime("%H:%M:%S"))

    def render_main_content(self, client: ETradeClient):
        st.title("📊 E*Trade Candlestick Strategy Dashboard")
        self._render_metrics_row(client)

        for symbol in st.session_state.symbols:
            self._render_symbol_section(client, symbol)

    def _render_symbol_section(self, client: ETradeClient, symbol: str):
        st.markdown(f"---\n### {symbol}")

        try:
            df = client.get_candles(symbol, interval="5min", days=1)
            if df.empty:
                st.warning(f"No data available for {symbol}")
                return

            self._render_chart(df, symbol)
            self._analyze_patterns(df, symbol)
            self._render_trading_controls(symbol, client)

        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            st.error(f"Error processing {symbol}: {str(e)}")

    def _render_chart(self, df: pd.DataFrame, symbol: str):
        fig = go.Figure(data=[
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Candles"
            )
        ])

        for i in range(2, len(df)):
            window = df.iloc[i-2:i+1]
            patterns = CandlestickPatterns.detect_patterns(window)
            if patterns:
                fig.add_trace(go.Scatter(
                    x=[window.index[-1]],
                    y=[window['close'].iloc[-1]],
                    mode='markers+text',
                    marker=dict(size=10, color='red'),
                    text=[" ".join(patterns)],
                    textposition="top center",
                    name=f"{symbol} Pattern"
                ))

        fig.update_layout(
            title=f"{symbol} 5-min Candlestick Chart",
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

    def _analyze_patterns(self, df: pd.DataFrame, symbol: str):
        patterns = CandlestickPatterns.detect_patterns(df)
        st.write(f"Detected Patterns for {symbol}: {', '.join(patterns) if patterns else 'None'}")

    def run(self):
        credentials = self.render_sidebar()

        if credentials:
            try:
                client = ETradeClient(**credentials)
                self.render_main_content(client)
            except Exception as e:
                logger.error(f"Failed to initialize dashboard: {str(e)}")
                st.error(f"Failed to initialize dashboard: {str(e)}")
        else:
            st.info("Enter your E*Trade credentials above to begin.")

if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.run()
