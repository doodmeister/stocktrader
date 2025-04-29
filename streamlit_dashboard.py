import logging
from typing import List, Optional, Dict
import streamlit as st
import pandas as pd
import torch
import threading
import plotly.graph_objs as go
from datetime import datetime

from etrade_candlestick_bot import ETradeClient, CandlestickPatterns
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
    """Manages Streamlit session state and configuration."""
    
    def __init__(self):
        if 'initialized' not in st.session_state:
            self._initialize_state()
            
    def _initialize_state(self):
        """Initialize default session state values."""
        st.session_state.update({
            'initialized': True,
            'model': None,
            'training': False,
            'symbols': ["AAPL", "MSFT"],
            'class_names': [
                "Hammer", "Bullish Engulfing", "Bearish Engulfing",
                "Doji", "Morning Star", "Evening Star"
            ],
            'risk_params': {
                'max_position_size': 0.02,  # 2% of portfolio
                'stop_loss_atr': 2.0,       # 2x ATR for stops
            }
        })

class Dashboard:
    """Main dashboard application class."""
    
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
        """Render sidebar components and return credentials if valid."""
        st.sidebar.title("⚙️ Configuration")
        
        credentials = {
            'consumer_key': st.sidebar.text_input("Consumer Key", type="password"),
            'consumer_secret': st.sidebar.text_input("Consumer Secret", type="password"),
            'oauth_token': st.sidebar.text_input("OAuth Token", type="password"),
            'oauth_token_secret': st.sidebar.text_input("OAuth Token Secret", type="password"),
            'account_id': st.sidebar.text_input("Account ID")
        }

        # Environment settings
        use_sandbox = st.sidebar.radio("Environment", ["Sandbox", "Live"], index=0) == "Sandbox"
        
        self._render_training_controls()
        self._render_risk_controls()
        self._render_symbol_manager()
        
        if all(credentials.values()):
            return {**credentials, 'sandbox': use_sandbox}
        return None

    def _render_training_controls(self):
        """Render model training controls."""
        st.sidebar.markdown("### 🔧 Training Hyperparameters")
        epochs = st.sidebar.number_input("Epochs", 1, 100, 10)
        seq_len = st.sidebar.number_input("Sequence Length", 2, 50, 10)
        lr = st.sidebar.number_input("Learning Rate", 1e-5, 1e-1, 1e-3, format="%.5f")

        if st.sidebar.button("Train Neural Model"):
            self._handle_model_training(epochs, seq_len, lr)

    def _handle_model_training(self, epochs: int, seq_len: int, lr: float):
        """Handle model training initiation."""
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

    def render_main_content(self, client: ETradeClient):
        """Render main dashboard content."""
        st.title("📊 E*Trade Candlestick Strategy Dashboard")
        
        # Render metrics row
        self._render_metrics_row(client)
        
        # Render charts and controls for each symbol
        for symbol in st.session_state.symbols:
            self._render_symbol_section(client, symbol)

    def _render_symbol_section(self, client: ETradeClient, symbol: str):
        """Render trading section for a single symbol."""
        st.markdown(f"---\n### {symbol}")
        
        try:
            # Fetch and validate data
            df = client.get_candles(symbol, interval="5min", days=1)
            if df.empty:
                st.warning(f"No data available for {symbol}")
                return

            # Render candlestick chart
            self._render_chart(df, symbol)
            
            # Pattern detection
            self._analyze_patterns(df, symbol, client)
            
            # Trading controls
            self._render_trading_controls(symbol, client)
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            st.error(f"Error processing {symbol}: {str(e)}")

    def run(self):
        """Main entry point for the dashboard application."""
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
