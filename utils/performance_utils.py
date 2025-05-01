"""
E*Trade Candlestick Trading Dashboard utility module.
Handles data visualization, pattern detection, and order execution.
"""
import logging
import functools
from typing import Dict, List, Optional, Tuple
import asyncio
from datetime import datetime
import streamlit as st
import pandas as pd
import torch
import plotly.graph_objs as go
from utils.etrade_candlestick_bot import ETradeClient
from patterns import CandlestickPatterns
from utils.patterns_nn import PatternNN
from train.trainer import train_pattern_model

@functools.lru_cache(maxsize=128)
def get_candles_cached(
    symbol: str,
    interval: str = "5min",
    days: int = 1
) -> pd.DataFrame:
    """
    Fetch OHLCV data via ETradeClient and cache up to 128 distinct calls.
    """
    client = ETradeClient()   # reads creds from .env
    return client.get_candles(symbol, interval=interval, days=days)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DashboardState:
    """Manages dashboard session state and configuration."""
    
    def __init__(self):
        self.initialize_session_state()
        
    @staticmethod
    def initialize_session_state():
        """Initialize or reset Streamlit session state variables."""
        if 'data' not in st.session_state:
            st.session_state.data = {}
        if 'model' not in st.session_state:
            st.session_state.model = None
            st.session_state.training = False
            st.session_state.class_names = [
                "Hammer", "Bullish Engulfing", "Bearish Engulfing", 
                "Doji", "Morning Star", "Evening Star"
            ]
        if 'symbols' not in st.session_state:
            st.session_state.symbols = ["AAPL", "MSFT"]

class DataManager:
    """Handles data fetching and caching operations."""

    async def fetch_all_candles(
        self, client: ETradeClient, symbols: List[str], interval: str, days: int
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple symbols asynchronously."""
        tasks = [
            asyncio.create_task(client.get_candles(symbol, interval=interval, days=days))
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching data for {symbol}: {result}")
            else:
                data[symbol] = result
        return data

    def __init__(self, client: Optional[ETradeClient] = None):
        self.client = client

    async def refresh_data(self, symbols: List[str], interval: str = '5min', 
                          days: int = 1) -> Dict[str, pd.DataFrame]:
        """Asynchronously fetch latest market data for all symbols."""
        if not self.client:
            raise ValueError("E*Trade client not initialized")
        
        try:
            data = await self.fetch_all_candles(
                self.client, symbols, interval=interval, days=days
            )
            return data
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            raise

class PatternDetector:
    """Handles both rule-based and ML-based pattern detection."""

    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> List[str]:
        """Detect candlestick patterns in the given DataFrame."""
        if len(df) < 3:
            return []

        detections = []
        last = df.iloc[-1]
        prev = df.iloc[-2]
        third = df.iloc[-3]

        pattern_checks = [
            (CandlestickPatterns.is_hammer, [last], "Hammer"),
            (CandlestickPatterns.is_bullish_engulfing, [prev, last], "Bullish Engulfing"),
            (CandlestickPatterns.is_bearish_engulfing, [prev, last], "Bearish Engulfing"),
            (CandlestickPatterns.is_doji, [last], "Doji"),
            (CandlestickPatterns.is_morning_star, [third, prev, last], "Morning Star"),
            (CandlestickPatterns.is_evening_star, [third, prev, last], "Evening Star")
        ]

        for check_fn, args, pattern_name in pattern_checks:
            try:
                if check_fn(*args):
                    detections.append(pattern_name)
            except Exception as e:
                logger.warning(f"Error checking {pattern_name} pattern: {e}")

        return detections

    @staticmethod
    def get_model_prediction(
        model: PatternNN, 
        df: pd.DataFrame, 
        seq_len: int,
        class_names: List[str]
    ) -> Optional[str]:
        """Get prediction from the neural model."""
        try:
            if len(df) < seq_len:
                return None
                
            seq = torch.tensor(
                df.tail(seq_len).values[None], 
                dtype=torch.float32
            )
            with torch.no_grad():
                logits = model(seq)
            pred = int(torch.argmax(logits, dim=1).item())
            return class_names[pred]
        except Exception as e:
            logger.error(f"Error getting model prediction: {e}")
            return None

class DashboardUI:
    """Handles UI rendering and user interactions."""

    @staticmethod
    def render_symbol_chart(df: pd.DataFrame, symbol: str) -> None:
        """Render candlestick chart for a symbol."""
        fig = go.Figure(data=[
            go.Candlestick(
                x=df.index,
                open=df['open'], 
                high=df['high'],
                low=df['low'], 
                close=df['close'],
                name=symbol
            )
        ])
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            title=f"{symbol} Price Chart",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_trading_controls(
        client: ETradeClient,
        symbol: str
    ) -> None:
        """Render buy/sell buttons with order execution."""
        buy_col, sell_col = st.columns(2)
        
        try:
            if buy_col.button(f"Buy {symbol}", key=f"buy_{symbol}"):
                with st.spinner("Placing buy order..."):
                    resp = client.place_market_order(symbol, 1, instruction="BUY")
                    buy_col.success(f"BUY order placed: {resp}")
                    
            if sell_col.button(f"Sell {symbol}", key=f"sell_{symbol}"):
                with st.spinner("Placing sell order..."):
                    resp = client.place_market_order(symbol, 1, instruction="SELL")
                    sell_col.success(f"SELL order placed: {resp}")
        except Exception as e:
            st.error(f"Order execution failed: {e}")
            logger.error(f"Order execution error for {symbol}: {e}")