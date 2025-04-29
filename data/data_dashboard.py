"""Stock Data Dashboard Module

A Streamlit dashboard for fetching, displaying, managing, and training on historical OHLCV data.
Follows SOLID principles and includes robust error handling.
"""

import logging
import os
from datetime import date, timedelta, datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

from stocktrader.data.data_loader import save_to_csv
from stocktrader.core.notifier import Notifier
from stocktrader.utils.validation import sanitize_input
from stocktrader.utils.io import create_zip_archive
from stocktrader.train.training_pipeline import feature_engineering
from stocktrader.config import Config

# Configure logging with rotation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.RotatingFileHandler(
            'dashboard.log',
            maxBytes=1024*1024,
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DashboardConfig:
    """Dashboard configuration settings."""
    DEFAULT_SYMBOLS: str = "AAPL,MSFT"
    VALID_INTERVALS: List[str] = ["1d", "1h", "30m", "15m", "5m", "1m"]
    MAX_INTRADAY_DAYS: int = 60
    REFRESH_INTERVAL: int = 300
    DATA_DIR: Path = Path("data")
    MODEL_DIR: Path = Path("models")
    CACHE_TTL: int = 3600

class DataValidator:
    """Handles input validation logic."""
    
    @staticmethod
    def validate_dates(start_date: date, end_date: date, interval: str) -> Tuple[bool, str]:
        """Validate date range and interval compatibility."""
        if not isinstance(start_date, date) or not isinstance(end_date, date):
            return False, "Invalid date format"
            
        if start_date > end_date:
            return False, "Start date must be before end date"

        if interval != "1d":
            delta = end_date - start_date
            if delta.days > DashboardConfig.MAX_INTRADAY_DAYS:
                return False, f"Intraday data limited to {DashboardConfig.MAX_INTRADAY_DAYS} days"

        if end_date > date.today():
            return False, "End date cannot be in the future"

        return True, ""

    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """Validate stock symbol format."""
        if not symbol or not isinstance(symbol, str):
            return False
        return bool(symbol.strip().isalnum())

class DataFetcher:
    """Handles data fetching and caching."""
    
    @staticmethod
    @st.cache_data(ttl=DashboardConfig.CACHE_TTL)
    def fetch_ohlcv(symbol: str, start: str, end: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data with caching and error handling."""
        try:
            data = yf.download(
                symbol,
                start=start,
                end=end,
                interval=interval,
                progress=False,
                timeout=10
            )
            if data.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
                
            # Validate data quality
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                logger.error(f"Missing required columns for {symbol}")
                return None
                
            return data[required_columns]
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

class ModelManager:
    """Handles model training and persistence."""
    
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model_dir.mkdir(exist_ok=True)

    def train_model(self, df: pd.DataFrame, symbol: str, interval: str) -> Tuple[bool, str]:
        """Train and save model with error handling."""
        try:
            # Feature engineering
            df_processed = feature_engineering(df)
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                       'returns', 'volatility', 'sma_5', 'sma_10']
            
            X = df_processed[features]
            y = df_processed['target']

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            # Save model
            model_path = self.model_dir / f"{symbol}_{interval}_model.pkl"
            joblib.dump(model, model_path)

            return True, str(model_path)

        except Exception as e:
            logger.error(f"Model training failed for {symbol}: {str(e)}")
            return False, str(e)

class DataDashboard:
    """Manages the stock data dashboard functionality."""

    def __init__(self):
        """Initialize dashboard components."""
        self.config = DashboardConfig()
        self.validator = DataValidator()
        self.fetcher = DataFetcher()
        self.model_manager = ModelManager(self.config.MODEL_DIR)
        
        self.data_dir = self.config.DATA_DIR
        self.data_dir.mkdir(exist_ok=True)
        self.saved_paths = []
        self.notifier = Notifier()

    def render_dashboard(self):
        """Render the Streamlit dashboard interface with error handling."""
        try:
            self._setup_page()
            self._render_inputs()
            self._handle_user_actions()
        except Exception as e:
            logger.error(f"Dashboard render error: {str(e)}")
            st.error("An unexpected error occurred. Please try again.")

    # Additional methods...

def main():
    """Main entry point for the dashboard."""
    try:
        dashboard = DataDashboard()
        dashboard.render_dashboard()
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        st.error("An unexpected error occurred. Please try again later.")

if __name__ == "__main__":
    main()
