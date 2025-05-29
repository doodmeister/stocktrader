"""
Real-time Stock Dashboard Module

Provides a comprehensive Streamlit dashboard for real-time stock data visualization,
technical analysis, pattern detection, and AI-powered insights.

Features:
- Real-time stock data fetching and processing
- Interactive candlestick and line charts
- Technical indicators (SMA, EMA)
- Candlestick pattern detection
- AI-powered analysis via ChatGPT
- Risk management metrics
- Multi-symbol watchlist
"""

import json
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
import streamlit as st
import yfinance as yf

# Optional imports
try:
    import ta
except ImportError:
    ta = None
    st.warning("📊 Technical analysis library (ta) not available. Some indicators may be disabled.")

from patterns.patterns import CandlestickPatterns
from utils.chatgpt import get_chatgpt_insight
from utils.data_validator import validate_ticker_symbol, validate_timeframe
from security.authentication import get_openai_api_key
from core.dashboard_utils import (
    safe_streamlit_metric, 
    handle_streamlit_error, 
    cache_key_builder,
    setup_page
)

# Dashboard logger setup
from utils.logger import get_dashboard_logger

# Page setup will be handled in main() function to avoid conflicts with main dashboard

# Constants
CACHE_TTL = 60  # Cache time-to-live in seconds
DEFAULT_SYMBOLS = ['AAPL', 'GOOGL', 'AMZN', 'MSFT']
VALID_TIME_PERIODS = ['1d', '1wk', '1mo', '1y', 'max']
VALID_CHART_TYPES = ['Candlestick', 'Line']
VALID_INDICATORS = ['SMA 20', 'EMA 20']

# Interval mapping for different time periods
INTERVAL_MAPPING = {
    '1d': '1m',
    '1wk': '30m',
    '1mo': '1d',
    '1y': '1wk',
    'max': '1wk'
}

# Initialize logger
logger = get_dashboard_logger(__name__)


class DataProcessor:
    """Handles stock data processing and validation."""
    
    @staticmethod
    def flatten_column(series_or_df: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        """
        Ensures the returned object is always a 1D Series.
        
        Args:
            series_or_df: Input data that could be Series or DataFrame
            
        Returns:
            pd.Series: Flattened 1D series
        """
        try:
            if isinstance(series_or_df, pd.DataFrame):
                return series_or_df.iloc[:, 0]
            elif hasattr(series_or_df, 'values') and series_or_df.values.ndim == 2:
                return pd.Series(series_or_df.values.flatten(), index=series_or_df.index)
            return series_or_df
        except Exception as e:
            logger.error(f"Error flattening column: {e}")
            return pd.Series()

    @staticmethod
    @st.cache_data(ttl=CACHE_TTL, show_spinner=False)
    def fetch_stock_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
        """
        Fetch stock data with caching and error handling.
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            period: Time period for data
            interval: Data interval
            
        Returns:
            pd.DataFrame: Stock data or empty DataFrame on error
        """
        try:
            # Validate inputs
            if not validate_ticker_symbol(ticker):
                logger.warning(f"Invalid ticker symbol: {ticker}")
                return pd.DataFrame()
                
            if not validate_timeframe(period):
                logger.warning(f"Invalid time period: {period}")
                return pd.DataFrame()

            logger.info(f"Fetching data for {ticker}, period: {period}, interval: {interval}")
            
            end_date = datetime.now()
            if period == '1wk':
                start_date = end_date - timedelta(days=7)
                data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            else:
                data = yf.download(ticker, period=period, interval=interval)
                
            if data.empty:
                logger.warning(f"No data returned for {ticker}")
                
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()

    @staticmethod
    def process_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw stock data for consistent format.
        
        Args:
            data: Raw stock data from yfinance
            
        Returns:
            pd.DataFrame: Processed data with standardized columns and timezone
        """
        try:
            if data.empty:
                return data
                
            # Flatten MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ['_'.join([str(c) for c in col]).rstrip('_') 
                               for col in data.columns.values]

            # Handle timezone
            if data.index.tzinfo is None:
                data.index = data.index.tz_localize('UTC')
            data.index = data.index.tz_convert('US/Eastern')
            
            # Reset index and standardize column names
            data.reset_index(inplace=True)
            data.rename(columns={'Date': 'Datetime'}, inplace=True)
            data.columns = [str(c).lower() for c in data.columns]
            
            logger.debug(f"Processed data shape: {data.shape}, columns: {data.columns.tolist()}")
            return data
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return pd.DataFrame()

    @staticmethod
    def find_column(data: pd.DataFrame, base: str, alternatives: Optional[List[str]] = None) -> str:
        """
        Find column with flexible matching.
        
        Args:
            data: DataFrame to search
            base: Base column name to find
            alternatives: Alternative column names to try
            
        Returns:
            str: Found column name
            
        Raises:
            KeyError: If no matching column found
        """
        if alternatives is None:
            alternatives = []
            
        # Try exact match first
        if base in data.columns:
            return base
            
        # Try alternatives
        for alt in alternatives:
            if alt in data.columns:
                return alt
                
        # Try partial match
        for col in data.columns:
            if col.startswith(base):
                return col
                
        raise KeyError(f"No '{base}' column found in data. Available columns: {list(data.columns)}")

    @classmethod
    def find_close_column(cls, data: pd.DataFrame) -> str:
        """Find the close price column."""
        return cls.find_column(data, 'close', ['adj close', 'close_sldp', 'adj close_sldp'])


class TechnicalAnalyzer:
    """Handles technical analysis calculations."""
    
    @staticmethod
    def calculate_metrics(data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate basic financial metrics from stock data.
        
        Args:
            data: Processed stock data
            
        Returns:
            Dict containing calculated metrics
            
        Raises:
            ValueError: If required columns are missing
        """
        try:
            processor = DataProcessor()
            
            close_col = processor.find_close_column(data)
            open_col = processor.find_column(data, 'open')
            high_col = processor.find_column(data, 'high')
            low_col = processor.find_column(data, 'low')
            volume_col = processor.find_column(data, 'volume')
            
            close_series = processor.flatten_column(data[close_col])
            
            if len(close_series) == 0:
                raise ValueError("Empty close series")
                
            last_close = float(close_series.iloc[-1])
            prev_close = float(close_series.iloc[0])
            change = last_close - prev_close
            pct_change = (change / prev_close) * 100 if prev_close != 0 else 0.0
            
            high = float(processor.flatten_column(data[high_col]).max())
            low = float(processor.flatten_column(data[low_col]).min())
            volume = float(processor.flatten_column(data[volume_col]).sum())
            
            return {
                'last_close': last_close,
                'change': change,
                'pct_change': pct_change,
                'high': high,
                'low': low,
                'volume': volume
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            raise ValueError(f"Failed to calculate metrics: {e}")

    @staticmethod
    def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the data.
        
        Args:
            data: Stock data
            
        Returns:
            pd.DataFrame: Data with added technical indicators
        """
        try:
            if data.empty:
                return data
                
            processor = DataProcessor()
            close_col = processor.find_close_column(data)
            close_series = processor.flatten_column(data[close_col])
            
            # Only calculate indicators if we have enough data
            if len(close_series) >= 20:
                data['sma_20'] = ta.trend.sma_indicator(close_series, window=20)
                data['ema_20'] = ta.trend.ema_indicator(close_series, window=20)
            else:
                logger.warning("Insufficient data for 20-period indicators")
                data['sma_20'] = None
                data['ema_20'] = None
                
            return data
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return data


class PatternDetector:
    """Handles candlestick pattern detection."""
    
    @staticmethod
    def normalize_window_columns(window: pd.DataFrame, open_col: str, high_col: str, 
                                low_col: str, close_col: str) -> pd.DataFrame:
        """
        Normalize column names for pattern detection.
        
        Args:
            window: Data window
            open_col, high_col, low_col, close_col: Column names
            
        Returns:
            pd.DataFrame: Window with normalized column names
        """
        columns_to_rename = {
            open_col: 'open',
            high_col: 'high',
            low_col: 'low',
            close_col: 'close',
        }
        return window.rename(columns={k: v for k, v in columns_to_rename.items() 
                                    if k in window.columns})

    @classmethod
    def detect_patterns(cls, data: pd.DataFrame, selected_patterns: List[str]) -> List[Dict[str, Any]]:
        """
        Detect candlestick patterns in the data.
        
        Args:
            data: Processed stock data
            selected_patterns: List of pattern names to detect
            
        Returns:
            List of detected patterns with metadata
        """
        detected_patterns = []
        
        try:
            if data.empty or len(selected_patterns) == 0:
                return detected_patterns
                
            processor = DataProcessor()
            
            # Find required columns
            close_col = processor.find_close_column(data)
            open_col = processor.find_column(data, 'open')
            high_col = processor.find_column(data, 'high')
            low_col = processor.find_column(data, 'low')
            
            # Create CandlestickPatterns instance
            patterns_detector = CandlestickPatterns()
            
            # Pattern detection with sliding window
            for i in range(len(data)):
                window = data.iloc[max(0, i-4):i+1].copy()
                
                if len(window) < 2:  # Need at least 2 candles for patterns
                    continue
                    
                normalized_window = cls.normalize_window_columns(
                    window, open_col, high_col, low_col, close_col
                )
                  # Only detect if all required columns are present
                required_cols = ['open', 'high', 'low', 'close']
                if not all(col in normalized_window.columns for col in required_cols):
                    continue
                
                try:
                    # Fix: Call detect_patterns on instance with df parameter
                    pattern_results = patterns_detector.detect_patterns(df=normalized_window)
                    
                    # Handle the results based on what your CandlestickPatterns returns
                    if isinstance(pattern_results, list):
                        # If it returns a list of PatternResult objects
                        for result in pattern_results:
                            if hasattr(result, 'detected') and hasattr(result, 'name'):
                                if result.detected and result.name in selected_patterns:
                                    detected_patterns.append({
                                        "index": i,
                                        "pattern": result.name,
                                        "datetime": data['datetime'].iloc[i],
                                        "confidence": getattr(result, 'confidence', 1.0)
                                    })
                            elif isinstance(result, str) and result in selected_patterns:
                                # If it returns a list of pattern names
                                detected_patterns.append({
                                    "index": i,
                                    "pattern": result,
                                    "datetime": data['datetime'].iloc[i],
                                    "confidence": 1.0
                                })
                    else:
                        # If it returns something else, handle accordingly
                        logger.warning(f"Unexpected pattern detection result type: {type(pattern_results)}")
                        
                except Exception as pattern_error:
                    logger.warning(f"Pattern detection error at index {i}: {pattern_error}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in pattern detection: {e}")
            
        logger.info(f"Detected {len(detected_patterns)} patterns")
        return detected_patterns


class ChartBuilder:
    """Handles chart creation and visualization."""
    
    @staticmethod
    def create_chart(data: pd.DataFrame, chart_type: str, indicators: List[str], 
                    detected_patterns: List[Dict], ticker: str, time_period: str) -> go.Figure:
        """
        Create interactive stock chart.
        
        Args:
            data: Stock data
            chart_type: Type of chart ('Candlestick' or 'Line')
            indicators: List of technical indicators to include
            detected_patterns: Detected patterns to mark
            ticker: Stock symbol
            time_period: Time period for title
            
        Returns:
            plotly.graph_objects.Figure: Interactive chart
        """
        try:
            processor = DataProcessor()
            
            # Find required columns
            close_col = processor.find_close_column(data)
            open_col = processor.find_column(data, 'open')
            high_col = processor.find_column(data, 'high')
            low_col = processor.find_column(data, 'low')
            
            fig = go.Figure()
            
            # Create main chart
            if chart_type == 'Candlestick':
                fig.add_trace(go.Candlestick(
                    x=data['datetime'],
                    open=processor.flatten_column(data[open_col]),
                    high=processor.flatten_column(data[high_col]),
                    low=processor.flatten_column(data[low_col]),
                    close=processor.flatten_column(data[close_col]),
                    name=ticker
                ))
            else:  # Line chart
                close_series = processor.flatten_column(data[close_col])
                fig.add_trace(go.Scatter(
                    x=data['datetime'],
                    y=close_series,
                    mode='lines',
                    name=f'{ticker} Close',
                    line=dict(width=2)
                ))
            
            # Add technical indicators
            for indicator in indicators:
                try:
                    if indicator == 'SMA 20' and 'sma_20' in data.columns:
                        sma_data = processor.flatten_column(data['sma_20'])
                        fig.add_trace(go.Scatter(
                            x=data['datetime'], 
                            y=sma_data, 
                            name='SMA 20',
                            line=dict(color='orange', width=1)
                        ))
                    elif indicator == 'EMA 20' and 'ema_20' in data.columns:
                        ema_data = processor.flatten_column(data['ema_20'])
                        fig.add_trace(go.Scatter(
                            x=data['datetime'], 
                            y=ema_data, 
                            name='EMA 20',
                            line=dict(color='purple', width=1)
                        ))
                except Exception as ind_error:
                    logger.warning(f"Error adding indicator {indicator}: {ind_error}")
            
            # Add pattern markers
            for pattern_info in detected_patterns:
                try:
                    pattern_index = pattern_info["index"]
                    if pattern_index < len(data):
                        fig.add_trace(go.Scatter(
                            x=[pattern_info["datetime"]],
                            y=[data[high_col].iloc[pattern_index]],
                            mode="markers+text",
                            marker=dict(symbol="triangle-up", size=12, color="green"),
                            text=[pattern_info["pattern"]],
                            textposition="top center",
                            name=pattern_info["pattern"],
                            hovertemplate=f"Pattern: {pattern_info['pattern']}<br>" +
                                        f"Time: {pattern_info['datetime']}<br>" +
                                        "<extra></extra>"
                        ))
                except Exception as pattern_error:
                    logger.warning(f"Error adding pattern marker: {pattern_error}")
            
            # Format chart
            fig.update_layout(
                title=f'{ticker} {time_period.upper()} Chart',
                xaxis_title='Time',
                yaxis_title='Price (USD)',
                height=600,
                xaxis_rangeslider_visible=False,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating chart: {e}")
            # Return empty figure on error
            fig = go.Figure()
            fig.update_layout(title="Chart Error", height=400)
            return fig


class DashboardState:
    """Manages dashboard session state."""
    
    @staticmethod
    def initialize_session_state():
        """Initialize session state variables."""
        if 'detected_patterns' not in st.session_state:
            st.session_state['detected_patterns'] = []
        if 'last_update' not in st.session_state:
            st.session_state['last_update'] = None
        if 'error_count' not in st.session_state:
            st.session_state['error_count'] = 0
        # Clear any stale analysis data on fresh load
        if 'realtime_dashboard_analysis_data' not in st.session_state:
            st.session_state['realtime_dashboard_analysis_data'] = None

    @staticmethod
    def update_patterns(patterns: List[Dict]):
        """Update detected patterns in session state."""
        st.session_state['detected_patterns'] = patterns
        st.session_state['last_update'] = datetime.now()

    @staticmethod
    def get_patterns() -> List[Dict]:
        """Get detected patterns from session state."""
        return st.session_state.get('detected_patterns', [])


class AIAnalyzer:
    """Handles AI-powered analysis."""
    
    @staticmethod
    def generate_summary(ticker: str, time_period: str, detected_patterns: List[Dict]) -> str:
        """
        Generate analysis summary text.
        
        Args:
            ticker: Stock symbol
            time_period: Analysis time period
            detected_patterns: Detected patterns
            
        Returns:
            str: Formatted summary text
        """
        try:
            patterns_detail = json.dumps(detected_patterns, indent=2, default=str) if detected_patterns else "None"
            
            summary_lines = [
                f"Ticker: {ticker}",
                f"Time Period: {time_period}",
                f"Patterns Detected: {', '.join(set(p['pattern'] for p in detected_patterns)) or 'None'}",
                f"Total Pattern Count: {len(detected_patterns)}",
                f"Analysis Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "Detected Patterns Detail:",
                patterns_detail
            ]
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Error generating summary: {e}"

    @staticmethod
    def get_ai_insight(summary_text: str) -> str:
        """
        Get AI insight from ChatGPT.
        
        Args:
            summary_text: Analysis summary
            
        Returns:
            str: AI analysis result
        """
        try:
            api_key = get_openai_api_key()
            if not api_key:
                return "OpenAI API key not configured"
                
            result = get_chatgpt_insight(summary_text)
            return result
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return f"AI analysis failed: {e}"


def render_sidebar_prices():
    """Render real-time prices in sidebar with enhanced rate limiting protection."""
    st.sidebar.header('Real-Time Stock Prices')
    
    # Check if we should fetch new data (limit to every 30 seconds)
    current_time = time.time()
    last_fetch_key = 'last_sidebar_fetch'
    
    if last_fetch_key not in st.session_state:
        st.session_state[last_fetch_key] = 0
    
    time_since_last_fetch = current_time - st.session_state[last_fetch_key]
    
    if time_since_last_fetch < 30:  # Wait at least 30 seconds between fetches
        # Show cached data if available
        if 'sidebar_price_cache' in st.session_state:
            st.sidebar.caption(f"⏰ Next update in {30 - int(time_since_last_fetch)} seconds")
            for cached_item in st.session_state['sidebar_price_cache']:
                with st.sidebar:
                    safe_streamlit_metric(
                        cached_item['symbol'],
                        cached_item['price'],
                        cached_item['change']
                    )
        else:
            st.sidebar.caption("⚠️ Loading prices...")
        return
    
    # Update fetch timestamp
    st.session_state[last_fetch_key] = current_time
    
    processor = DataProcessor()
    
    # Add rate limiting info
    st.sidebar.caption("📊 Fetching latest prices...")
    
    # Initialize cache
    sidebar_cache = []
    
    for i, symbol in enumerate(DEFAULT_SYMBOLS):
        try:            # Add substantial delay between requests to avoid rate limiting
            if i > 0:
                time.sleep(2.0)  # 2 second delay between requests
                
            # Use even longer interval to reduce API calls
            real_time_data = processor.fetch_stock_data(symbol, '1d', '15m')  # Use 15m interval
            if not real_time_data.empty:
                real_time_data = processor.process_data(real_time_data)
                
                close_col = processor.find_close_column(real_time_data)
                open_col = processor.find_column(real_time_data, 'open')
                
                last_price = float(processor.flatten_column(real_time_data[close_col]).iloc[-1])
                open_price = float(processor.flatten_column(real_time_data[open_col]).iloc[0])
                change = last_price - open_price
                pct_change = (change / open_price) * 100 if open_price != 0 else 0.0
                
                # Cache the data
                cached_item = {
                    'symbol': f"{symbol}",
                    'price': f"{last_price:.2f} USD",
                    'change': f"{change:.2f} ({pct_change:.2f}%)"
                }
                sidebar_cache.append(cached_item)
                  # Use safe metric display instead of direct st.sidebar.metric
                with st.sidebar:
                    safe_streamlit_metric(
                        cached_item['symbol'], 
                        cached_item['price'], 
                        cached_item['change']
                    )
            else:
                # Show placeholder for failed symbols
                cached_item = {
                    'symbol': f"{symbol}",
                    'price': "Data unavailable",
                    'change': ""
                }
                sidebar_cache.append(cached_item)
                
                with st.sidebar:
                    st.caption(f"{symbol}: Data unavailable")
                
        except Exception as e:
            # Show user-friendly error instead of crashing
            cached_item = {
                'symbol': f"{symbol}",
                'price': "Rate limited",
                'change': ""
            }
            sidebar_cache.append(cached_item)
            
            with st.sidebar:
                st.caption(f"{symbol}: Rate limited")
            logger.warning(f"Error in sidebar prices for {symbol}: {e}")
    
    # Store cache for next time
    st.session_state['sidebar_price_cache'] = sidebar_cache


def render_main_dashboard():
    """Render the main dashboard content."""
    # Initialize components
    processor = DataProcessor()
    analyzer = TechnicalAnalyzer()
    detector = PatternDetector()
    chart_builder = ChartBuilder()
    state_manager = DashboardState()
    ai_analyzer = AIAnalyzer()
    
    # Initialize session state
    state_manager.initialize_session_state()    # Sidebar form for parameters    # Create unique form key based on session to avoid conflicts
    form_key = f"realtime_dashboard_v3_form_{int(time.time() * 1000) % 100000}"
    
    with st.sidebar.form(form_key):
        st.header('Chart Parameters')
        ticker = st.text_input('Ticker', 'ADBE').upper().strip()
        time_period = st.selectbox('Time Period', VALID_TIME_PERIODS)
        chart_type = st.selectbox('Chart Type', VALID_CHART_TYPES)
        indicators = st.multiselect('Technical Indicators', VALID_INDICATORS)        # Pattern selection - fix: create instance of CandlestickPatterns
        try:
            patterns_instance = CandlestickPatterns()
            pattern_names = patterns_instance.get_pattern_names()
            selected_patterns = st.multiselect(
                "Patterns to scan for", 
                pattern_names, 
                default=pattern_names[:6] if len(pattern_names) >= 6 else pattern_names
            )
        except Exception as e:
            st.error(f"Error loading patterns: {e}")
            selected_patterns = []
            
        submitted = st.form_submit_button("Update")
    
    # Debug information for troubleshooting
    if st.sidebar.checkbox("Show Form Debug", value=False):
        st.sidebar.write(f"Form submitted: {submitted}")
        st.sidebar.write(f"Ticker: {ticker if 'ticker' in locals() else 'Not set'}")
        st.sidebar.write(f"Time period: {time_period if 'time_period' in locals() else 'Not set'}")    
    # Main content area
    analysis_data = None  # Track if we have analysis data
    
    if submitted and ticker:
        try:
            st.success(f"Processing request for {ticker} ({time_period})...")
            with st.spinner(f"Loading data for {ticker}..."):
                # Fetch and process data
                interval = INTERVAL_MAPPING.get(time_period, '1d')
                raw_data = processor.fetch_stock_data(ticker, time_period, interval)
                
                if raw_data.empty:
                    st.error(f"No data available for {ticker} in the {time_period} period.")
                    return
                    
                data = processor.process_data(raw_data)
                
                if data.empty:
                    st.error("Failed to process data.")
                    return
                
                # Add technical indicators
                data = analyzer.add_technical_indicators(data)
                
                # Calculate metrics
                metrics = analyzer.calculate_metrics(data)
                
                # Display main metrics using safe metric display
                safe_streamlit_metric(
                    label=f"{ticker} Last Price", 
                    value=f"{metrics['last_close']:.2f} USD", 
                    delta=f"{metrics['change']:.2f} ({metrics['pct_change']:.2f}%)"
                )
                
                # Display additional metrics in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    safe_streamlit_metric("High", f"{metrics['high']:.2f} USD")
                with col2:
                    safe_streamlit_metric("Low", f"{metrics['low']:.2f} USD")
                with col3:
                    safe_streamlit_metric("Volume", f"{metrics['volume']:,.0f}")
                
                # Pattern detection
                detected_patterns = detector.detect_patterns(data, selected_patterns)
                state_manager.update_patterns(detected_patterns)
                
                # Display detected patterns
                if detected_patterns:
                    st.subheader("Detected Patterns")
                    pattern_df = pd.DataFrame(detected_patterns)
                    st.dataframe(pattern_df, use_container_width=True)
                else:
                    st.info("No selected patterns detected in this data.")
                
                # Create and display chart
                fig = chart_builder.create_chart(
                    data, chart_type, indicators, detected_patterns, ticker, time_period
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Store analysis data for AI section and session state
                analysis_data = {
                    'ticker': ticker,
                    'time_period': time_period,
                    'detected_patterns': detected_patterns
                }
                st.session_state['realtime_dashboard_analysis_data'] = analysis_data
                
        except Exception as e:
            # Use centralized error handling
            handle_streamlit_error(e, f"processing {ticker}")
            logger.error(f"Dashboard error for {ticker}: {e}\n{traceback.format_exc()}")
            st.session_state['error_count'] = st.session_state.get('error_count', 0) + 1
    else:
        # Show instructions when no form submission
        if not submitted:
            st.info("👈 Please fill out the form in the sidebar and click 'Update' to load stock data and analysis.")
        else:
            st.warning("Please enter a valid ticker symbol.")
    
    # AI Analysis Section
    st.subheader("AI-Powered Analysis")    # Add a clear button for troubleshooting
    col1, col2 = st.columns([3, 1])
    with col2:
        clear_btn_key = f"clear_data_btn_{int(time.time() * 1000) % 10000}"
        if st.button("🗑️ Clear Data", key=clear_btn_key, help="Clear cached analysis data"):
            st.session_state['realtime_dashboard_analysis_data'] = None
            st.rerun()
    
    # Get analysis data - either from current execution or session state
    if 'analysis_data' not in locals() or analysis_data is None:
        analysis_data = st.session_state.get('realtime_dashboard_analysis_data', None)
    
    # Generate summary only if we have analysis data
    if analysis_data:
        summary_text = ai_analyzer.generate_summary(
            analysis_data['ticker'], 
            analysis_data['time_period'], 
            analysis_data['detected_patterns']
        )
    else:
        summary_text = "No analysis data available. Please submit the form above to load stock data."
    
    st.text_area("Copyable Analysis Summary", summary_text, height=200)    # AI Insight button
    insight_btn_key = f"chatgpt_insight_btn_{int(time.time() * 1000) % 10000}"
    if st.button("Get ChatGPT Insight", key=insight_btn_key):
        if not analysis_data:
            st.warning("Please submit the form first to generate analysis data.")
        else:
            with st.spinner("Contacting ChatGPT..."):
                insight = ai_analyzer.get_ai_insight(summary_text)
                st.markdown("**AI Analysis Results:**")
                st.write(insight)


def main():
    """Main dashboard function."""
    try:
        # Only setup page if we're not being loaded by the main dashboard
        # The main dashboard handles page configuration
        if '__main__' in str(globals().get('__name__', '')):
            setup_page(
                title="📊 Real-time Stock Dashboard V3",
                logger_name=__name__,
                sidebar_title="Dashboard Controls"
            )
        else:
            # When loaded by main dashboard, just set the title
            st.title('📊 Real-Time Stock Dashboard V3')
          # Render main dashboard
        render_main_dashboard()
        
        # Render sidebar content - temporarily disabled due to rate limiting
        # render_sidebar_prices()
        
        # Sidebar information
        st.sidebar.subheader('About')
        st.sidebar.info(
            'This dashboard provides stock data and technical indicators for various time periods. '
            'Use the sidebar to customize your view and get AI-powered insights.'
        )
          # Display debug info in development
        if st.sidebar.checkbox("Show Debug Info", value=False):
            st.sidebar.subheader("Debug Information")
            st.sidebar.json({
                "Session State Keys": list(st.session_state.keys()),
                "Error Count": st.session_state.get('error_count', 0),
                "Last Update": str(st.session_state.get('last_update', 'Never'))
            })
            
    except Exception as e:
        st.error(f"Critical dashboard error: {e}")
        logger.critical(f"Critical dashboard error: {e}\n{traceback.format_exc()}")

class RealtimeDashboardV3:
    def __init__(self):
        pass
    
    def run(self):
        """Main dashboard application entry point."""
        main()

# Execute the main function
if __name__ == "__main__":
    try:
        dashboard = RealtimeDashboardV3()
        dashboard.run()
    except Exception as e:
        handle_streamlit_error(e, "Realtime Dashboard V3")

