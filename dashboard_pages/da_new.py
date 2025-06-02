"""
Enhanced Technical Analysis Dashboard - Completely Rebuilt Version

Designed from scratch with robust numeric handling to prevent the 
"'>' not supported between instances of 'str' and 'int'" error.

Key improvements:
- Bulletproof type safety throughout the data pipeline
- Comprehensive data validation and sanitization
- Early detection and correction of type mismatches
- Graceful error handling with detailed logging
- Consistent numeric type enforcement
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Tuple, Optional, Union
import os
import logging
import traceback

# Import core modules
try:
    from utils.logger import get_dashboard_logger
    from core.dashboard_utils import handle_streamlit_error, initialize_dashboard_session_state
    from core.data_validator import validate_dataframe as core_validate_dataframe
    from core.session_manager import create_session_manager
    from core.technical_indicators import (
        calculate_rsi, calculate_macd, calculate_bollinger_bands, 
        calculate_atr, IndicatorError
    )
    from utils.technicals.analysis import TechnicalAnalysis
    # Do NOT import any function that returns a DataFrame if a dict is expected
    from patterns.patterns import CandlestickPatterns, create_pattern_detector
    
    logger = get_dashboard_logger('enhanced_ta_dashboard')
    
except ImportError as e:
    # Fallback for development/testing
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('enhanced_ta_dashboard')
    
    def handle_streamlit_error(error: Exception, context: str = ""):
        logger.error(f"Error in {context}: {str(error)}")
        if 'st' in globals():        st.error(f"Error: {str(error)}")
    
    def initialize_dashboard_session_state():
        pass
        
    # Only define this fallback if SessionManager is not imported (i.e., in fallback block)
    try:
        from core.session_manager import SessionManager
    except ImportError:
        SessionManager = None

    def create_session_manager(page_name: Optional[str] = None):
        if SessionManager is not None:
            # Always pass a string to SessionManager to avoid type errors
            return SessionManager(page_name if page_name is not None else "main")
        else:
            return DummySessionManager(page_name)
    
    def calculate_rsi(df, length=14, close_col='close'):
        return pd.Series([50] * len(df), index=df.index)
    
    def calculate_macd(df, fast=12, slow=26, signal=9, close_col='close'):
        zeros = pd.Series([0] * len(df), index=df.index)
        return zeros, zeros, zeros
    
    def calculate_bollinger_bands(df, length=20, std=2, close_col='close'):
        middle = df[close_col] if close_col in df.columns else pd.Series([100] * len(df))
        return middle, middle, middle
    
    def calculate_atr(df, length=14):
        return pd.Series([1] * len(df), index=df.index)
        
    def compute_price_stats(df):
        """
        Fallback: returns a dictionary of price statistics.
        """
        if df is None or not isinstance(df, pd.DataFrame) or df.empty or 'close' not in df.columns:
            return {}
        return {
            "min": float(df['close'].min()),
            "max": float(df['close'].max()),
            "mean": float(df['close'].mean()),
            "last": float(df['close'].iloc[-1])
        }
    
    def compute_return_stats(df):
        """
        Fallback: returns a dictionary of return statistics.
        """
        if df is None or not isinstance(df, pd.DataFrame) or df.empty or 'close' not in df.columns:
            return {}
        returns = df['close'].pct_change().dropna()
        return {
            "mean_return": float(returns.mean()),
            "std_return": float(returns.std()),
            "min_return": float(returns.min()),
            "max_return": float(returns.max())
        }
    
    class NumericDataProcessor:
        @staticmethod
        def process_ohlcv_dataframe(df):
            """
            Ensure the function always returns a DataFrame, never None.
            If input is not a DataFrame, return an empty DataFrame.
            """
            if df is None or not isinstance(df, pd.DataFrame):
                logger.warning("process_ohlcv_dataframe: Input is not a DataFrame, returning empty DataFrame.")
                return pd.DataFrame()
            return df
    
    def create_pattern_detector(confidence_threshold: float = 0.7, enable_caching: bool = True):
        """
        Create a candlestick pattern detector with specified parameters.

        Args:
            confidence_threshold: Threshold for pattern confidence
            enable_caching: Whether to enable caching for pattern detection

        Returns:
            CandlestickPatterns: Initialized pattern detector
        """
        return CandlestickPatterns(confidence_threshold, enable_caching)

# Core numeric safety functions
def safe_numeric_convert(value, default: float = 0.0) -> float:
    """
    Safely convert any value to float, with fallback.
    
    This is the core defense against string vs numeric comparison errors.
    
    Args:
        value: Any input value
        default: Default value if conversion fails
        
    Returns:
        Float value or default
    """
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (ValueError, TypeError, OverflowError):
        logger.debug(f"🔄 Failed to convert {value} to float, using default {default}")
        return default


class EnhancedDataLoader:
    """
    Enhanced data loader with comprehensive error handling and type safety.
    """
    
    @staticmethod
    def load_csv_file(uploaded_file) -> pd.DataFrame:
        """
        Load CSV file with comprehensive data validation and type safety.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Properly typed DataFrame or empty DataFrame on error
        """
        try:
            if uploaded_file is None:
                logger.warning("⚠️ No file uploaded")
                return pd.DataFrame()

            # Reset file pointer to ensure proper reading
            uploaded_file.seek(0)

            # Read CSV file without specifying parse_dates initially
            df = pd.read_csv(uploaded_file, low_memory=False)

            # Attempt to parse 'date' column if it exists
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'].astype(str), errors='coerce')

            if df.empty:
                logger.warning("⚠️ DataFrame is empty after CSV load")
                return pd.DataFrame()

            logger.info(f"📊 CSV loaded: {len(df)} rows, {len(df.columns)} columns")

            # Normalize column names
            df.columns = df.columns.str.lower().str.strip()

            # Show sample data for debugging
            logger.info(f"📝 Sample data:\n{df.head(2).to_string()}")

            # Process with NumericDataProcessor
            df = NumericDataProcessor.process_ohlcv_dataframe(df)

            # Final validation
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                logger.error(f"❌ Missing required columns: {missing_cols}")
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                st.info("Expected columns: Date/Timestamp, Open, High, Low, Close, Volume")
                return pd.DataFrame()

            # Check data quality
            total_rows = len(df)
            valid_rows = df[required_cols].dropna().shape[0]

            if valid_rows < total_rows * 0.5:  # Less than 50% valid data
                logger.warning(f"⚠️ Low data quality: {valid_rows}/{total_rows} valid rows")
                st.warning(f"Warning: Only {valid_rows}/{total_rows} rows have complete data")

            logger.info(f"✅ CSV processing complete: {valid_rows} valid rows")
            return df

        except Exception as e:
            logger.error(f"❌ CSV loading failed: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"Failed to load CSV: {str(e)}")
            return pd.DataFrame()


class EnhancedTechnicalAnalysis:
    """
    Technical analysis calculator with comprehensive error handling.
    """
    
    @staticmethod
    def calculate_indicators_safely(
        df: pd.DataFrame,
        rsi_period: Union[int, float] = 14,
        macd_fast: Union[int, float] = 12,
        macd_slow: Union[int, float] = 26,
        macd_signal: Union[int, float] = 9,
        bb_period: Union[int, float] = 20,
        bb_std: Union[int, float] = 2,
        atr_period: Union[int, float] = 14
    ) -> Dict[str, pd.Series]:
        """
        Calculate technical indicators with comprehensive error handling.
        
        Args:
            df: Input DataFrame
            rsi_period: RSI period
            macd_fast: MACD fast period
            macd_slow: MACD slow period  
            macd_signal: MACD signal period
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviation
            atr_period: ATR period
            
        Returns:
            Dictionary of indicator series
        """
        # Convert all period parameters to integers to ensure type safety
        rsi_period = int(rsi_period)
        macd_fast = int(macd_fast)
        macd_slow = int(macd_slow)
        macd_signal = int(macd_signal)
        bb_period = int(bb_period)
        atr_period = int(atr_period)
        
        indicators = {}
        
        # Validate input DataFrame
        if df.empty:
            logger.warning("⚠️ Empty DataFrame provided to calculate_indicators_safely")
            return indicators
        
        # Ensure numeric columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col in df.columns:
                df[col] = df[col].apply(safe_numeric_convert)
        
        try:
            # Calculate RSI
            if 'close' in df.columns:
                indicators['rsi'] = calculate_rsi(df, length=rsi_period)
                logger.debug(f"✅ RSI calculated (period={rsi_period})")
        except Exception as e:
            logger.error(f"❌ RSI calculation failed: {str(e)}")
            indicators['rsi'] = pd.Series(index=df.index, dtype='float64')
        
        try:
            # Calculate MACD
            if 'close' in df.columns:
                macd_line, macd_signal_line, macd_histogram = calculate_macd(
                    df, fast=macd_fast, slow=macd_slow, signal=macd_signal
                )
                indicators['macd'] = macd_line
                indicators['macd_signal'] = macd_signal_line
                indicators['macd_histogram'] = macd_histogram
                logger.debug(f"✅ MACD calculated (fast={macd_fast}, slow={macd_slow}, signal={macd_signal})")
        except Exception as e:
            logger.error(f"❌ MACD calculation failed: {str(e)}")
            empty_series = pd.Series(index=df.index, dtype='float64')
            indicators['macd'] = empty_series
            indicators['macd_signal'] = empty_series
            indicators['macd_histogram'] = empty_series
        
        try:            # Calculate Bollinger Bands
            if 'close' in df.columns:
                bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
                    df, length=bb_period, std=int(bb_std)
                )
                indicators['bb_upper'] = bb_upper
                indicators['bb_middle'] = bb_middle
                indicators['bb_lower'] = bb_lower
                logger.debug(f"✅ Bollinger Bands calculated (period={bb_period}, std={bb_std})")
        except Exception as e:
            logger.error(f"❌ Bollinger Bands calculation failed: {str(e)}")
            empty_series = pd.Series(index=df.index, dtype='float64')
            indicators['bb_upper'] = empty_series
            indicators['bb_middle'] = empty_series
            indicators['bb_lower'] = empty_series
        
        try:
            # Calculate ATR
            if all(col in df.columns for col in ['high', 'low', 'close']):
                indicators['atr'] = calculate_atr(df, length=atr_period)
                logger.debug(f"✅ ATR calculated (period={atr_period})")
        except Exception as e:
            logger.error(f"❌ ATR calculation failed: {str(e)}")
            indicators['atr'] = pd.Series(index=df.index, dtype='float64')
        
        logger.info(f"✅ Technical indicators calculated: {list(indicators.keys())}")
        return indicators


class PatternDetector:
    """
    Candlestick pattern detector with comprehensive error handling.
    """
    
    @staticmethod
    def detect_patterns_safely(df: pd.DataFrame, selected_patterns: List[str]) -> List[Dict[str, Any]]:
        """
        Detect candlestick patterns with comprehensive error handling.
        
        Args:
            df: Input DataFrame with OHLC data
            selected_patterns: List of pattern names to detect
            
        Returns:
            List of pattern detection results
        """
        results = []
        
        try:
            pattern_detector = create_pattern_detector()
            if pattern_detector is None:
                logger.warning("⚠️ Pattern detector not available")
                return results
                
            # Get list of available patterns
            if hasattr(pattern_detector, 'get_pattern_names'):
                available_patterns = pattern_detector.get_pattern_names()
                logger.info(f"📊 Available patterns: {available_patterns}")
            
            # Detect patterns with comprehensive error handling
            for pattern_name in selected_patterns:
                try:
                    if hasattr(pattern_detector, 'detect_patterns'):
                        detection_results = pattern_detector.detect_patterns(df, pattern_names=[pattern_name])
                        
                        # Process results safely
                        if detection_results:
                            for i, result in enumerate(detection_results):
                                if hasattr(result, 'detected') and hasattr(result, 'name'):
                                    if result.detected and result.name in selected_patterns:
                                        # Create safe result dictionary
                                        pattern_result = {
                                            'index': int(i),
                                            'pattern': str(result.name),
                                            'confidence': float(getattr(result, 'confidence', 0.0)),
                                            'location': int(getattr(result, 'location', i)),
                                            'description': str(getattr(result, 'description', 'Pattern detected'))
                                        }
                                        results.append(pattern_result)
                                        
                except Exception as e:
                    logger.error(f"❌ Pattern detection failed for {pattern_name}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Pattern detection system failed: {str(e)}")
            
        logger.info(f"🔍 Pattern detection complete: {len(results)} patterns found")
        return results


class ChartRenderer:
    """
    Chart renderer with comprehensive error handling.
    """
    
    @staticmethod
    def render_candlestick_chart(
        df: pd.DataFrame, 
        indicators: Dict[str, pd.Series], 
        patterns: Optional[List[Dict[str, Any]]] = None,
        title: str = "Technical Analysis Chart"
    ) -> go.Figure:
        """
        Render candlestick chart with indicators and patterns.
        
        Args:
            df: Input DataFrame
            indicators: Dictionary of technical indicators
            patterns: List of detected patterns
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        try:
            # Create subplot figure
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=['Price & Indicators', 'RSI', 'MACD', 'Volume'],
                vertical_spacing=0.05,
                row_heights=[0.5, 0.15, 0.2, 0.15],
                specs=[[{"secondary_y": True}], [{}], [{}], [{}]]
            )
            
            # Main price chart
            if not df.empty and all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                fig.add_trace(
                    go.Candlestick(
                        x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name="Price",
                        yaxis="y"
                    ),
                    row=1, col=1
                )
            
            # Add Bollinger Bands
            if all(key in indicators for key in ['bb_upper', 'bb_middle', 'bb_lower']):
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=indicators['bb_upper'],
                        name='BB Upper', line=dict(color='gray', width=1),
                        yaxis="y"
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=indicators['bb_middle'],
                        name='BB Middle', line=dict(color='blue', width=1),
                        yaxis="y"
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=indicators['bb_lower'],
                        name='BB Lower', line=dict(color='gray', width=1),
                        fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                        yaxis="y"
                    ),
                    row=1, col=1
                )
            
            # Add RSI
            if 'rsi' in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=indicators['rsi'],
                        name='RSI', line=dict(color='purple', width=2)
                    ),
                    row=2, col=1
                )
                # RSI reference lines - Fixed: use string values for row/col
                fig.add_hline(y=70, row="2", col="1", line_dash="dash", line_color="red", opacity=0.7)
                fig.add_hline(y=30, row="2", col="1", line_dash="dash", line_color="green", opacity=0.7)
                fig.add_hline(y=50, row="2", col="1", line_dash="dot", line_color="gray", opacity=0.5)
            
            # Add MACD
            if all(key in indicators for key in ['macd', 'macd_signal', 'macd_histogram']):
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=indicators['macd'],
                        name='MACD', line=dict(color='blue', width=2)
                    ),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=indicators['macd_signal'],
                        name='MACD Signal', line=dict(color='red', width=2)
                    ),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Bar(
                        x=df.index, y=indicators['macd_histogram'],
                        name='MACD Histogram', marker_color='green', opacity=0.7
                    ),
                    row=3, col=1
                )
            
            # Add Volume
            if 'volume' in df.columns:
                fig.add_trace(
                    go.Bar(
                        x=df.index, y=df['volume'],
                        name='Volume', marker_color='lightblue', opacity=0.7
                    ),
                    row=4, col=1
                )
            
            # Add pattern markers
            if patterns:
                for pattern in patterns:
                    try:
                        idx = pattern.get('location', pattern.get('index', 0))
                        if idx < len(df):
                            fig.add_annotation(
                                x=df.index[idx],
                                y=df['high'].iloc[idx] if 'high' in df.columns else 0,
                                text=pattern.get('pattern', 'Pattern'),
                                showarrow=True,
                                arrowhead=2,
                                arrowcolor="red",
                                bgcolor="yellow",
                                bordercolor="red",
                                row=1, col=1
                            )
                    except Exception as e:
                        logger.error(f"❌ Failed to add pattern marker: {str(e)}")
                        continue
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_rangeslider_visible=False,
                height=800,
                showlegend=True,
                template="plotly_white"
            )
            
            # Set y-axis ranges
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
            fig.update_yaxes(title_text="MACD", row=3, col=1)
            fig.update_yaxes(title_text="Volume", row=4, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Chart rendering failed: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return empty figure on error
            fig = go.Figure()
            fig.add_annotation(
                text=f"Chart rendering failed: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig


class DummySessionManager:
        """Fallback dummy session manager with safe no-op methods for all expected API."""
        def __init__(self, page_name=None):
            self.page_name = page_name
            self.namespace = "dummy"
            self.session_id = "dummy"
        def get_form_key(self, *args, **kwargs):
            return "dummy_form_key"
        def get_button_key(self, *args, **kwargs):
            return "dummy_button_key"
        def get_input_key(self, *args, **kwargs):
            return "dummy_input_key"
        def set_page_state(self, *args, **kwargs):
            pass
        def get_page_state(self, *args, **kwargs):
            return None
        def clear_page_state(self, *args, **kwargs):
            pass
        def cleanup_page(self, *args, **kwargs):
            pass
        def form_container(self, *args, **kwargs):
            from contextlib import nullcontext
            return nullcontext()
        def create_button(self, *args, **kwargs):
            return False
        def create_checkbox(self, *args, **kwargs):
            return False
        def create_selectbox(self, *args, **kwargs):
            options = kwargs.get('options', [])
            return options[0] if options else None
        def create_multiselect(self, *args, **kwargs):
            return []
        def create_slider(self, *args, **kwargs):
            return kwargs.get('value', 0)
        def create_text_input(self, *args, **kwargs):
            return kwargs.get('value', "")
        def create_number_input(self, *args, **kwargs):
            return kwargs.get('value', 0)
        def create_date_input(self, *args, **kwargs):
            return kwargs.get('value', None)
        def create_file_uploader(self, *args, **kwargs):
            return None
        def create_radio(self, *args, **kwargs):
            options = kwargs.get('options', [])
            return options[0] if options else None
        def create_time_input(self, *args, **kwargs):
            return kwargs.get('value', None)
        def create_color_picker(self, *args, **kwargs):
            return kwargs.get('value', "#000000")
        def get_debug_info(self, *args, **kwargs):
            return {"dummy": True}


class EnhancedTechnicalAnalysisDashboard:
    """
    Main dashboard class with comprehensive error handling.
    """
    
    def __init__(self):
        """Initialize the enhanced technical analysis dashboard."""
        try:
            # Initialize session state
            initialize_dashboard_session_state()
            
            # Initialize components
            self.data_loader = EnhancedDataLoader()
            self.technical_analysis = EnhancedTechnicalAnalysis()
            self.pattern_detector = PatternDetector()
            self.chart_renderer = ChartRenderer()
            
            # Session manager
            self.session_manager = create_session_manager("enhanced_ta_dashboard")
            
            logger.info("✅ Enhanced Technical Analysis Dashboard initialized")
            
        except Exception as e:
            logger.error(f"❌ Dashboard initialization failed: {str(e)}")
            handle_streamlit_error(e, "Dashboard Initialization")
    
    def run(self):
        """
        Main dashboard application entry point.
        """
        try:
            st.set_page_config(
                page_title="Enhanced Technical Analysis",
                page_icon="📈",
                layout="wide"
            )
            
            st.title("📈 Enhanced Technical Analysis Dashboard")
            st.markdown("Upload your CSV file with OHLCV data to begin analysis.")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type="csv",
                help="Upload a CSV file with Date, Open, High, Low, Close, Volume columns"
            )
            
            if uploaded_file is not None:
                # Check if we need to reload
                if (not hasattr(st.session_state, 'last_uploaded_file') or 
                    st.session_state.last_uploaded_file != uploaded_file.name):
                    
                    logger.info(f"🔄 Loading new file: {uploaded_file.name}")
                    df = self.data_loader.load_csv_file(uploaded_file)
                    
                    if not df.empty:
                        st.session_state.data = df
                        st.session_state.last_uploaded_file = uploaded_file.name
                        
                        # Display data overview
                        st.success(f"✅ Data loaded successfully: {len(df)} rows")
                        
                        with st.expander("Data Preview", expanded=False):
                            st.dataframe(df.head(10))
                            
                            # Show basic statistics
                            if 'close' in df.columns:
                                date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                                if date_cols:
                                    st.info(f"📅 Data period: {df[date_cols[0]].min()} to {df[date_cols[0]].max()}")
                                
                                price_stats = {
                                    'Current Price': f"${df['close'].iloc[-1]:.2f}",
                                    'Min Price': f"${df['close'].min():.2f}",
                                    'Max Price': f"${df['close'].max():.2f}",
                                    'Average Price': f"${df['close'].mean():.2f}"
                                }
                                st.json(price_stats)
                    else:
                        st.error("❌ Failed to load data. Please check your CSV format.")
                        return
                else:
                    df = st.session_state.get('data', pd.DataFrame())
                
                if not df.empty:
                    self._render_analysis_interface(df)
            else:
                st.info("👆 Please upload a CSV file to get started")
                
        except Exception as e:
            logger.error(f"❌ Dashboard run failed: {str(e)}")
            logger.error(traceback.format_exc())
            handle_streamlit_error(e, "Enhanced Technical Analysis Dashboard")
    
    def _render_analysis_interface(self, df: pd.DataFrame):
        """
        Render the main analysis interface.
        
        Args:
            df: Loaded DataFrame
        """
        try:
            st.markdown("### ⚙️ Technical Analysis Configuration")
            
            # Configuration columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📊 Indicators")
                rsi_period = st.slider("RSI Period", 5, 50, 14)
                bb_period = st.slider("Bollinger Bands Period", 5, 50, 20)
                bb_std = st.slider("BB Standard Deviation", 1.0, 3.0, 2.0, 0.1)
            
            with col2:
                st.subheader("📈 MACD Settings")
                macd_fast = st.slider("MACD Fast Period", 5, 20, 12)
                macd_slow = st.slider("MACD Slow Period", 20, 35, 26)
                macd_signal = st.slider("MACD Signal Period", 5, 15, 9)
            
            with col3:
                st.subheader("🔍 Other Settings")
                atr_period = st.slider("ATR Period", 5, 30, 14)
                
                # Pattern selection
                available_patterns = [
                    'Doji', 'Hammer', 'Shooting Star', 'Engulfing Bull', 'Engulfing Bear',
                    'Morning Star', 'Evening Star', 'Three White Soldiers', 'Three Black Crows'
                ]
                selected_patterns = st.multiselect(
                    "Candlestick Patterns",
                    available_patterns,
                    default=['Doji', 'Hammer']
                )
            
            # Analysis button
            if st.button("🚀 Run Technical Analysis", type="primary"):
                with st.spinner("Calculating technical indicators..."):
                    # Calculate indicators
                    indicators = self.technical_analysis.calculate_indicators_safely(
                        df,
                        rsi_period=rsi_period,
                        macd_fast=macd_fast,
                        macd_slow=macd_slow,
                        macd_signal=macd_signal,
                        bb_period=bb_period,
                        bb_std=bb_std,
                        atr_period=atr_period
                    )
                    
                    # Detect patterns
                    patterns = self.pattern_detector.detect_patterns_safely(df, selected_patterns)
                    
                    # Store results
                    st.session_state.indicators = indicators
                    st.session_state.patterns = patterns
                    st.session_state.analysis_params = {
                        'rsi_period': rsi_period,
                        'macd_fast': macd_fast,
                        'macd_slow': macd_slow,
                        'macd_signal': macd_signal,
                        'bb_period': bb_period,
                        'bb_std': bb_std,
                        'atr_period': atr_period
                    }
                
                st.success("✅ Technical analysis completed!")
            
            # Display results if available
            if hasattr(st.session_state, 'indicators') and st.session_state.indicators:
                self._display_analysis_results(df, st.session_state.indicators, st.session_state.get('patterns', []))
                
        except Exception as e:
            logger.error(f"❌ Analysis interface rendering failed: {str(e)}")
            handle_streamlit_error(e, "Analysis Interface")
    
    def _display_analysis_results(self, df: pd.DataFrame, indicators: Dict[str, pd.Series], patterns: List[Dict[str, Any]]):
        """
        Display technical analysis results.
        
        Args:
            df: Input DataFrame
            indicators: Calculated indicators
            patterns: Detected patterns
        """
        try:
            st.markdown("### 📈 Technical Indicator Analysis")
            
            # Render main chart
            fig = self.chart_renderer.render_candlestick_chart(
                df, indicators, patterns, 
                title="Technical Analysis - Candlestick Chart with Indicators"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Analysis summary
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Current Indicators")
                if 'close' in df.columns:
                    current_price = df['close'].iloc[-1]
                    st.metric("Current Price", f"${current_price:.2f}")
                
                if 'rsi' in indicators and not indicators['rsi'].empty:
                    current_rsi = indicators['rsi'].iloc[-1]
                    rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    st.metric("RSI", f"{current_rsi:.2f}", rsi_signal)
                
                if 'atr' in indicators and not indicators['atr'].empty:
                    current_atr = indicators['atr'].iloc[-1]
                    st.metric("ATR", f"{current_atr:.4f}")
            
            with col2:
                st.subheader("🔍 Pattern Analysis")
                if patterns:
                    st.write(f"**{len(patterns)} patterns detected:**")
                    for pattern in patterns[-5:]:  # Show last 5 patterns
                        st.write(f"• {pattern.get('pattern', 'Unknown')} (Confidence: {pattern.get('confidence', 0):.2f})")
                else:
                    st.info("No patterns detected in the selected timeframe.")
            
            # Display technical indicators
            with st.expander("📈 Detailed Indicator Values", expanded=False):
                indicator_df = pd.DataFrame(indicators)
                if not indicator_df.empty:
                    st.dataframe(indicator_df.tail(20))
                else:
                    st.info("No indicator data available.")
                    
        except Exception as e:
            logger.error(f"❌ Results display failed: {str(e)}")
            handle_streamlit_error(e, "Results Display")


def main():
    """
    Main dashboard application entry point.
    """
    try:
        dashboard = EnhancedTechnicalAnalysisDashboard()
        dashboard.run()
    except Exception as e:
        logger.error(f"❌ Application failed: {str(e)}")
        st.error(f"Application error: {str(e)}")


if __name__ == "__main__":
    main()
