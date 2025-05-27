"""
Shared utilities for dashboard functionality.
"""

import streamlit as st
from typing import Dict, Any, List, Tuple, Optional
import logging
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import tempfile
import shutil

logger = logging.getLogger(__name__)

# Chart configuration constants
CHART_COLORS = {
    "bullish": "#2E8B57",
    "bearish": "#DC143C", 
    "neutral": "#4682B4",
    "highlight": "#FFD700",
    "background": "#F8F9FA"
}

DEFAULT_CHART_CONFIG = {
    "displayModeBar": False,
    "staticPlot": False,
    "responsive": True
}

def setup_page(
    title: str,
    logger_name: str = __name__,
    initialize_session: bool = True,
    sidebar_title: Optional[str] = None
):
    """
    Standard page setup for all dashboard pages.
    This should be called at the start of every page.
    
    Args:
        title: Page title to display
        logger_name: Logger name (usually __name__)
        initialize_session: Whether to initialize session state
        sidebar_title: Optional sidebar title
    
    Returns:
        Logger instance
    """
    from utils.logger import setup_logger
    
    # Set page title (don't call st.set_page_config here)
    st.title(title)
    
    # Setup logger
    logger = setup_logger(logger_name)
    
    # Initialize session state if requested
    if initialize_session:
        initialize_dashboard_session_state()
    
    # Setup sidebar if title provided
    if sidebar_title:
        st.sidebar.header(sidebar_title)
    
    return logger

def safe_streamlit_metric(label: str, value: str, delta: str = None) -> None:
    """
    Safely display a Streamlit metric with error handling.
    
    Args:
        label: Metric label
        value: Metric value
        delta: Optional delta value
    """
    try:
        if delta:
            st.metric(label, value, delta)
        else:
            st.metric(label, value)
    except Exception as e:
        logger.error(f"Error displaying metric {label}: {e}")
        st.text(f"{label}: {value}")

def handle_streamlit_error(error: Exception, context: str = "") -> None:
    """
    Handle and display Streamlit errors gracefully.
    
    Args:
        error: Exception that occurred
        context: Additional context about where error occurred
    """
    error_msg = f"Error in {context}: {str(error)}" if context else str(error)
    logger.error(error_msg)
    st.error(f"An error occurred: {error_msg}")

def cache_key_builder(*args, **kwargs) -> str:
    """Build cache key from arguments."""
    import hashlib
    key_parts = [str(arg) for arg in args] + [f"{k}={v}" for k, v in sorted(kwargs.items())]
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()

def initialize_dashboard_session_state():
    """Initialize common Streamlit session state variables if they don't exist."""
    defaults = {
        'dashboard_initialized': True,
        'error_count': 0,
        'last_update': None,
        'user_preferences': {},
        'cache': {},
        'notifications': [],
        'current_page': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Chart utilities
def create_candlestick_chart(
    df: pd.DataFrame, 
    title: str = "Price Chart",
    detections: List[Tuple[int, float]] = None,
    pattern_name: str = None,
    height: int = 500,
    debug: bool = False
) -> go.Figure:
    """Create standardized candlestick chart used across all dashboards."""
    
    if debug:
        st.write(f"🔍 **Chart Debug:** Input DataFrame shape: {df.shape}")
        st.write(f"🔍 **Chart Debug:** Columns: {list(df.columns)}")
        st.write(f"🔍 **Chart Debug:** Index type: {type(df.index)}")
        st.write(f"🔍 **Chart Debug:** First few rows:")
        st.dataframe(df.head(3))
    
    try:
        # Create a copy to avoid modifying the original
        chart_df = df.copy()
        
        # Ensure we have the required OHLC columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in chart_df.columns]
        
        if missing_cols:
            if debug:
                st.write(f"❌ **Chart Debug:** Missing columns: {missing_cols}")
            
            # Try to create a simple line chart with available price data
            price_col = None
            for col in ['close', 'price', 'value', 'last']:
                if col in chart_df.columns:
                    price_col = col
                    break
            
            if price_col:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=chart_df.index,
                    y=chart_df[price_col],
                    mode='lines',
                    name=f'{price_col.title()} Price',
                    line=dict(color='blue', width=2)
                ))
                
                fig.update_layout(
                    title=f"{title} (Line Chart - Limited Data)",
                    height=height,
                    xaxis_title='Time',
                    yaxis_title='Price',
                    **DEFAULT_CHART_CONFIG
                )
                
                if debug:
                    st.write(f"✅ **Chart Debug:** Created line chart using {price_col}")
                
                return fig
            else:
                # No price data available
                fig = go.Figure()
                fig.add_annotation(
                    text="No price data available for chart",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16)
                )
                fig.update_layout(
                    title=f"{title} (No Data)",
                    height=height,
                    **DEFAULT_CHART_CONFIG
                )
                return fig
        
        # Prepare the data for candlestick chart
        # Ensure index is datetime-like
        if not isinstance(chart_df.index, pd.DatetimeIndex):
            # Try to find a timestamp column
            timestamp_cols = [col for col in chart_df.columns 
                            if any(keyword in col.lower() for keyword in ['time', 'date', 'timestamp'])]
            
            if timestamp_cols:
                timestamp_col = timestamp_cols[0]
                if debug:
                    st.write(f"🔍 **Chart Debug:** Using {timestamp_col} as x-axis")
                
                # Convert to datetime and set as index
                chart_df[timestamp_col] = pd.to_datetime(chart_df[timestamp_col])
                chart_df = chart_df.set_index(timestamp_col)
            else:
                # Use integer index as fallback
                if debug:
                    st.write(f"⚠️ **Chart Debug:** No timestamp column found, using integer index")
        
        # Ensure OHLC columns are numeric
        for col in required_cols:
            if col in chart_df.columns:
                chart_df[col] = pd.to_numeric(chart_df[col], errors='coerce')
        
        # Remove any rows with NaN values in OHLC columns
        chart_df = chart_df.dropna(subset=required_cols)
        
        if chart_df.empty:
            if debug:
                st.write(f"❌ **Chart Debug:** DataFrame is empty after cleaning")
            
            fig = go.Figure()
            fig.add_annotation(
                text="No valid data for chart after cleaning",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title=f"{title} (No Valid Data)",
                height=height,
                **DEFAULT_CHART_CONFIG
            )
            return fig
        
        if debug:
            st.write(f"✅ **Chart Debug:** Final chart data shape: {chart_df.shape}")
            st.write(f"🔍 **Chart Debug:** OHLC sample values:")
            st.write(f"  Open: {chart_df['open'].iloc[-1]:.2f}")
            st.write(f"  High: {chart_df['high'].iloc[-1]:.2f}")
            st.write(f"  Low: {chart_df['low'].iloc[-1]:.2f}")
            st.write(f"  Close: {chart_df['close'].iloc[-1]:.2f}")
        
        # Create the candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=chart_df.index,
            open=chart_df['open'],
            high=chart_df['high'],
            low=chart_df['low'],
            close=chart_df['close'],
            name="Price",
            increasing_line_color='green',
            decreasing_line_color='red'
        )])
        
        # Add pattern detections if provided
        if detections:
            if debug:
                st.write(f"🔍 **Chart Debug:** Adding {len(detections)} pattern detections")
            
            for idx, price in detections:
                try:
                    # Ensure idx is within bounds
                    if 0 <= idx < len(chart_df):
                        x_val = chart_df.index[idx]
                        fig.add_annotation(
                            x=x_val,
                            y=price,
                            text=pattern_name or "Pattern",
                            showarrow=True,
                            arrowhead=2,
                            arrowcolor="red",
                            bgcolor="yellow",
                            bordercolor="red",
                            font=dict(size=10)
                        )
                except Exception as e:
                    if debug:
                        st.write(f"⚠️ **Chart Debug:** Failed to add pattern at {idx}: {e}")
        
        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            xaxis_title='Time',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            showlegend=True,
            **DEFAULT_CHART_CONFIG
        )
        
        # Add volume if available
        if 'volume' in chart_df.columns:
            fig.add_trace(go.Bar(
                x=chart_df.index,
                y=chart_df['volume'],
                name='Volume',
                yaxis='y2',
                opacity=0.3
            ))
            
            # Update layout for secondary y-axis
            fig.update_layout(
                yaxis2=dict(
                    title='Volume',
                    overlaying='y',
                    side='right'
                )
            )
        
        if debug:
            st.write(f"✅ **Chart Debug:** Chart created successfully")
        
        return fig
        
    except Exception as e:
        if debug:
            st.write(f"❌ **Chart Debug:** Chart creation failed: {e}")
            st.write(f"🔍 **Chart Debug:** Exception type: {type(e).__name__}")
            import traceback
            st.code(traceback.format_exc())
        
        logger.error(f"Failed to create candlestick chart: {e}")
        
        # Return error chart
        fig = go.Figure()
        fig.add_annotation(
            text=f"Chart Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color='red')
        )
        fig.update_layout(
            title=f"{title} (Error)",
            height=height,
            **DEFAULT_CHART_CONFIG
        )
        return fig
def validate_ohlc_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate OHLC dataframe structure used across dashboards."""
    required_columns = ['open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    # Check for valid OHLC relationships
    invalid_rows = df[(df['high'] < df['low']) | 
                     (df['high'] < df['open']) | 
                     (df['high'] < df['close']) |
                     (df['low'] > df['open']) | 
                     (df['low'] > df['close'])]
    
    if not invalid_rows.empty:
        return False, f"Invalid OHLC relationships in {len(invalid_rows)} rows"
    
    return True, "Valid OHLC data"

def safe_file_write(file_path: Path, content: str, create_backup: bool = True) -> Tuple[bool, str, Optional[Path]]:
    """Safely write file with backup and atomic operations."""
    backup_path = None
    
    try:
        # Create backup if file exists and backup requested
        if create_backup and file_path.exists():
            backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
            shutil.copy2(file_path, backup_path)
        
        # Write to temporary file first
        with tempfile.NamedTemporaryFile(mode='w', delete=False, 
                                       suffix=file_path.suffix) as tmp_file:
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)
        
        # Atomic move
        shutil.move(str(tmp_path), str(file_path))
        
        return True, "File written successfully", backup_path
        
    except Exception as e:
        return False, f"Failed to write file: {e}", backup_path

def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe columns to standard OHLC format."""
    column_mapping = {
        'Open': 'open', 'HIGH': 'high', 'High': 'high',
        'LOW': 'low', 'Low': 'low', 'CLOSE': 'close', 
        'Close': 'close', 'VOLUME': 'volume', 'Volume': 'volume',
        'Date': 'timestamp', 'DATE': 'timestamp', 'Datetime': 'timestamp'
    }
    
    # Apply column mapping
    df_normalized = df.rename(columns=column_mapping)
    
    # Ensure lowercase
    df_normalized.columns = df_normalized.columns.str.lower()
    
    return df_normalized

def render_file_upload_section(
    title: str,
    file_types: List[str], 
    max_size_mb: int = 10,
    help_text: str = ""
) -> Optional[Any]:
    """Render standardized file upload section."""
    st.subheader(title)
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=file_types,
        help=help_text or f"Max size: {max_size_mb}MB"
    )
    
    if uploaded_file:
        # Check file size
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            st.error(f"File size ({file_size_mb:.1f}MB) exceeds limit ({max_size_mb}MB)")
            return None
        
        st.success(f"File uploaded: {uploaded_file.name} ({file_size_mb:.1f}MB)")
    
    return uploaded_file

def show_success_with_actions(
    message: str,
    actions: List[Tuple[str, callable]] = None
) -> None:
    """Show success message with optional action buttons."""
    st.success(message)
    
    if actions:
        cols = st.columns(len(actions))
        for i, (action_text, action_func) in enumerate(actions):
            if cols[i].button(action_text):
                action_func()

class DashboardStateManager:
    """Manages dashboard session state and configuration."""
    
    def initialize_session_state(self):
        """Initialize dashboard-specific session state."""
        initialize_dashboard_session_state()
        
        # Additional dashboard-specific state
        dashboard_defaults = {
            'active_symbols': [],
            'trading_mode': 'demo',
            'risk_settings': {
                'max_position_size': 1000,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.05
            }
        }
        
        for key, default_value in dashboard_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value