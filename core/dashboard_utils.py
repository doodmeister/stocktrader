"""
Shared utilities for dashboard functionality.

Enhanced version with improved performance, error handling, and features
while maintaining full backward compatibility.
"""

import streamlit as st
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import logging
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import tempfile
import shutil
import time
import functools
import hashlib
from datetime import datetime
from contextlib import contextmanager
import traceback
import json

import psutil

# Check if psutil is available
try:
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Import existing pattern detection system
from patterns.patterns import create_pattern_detector, PatternResult

# Import security utilities
from security.utils import sanitize_user_input as _sanitize_user_input, validate_file_path as _validate_file_path
from security.encryption import generate_secure_token as _generate_secure_token

logger = logging.getLogger(__name__)

# Enhanced chart configuration constants
CHART_COLORS = {
    "bullish": "#2E8B57",
    "bearish": "#DC143C", 
    "neutral": "#4682B4",
    "highlight": "#FFD700",
    "background": "#F8F9FA",
    "grid": "#E5E5E5",
    "accent": "#FF6B6B",
    "success": "#28A745",
    "warning": "#FFC107",
    "danger": "#DC3545",
    "info": "#17A2B8"
}

DEFAULT_CHART_CONFIG = {
    "displayModeBar": False,
    "staticPlot": False,
    "responsive": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        'pan2d', 'lasso2d', 'select2d', 'autoScale2d',
        'resetScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian'
    ]
}

# Performance monitoring decorator
def monitor_performance(operation_name: str = ""):
    """Decorator to monitor performance of dashboard operations."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            memory_before = _get_memory_usage() if HAS_PSUTIL else 0
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                memory_after = _get_memory_usage() if HAS_PSUTIL else 0
                
                # Log performance if debug mode is enabled
                if st.session_state.get('enable_debug_mode', False):
                    memory_delta = memory_after - memory_before if HAS_PSUTIL else 0
                    logger.debug(
                        f"Performance [{operation_name or func.__name__}]: "
                        f"{duration:.3f}s, Memory: {memory_delta:+.1f}MB"
                    )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.warning(f"Performance [{operation_name or func.__name__}]: {duration:.3f}s, Error: {e}")
                raise
        return wrapper
    return decorator

def _get_memory_usage() -> float:
    """Get current memory usage in MB."""
    if not HAS_PSUTIL:
        return 0.0
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0

@monitor_performance("page_setup")
def setup_page(
    title: str,
    logger_name: str = __name__,
    initialize_session: bool = True,
    sidebar_title: Optional[str] = None,
    enable_debug: bool = False
):
    """
    Enhanced page setup for all dashboard pages.
    
    Args:
        title: Page title to display
        logger_name: Logger name (usually __name__)
        initialize_session: Whether to initialize session state
        sidebar_title: Optional sidebar title
        enable_debug: Enable debug features
    
    Returns:
        Logger instance
    """
    from utils.logger import setup_logger
    
    # Check if we're running within the modular dashboard system
    is_modular_mode = st.session_state.get('dashboard_initialized', False)
    
    # Enhanced page title with debug info
    title_display = title
    if enable_debug and HAS_PSUTIL:
        memory_usage = _get_memory_usage()
        title_display += f" (Memory: {memory_usage:.1f}MB)"
    
    # Only set page title if NOT in modular mode (to avoid overriding main dashboard header)
    if not is_modular_mode:
        st.title(title_display)
    else:
        # In modular mode, just add a subheader for the specific page
        st.subheader(title_display)
    
    # Setup logger
    logger = setup_logger(logger_name)
    
    # Initialize session state if requested
    if initialize_session:
        initialize_dashboard_session_state()
        # Add debug mode to session state
        if 'enable_debug_mode' not in st.session_state:
            st.session_state['enable_debug_mode'] = enable_debug
    
    # Setup sidebar if title provided
    if sidebar_title:
        st.sidebar.header(sidebar_title)
        
        # Add debug controls if enabled
        if enable_debug:
            _render_debug_controls()
    
    return logger

def _render_debug_controls():
    """Render debug controls in sidebar."""
    with st.sidebar.expander("🔧 Debug Controls", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Cache", help="Clear Streamlit cache"):
                st.cache_data.clear()
                if hasattr(st, 'cache_resource'):
                    st.cache_resource.clear()
                st.success("Cache cleared")
        
        with col2:
            if st.button("Session Info", help="Show session state info"):
                st.write(f"Session keys: {len(st.session_state)}")
                if HAS_PSUTIL:
                    st.write(f"Memory: {_get_memory_usage():.1f}MB")
        
        # Performance metrics
        if HAS_PSUTIL:
            memory_usage = _get_memory_usage()
            st.metric("Memory Usage", f"{memory_usage:.1f} MB")

def safe_streamlit_metric(label: str, value: str, delta: Optional[str] = None) -> None:
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

def enhanced_handle_streamlit_error(
    error: Exception, 
    context: str = "",
    show_traceback: bool = False,
    show_recovery: bool = True
) -> None:
    """
    Enhanced error handling with better context and recovery options.
    
    Args:
        error: Exception that occurred
        context: Additional context about where error occurred
        show_traceback: Whether to show full traceback in debug mode
        show_recovery: Whether to show recovery suggestions
    """
    error_id = hashlib.md5(f"{str(error)}{context}{time.time()}".encode()).hexdigest()[:8]
    error_msg = f"Error [{error_id}] in {context}: {str(error)}" if context else f"Error [{error_id}]: {str(error)}"
    logger.error(error_msg)
    
    st.error(f"🚨 {error_msg}")
    
    # Show technical details in debug mode
    if show_traceback and st.session_state.get('enable_debug_mode', False):
        with st.expander("🔍 Technical Details", expanded=False):
            st.code(traceback.format_exc())
    
    # Recovery suggestions
    if show_recovery:
        with st.expander("💡 Recovery Suggestions", expanded=False):
            recovery_suggestions = _get_recovery_suggestions(error, context)
            for suggestion in recovery_suggestions:
                st.info(suggestion)

def _get_recovery_suggestions(error: Exception, context: str) -> List[str]:
    """Get recovery suggestions based on error type and context."""
    suggestions = []
    
    error_type = type(error).__name__
    error_msg = str(error).lower()
    
    if "connection" in error_msg or "timeout" in error_msg:
        suggestions.append("🌐 Check your internet connection and try again")
        suggestions.append("⏱️ The service might be temporarily unavailable")
    
    elif "memory" in error_msg or error_type == "MemoryError":
        suggestions.append("💾 Try reducing the amount of data being processed")
        suggestions.append("🔄 Clear cache and restart the dashboard")
    
    elif "file" in error_msg or "permission" in error_msg:
        suggestions.append("📁 Check if the file exists and you have permission to access it")
        suggestions.append("🔒 Ensure the file is not open in another application")
    
    elif "key" in error_msg and context == "chart":
        suggestions.append("📊 Check that your data has the required columns (open, high, low, close)")
        suggestions.append("🔄 Try refreshing the data")
    
    else:
        suggestions.append("🔄 Try refreshing the page")
        suggestions.append("📋 Copy the error ID and report it if the problem persists")
    
    return suggestions

def handle_streamlit_error(error: Exception, context: str = "") -> None:
    """Original error handler for backward compatibility."""
    enhanced_handle_streamlit_error(error, context)

def enhanced_cache_key_builder(*args, include_session: bool = False, **kwargs) -> str:
    """Enhanced cache key builder with session state support."""
    key_parts = [str(arg) for arg in args] + [f"{k}={v}" for k, v in sorted(kwargs.items())]
    
    if include_session:
        # Include relevant session state for cache invalidation
        session_keys = ['user_id', 'session_id', 'data_version']
        session_parts = [
            f"session_{k}={st.session_state.get(k, 'none')}"
            for k in session_keys
            if k in st.session_state
        ]
        key_parts.extend(session_parts)
    
    key_string = "|".join(key_parts)
    return hashlib.sha256(key_string.encode()).hexdigest()[:16]

def cache_key_builder(*args, **kwargs) -> str:
    """Original cache key builder for backward compatibility."""
    return enhanced_cache_key_builder(*args, **kwargs)

def initialize_dashboard_session_state():
    """Initialize enhanced Streamlit session state variables."""
    defaults = {
        'dashboard_initialized': True,
        'error_count': 0,
        'last_update': datetime.now(),
        'user_preferences': {},
        'cache': {},
        'notifications': [],
        'current_page': None,
        'performance_metrics': [],
        'enable_debug_mode': False,
        'session_id': hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8],
        'memory_warnings': [],
        'last_memory_check': time.time(),
        'chart_config': DEFAULT_CHART_CONFIG.copy()
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

@contextmanager
def session_state_backup():
    """Context manager to backup and restore session state on errors."""
    backup = dict(st.session_state)
    try:
        yield
    except Exception:
        # Restore session state on error
        st.session_state.clear()
        st.session_state.update(backup)
        raise

def validate_session_state() -> bool:
    """Validate session state integrity and memory usage."""
    required_keys = ['dashboard_initialized', 'session_id']
    
    for key in required_keys:
        if key not in st.session_state:
            logger.warning(f"Missing required session state key: {key}")
            return False
    
    # Check memory usage if psutil is available
    if HAS_PSUTIL:
        memory_usage = _get_memory_usage()
        memory_limit = 500.0  # Default 500MB limit
        
        if memory_usage > memory_limit:
            logger.warning(f"Memory usage ({memory_usage:.1f}MB) exceeds limit ({memory_limit}MB)")
            st.session_state.setdefault('memory_warnings', []).append({
                'timestamp': datetime.now(),
                'usage': memory_usage,
                'limit': memory_limit
            })
            return False
    
    return True

# Enhanced chart utilities
@monitor_performance("chart_creation")
def create_candlestick_chart(
    df: pd.DataFrame, 
    title: str = "Price Chart",
    detections: Optional[List[Dict[str, Any]]] = None,  # Enhanced to use pattern analysis results
    pattern_name: Optional[str] = None,
    height: int = 500,
    debug: bool = False,
    show_patterns: bool = True,
    pattern_confidence_threshold: float = 0.7
) -> go.Figure:
    """
    Create enhanced candlestick chart with integrated pattern detection.
    
    This function now integrates with the existing comprehensive pattern detection
    system from patterns.patterns module to provide rich pattern annotations.
    
    Args:
        df: DataFrame with OHLC data
        title: Chart title
        detections: List of pattern detection results from analyze_patterns_for_chart()
        pattern_name: Optional pattern name for legacy compatibility
        height: Chart height in pixels
        debug: Enable debug output
        show_patterns: Whether to show pattern annotations
        pattern_confidence_threshold: Minimum confidence for showing patterns
        
    Returns:
        Plotly Figure object
    """
    
    if debug:
        st.write(f"🔍 **Chart Debug:** Input DataFrame shape: {df.shape}")
        st.write(f"🔍 **Chart Debug:** Columns: {list(df.columns)}")
        st.write(f"🔍 **Chart Debug:** Index type: {type(df.index)}")
        if not df.empty:
            st.write("🔍 **Chart Debug:** First few rows:")
            st.dataframe(df.head(3))
    
    try:
        # Create a copy to avoid modifying the original
        chart_df = df.copy()
        
        # Performance optimization for large datasets
        max_points = st.session_state.get('chart_config', {}).get('max_points', 10000)
        if len(chart_df) > max_points:
            # Intelligent sampling to preserve chart patterns
            step = len(chart_df) // max_points
            indices = list(range(0, len(chart_df), step))
            if indices[-1] != len(chart_df) - 1:
                indices.append(len(chart_df) - 1)
            chart_df = chart_df.iloc[indices].copy()
            
            if debug:
                st.info(f"📊 Chart optimized: {len(chart_df)} points (reduced from {len(df)})")
        
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
                    line=dict(color=CHART_COLORS['neutral'], width=2),
                    hovertemplate=f"<b>%{{x}}</b><br>{price_col.title()}: $%{{y:.2f}}<extra></extra>"
                ))
                
                fig.update_layout(
                    title=f"{title} (Line Chart - Limited Data)",
                    height=height,
                    xaxis_title='Time',
                    yaxis_title='Price',
                    template='plotly_white',
                    **DEFAULT_CHART_CONFIG
                )
                
                if debug:
                    st.write(f"✅ **Chart Debug:** Created line chart using {price_col}")
                
                return fig
            else:
                # No price data available
                return _create_empty_chart(title, height, "No price data available for chart")
        
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
                    st.write("⚠️ **Chart Debug:** No timestamp column found, using integer index")
        
        # Ensure OHLC columns are numeric
        for col in required_cols:
            if col in chart_df.columns:
                chart_df[col] = pd.to_numeric(chart_df[col], errors='coerce')
        
        # Remove any rows with NaN values in OHLC columns
        chart_df = chart_df.dropna(subset=required_cols)
        
        if chart_df.empty:
            if debug:
                st.write("❌ **Chart Debug:** DataFrame is empty after cleaning")
            return _create_empty_chart(title, height, "No valid data for chart after cleaning")
        
        if debug:
            st.write(f"✅ **Chart Debug:** Final chart data shape: {chart_df.shape}")
            st.write("🔍 **Chart Debug:** OHLC sample values:")
            st.write(f"  Open: {chart_df['open'].iloc[-1]:.2f}")
            st.write(f"  High: {chart_df['high'].iloc[-1]:.2f}")
            st.write(f"  Low: {chart_df['low'].iloc[-1]:.2f}")
            st.write(f"  Close: {chart_df['close'].iloc[-1]:.2f}")
        
        # Create enhanced candlestick chart
        fig = go.Figure()
        
        # Add candlestick trace with enhanced styling
        candlestick = go.Candlestick(
            x=chart_df.index,
            open=chart_df['open'],
            high=chart_df['high'],
            low=chart_df['low'],
            close=chart_df['close'],
            name="Price",
            increasing_line_color=CHART_COLORS['bullish'],
            decreasing_line_color=CHART_COLORS['bearish'],
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Open: $%{open:.2f}<br>"
                "High: $%{high:.2f}<br>"
                "Low: $%{low:.2f}<br>"
                "Close: $%{close:.2f}<br>"
                "<extra></extra>"
            )
        )
        
        fig.add_trace(candlestick)
          # Add enhanced pattern detections if provided
        if detections and show_patterns:
            if debug:
                st.write(f"🔍 **Chart Debug:** Adding {len(detections)} pattern detections")
            
            # Handle both legacy format (List[Tuple[int, float]]) and new format (List[Dict])
            for detection in detections[:20]:  # Increased limit for better visualization
                try:
                    if isinstance(detection, dict):
                        # New enhanced format from analyze_patterns_for_chart()
                        idx = detection.get('index', 0)
                        pattern = detection.get('pattern', 'Pattern')
                        confidence = detection.get('confidence', 0.0)
                        y_position = detection.get('y_position')
                        pattern_type = detection.get('pattern_type', 'unknown')
                        
                        # Filter by confidence threshold
                        if confidence < pattern_confidence_threshold:
                            continue
                            
                        # Determine annotation style based on pattern type
                        if 'bullish' in pattern_type.lower():
                            arrow_color = CHART_COLORS['bullish']
                            text_prefix = "🔺"
                        elif 'bearish' in pattern_type.lower():
                            arrow_color = CHART_COLORS['bearish']
                            text_prefix = "🔻"
                        else:
                            arrow_color = CHART_COLORS['neutral']
                            text_prefix = "📈"
                        
                        # Use calculated y_position or fallback to high price
                        if y_position is None and 0 <= idx < len(chart_df):
                            y_position = chart_df['high'].iloc[idx] * 1.02
                        
                        # Ensure idx is within bounds
                        if 0 <= idx < len(chart_df) and y_position is not None:
                            x_val = chart_df.index[idx]
                            fig.add_annotation(
                                x=x_val,
                                y=y_position,
                                text=f"{text_prefix} {pattern}<br>({confidence:.1%})",
                                showarrow=True,
                                arrowhead=2,
                                arrowcolor=arrow_color,
                                bgcolor="rgba(255,255,255,0.9)",
                                bordercolor=arrow_color,
                                font=dict(size=9, color='black'),
                                opacity=0.9,
                                ax=0,
                                ay=-30
                            )
                            
                    elif isinstance(detection, (tuple, list)) and len(detection) >= 2:
                        # Legacy format (idx, price) for backward compatibility
                        idx, price = detection[0], detection[1]
                        if 0 <= idx < len(chart_df):
                            x_val = chart_df.index[idx]
                            fig.add_annotation(
                                x=x_val,
                                y=price,
                                text=f"📈 {pattern_name or 'Pattern'}",
                                showarrow=True,
                                arrowhead=2,
                                arrowcolor=CHART_COLORS['highlight'],
                                bgcolor=CHART_COLORS['background'],
                                bordercolor=CHART_COLORS['highlight'],
                                font=dict(size=10),
                                opacity=0.9
                            )
                            
                except Exception as e:
                    if debug:
                        st.write(f"⚠️ **Chart Debug:** Failed to add pattern detection: {e}")
        
        # Enhanced layout
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            height=height,
            xaxis_title='Time',
            yaxis_title='Price ($)',
            xaxis_rangeslider_visible=False,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white',
            margin=dict(l=50, r=50, t=80, b=50),
            **DEFAULT_CHART_CONFIG
        )
        
        # Add volume if available
        if 'volume' in chart_df.columns and not chart_df['volume'].isna().all():
            # Create subplot with secondary y-axis
            volume_trace = go.Bar(
                x=chart_df.index,
                y=chart_df['volume'],
                name='Volume',
                yaxis='y2',
                opacity=0.3,
                marker_color=CHART_COLORS['info'],
                hovertemplate="Volume: %{y:,.0f}<extra></extra>"
            )
            
            fig.add_trace(volume_trace)
            
            # Update layout for secondary y-axis
            fig.update_layout(
                yaxis2=dict(
                    title='Volume',
                    overlaying='y',
                    side='right',
                    showgrid=False
                )
            )
        
        if debug:
            st.write("✅ **Chart Debug:** Chart created successfully")
        
        return fig
        
    except Exception as e:
        if debug:
            st.write(f"❌ **Chart Debug:** Chart creation failed: {e}")
            st.write(f"🔍 **Chart Debug:** Exception type: {type(e).__name__}")
            st.code(traceback.format_exc())
        
        logger.error(f"Failed to create candlestick chart: {e}")
        return _create_error_chart(title, height, str(e))

def _create_empty_chart(title: str, height: int, message: str) -> go.Figure:
    """Create empty chart with message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color=CHART_COLORS['neutral'])
    )
    fig.update_layout(
        title=title,
        height=height,
        template='plotly_white',
        **DEFAULT_CHART_CONFIG
    )
    return fig

def _create_error_chart(title: str, height: int, error_msg: str) -> go.Figure:
    """Create error chart with enhanced styling."""
    fig = go.Figure()
    fig.add_annotation(
        text=f"⚠️ Chart Error: {error_msg}",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color=CHART_COLORS['danger'])
    )
    fig.update_layout(
        title=f"{title} (Error)",
        height=height,
        template='plotly_white',
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
    actions: Optional[List[Tuple[str, Callable]]] = None
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

# =============================================================================
# Security and Input Validation Functions
# =============================================================================

def sanitize_user_input(input_text: str, max_length: int = 1000, allow_html: bool = False) -> str:
    """
    Sanitize user input to prevent XSS and other security issues.
    
    .. deprecated:: 
        Use security.utils.sanitize_user_input instead.
    
    Args:
        input_text: The input text to sanitize
        max_length: Maximum allowed length
        allow_html: Whether to allow HTML tags
    
    Returns:
        Sanitized input text
    """
    import warnings
    warnings.warn(
        "dashboard_utils.sanitize_user_input is deprecated. Use security.utils.sanitize_user_input instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _sanitize_user_input(input_text, max_length, allow_html)


def validate_file_path(file_path: Union[str, Path], allowed_extensions: Optional[List[str]] = None) -> bool:
    """
    Validate if a file path is safe and allowed.
    
    .. deprecated::
        Use security.utils.validate_file_path instead.
    
    Args:
        file_path: Path to validate
        allowed_extensions: List of allowed file extensions (e.g., ['.csv', '.json'])
    
    Returns:
        True if path is valid and safe
    """
    import warnings
    warnings.warn(
        "dashboard_utils.validate_file_path is deprecated. Use security.utils.validate_file_path instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _validate_file_path(file_path, allowed_extensions)


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token.
    
    .. deprecated::
        Use security.encryption.generate_secure_token instead.
    """
    import warnings
    warnings.warn(
        "dashboard_utils.generate_secure_token is deprecated. Use security.encryption.generate_secure_token instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _generate_secure_token(length)


# =============================================================================
# Advanced File Operations
# =============================================================================

def safe_json_load(file_path: Union[str, Path], default: Any = None) -> Any:
    """
    Safely load JSON data from a file with error handling.
    
    Args:
        file_path: Path to JSON file
        default: Default value to return on error
    
    Returns:
        Loaded JSON data or default value
    """
    try:
        if not validate_file_path(file_path, ['.json']):
            logger.warning(f"Invalid file path for JSON load: {file_path}")
            return default
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return default


def safe_json_save(data: Any, file_path: Union[str, Path], indent: int = 2) -> bool:
    """
    Safely save data to JSON file with error handling.
    
    Args:
        data: Data to save
        file_path: Path to save JSON file
        indent: JSON indentation
    
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        
        logger.info(f"Successfully saved JSON to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        return False


def safe_csv_operations(df: pd.DataFrame, file_path: Union[str, Path], operation: str = 'save') -> Union[pd.DataFrame, bool]:
    """
    Safely perform CSV operations with enhanced error handling.
    
    Args:
        df: DataFrame to save (for save operation) or None (for load operation)
        file_path: Path to CSV file
        operation: 'save' or 'load'
    
    Returns:
        DataFrame (for load) or success boolean (for save)
    """
    try:
        path = Path(file_path)
        
        if operation == 'save':
            if df is None or df.empty:
                logger.warning("Cannot save empty DataFrame")
                return False
            
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            logger.info(f"Successfully saved CSV to {file_path}")
            return True
            
        elif operation == 'load':
            if not validate_file_path(file_path, ['.csv']):
                logger.warning(f"Invalid CSV file path: {file_path}")
                return pd.DataFrame()
            
            df = pd.read_csv(path)
            logger.info(f"Successfully loaded CSV from {file_path}")
            return df
        else:
            logger.warning(f"Invalid operation: {operation}")
            return False if operation == 'save' else pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error in CSV {operation} operation for {file_path}: {e}")
        if operation == 'save':
            return False
        else:  # load operation
            return pd.DataFrame()


# =============================================================================
# Enhanced Notification System
# =============================================================================

def create_advanced_notification(
    message: str,
    notification_type: str = "info",
    dismissible: bool = True,
    auto_close: Optional[int] = None,
    actions: Optional[List[Dict[str, Any]]] = None
) -> None:
    """
    Create an advanced notification with enhanced features.
    
    Args:
        message: Notification message
        notification_type: Type of notification (info, success, warning, error)
        dismissible: Whether notification can be dismissed
        auto_close: Auto-close timeout in seconds
        actions: List of action buttons with callbacks
    """
    # Create unique notification ID
    notification_id = f"notification_{generate_secure_token(8)}"
    
    # Store notification in session state
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []
    
    notification_data = {
        'id': notification_id,
        'message': sanitize_user_input(message),
        'type': notification_type,
        'dismissible': dismissible,
        'auto_close': auto_close,
        'actions': actions or [],
        'timestamp': datetime.now(),
        'dismissed': False
    }
    
    st.session_state.notifications.append(notification_data)
    
    # Display notification
    _render_notification(notification_data)


def _render_notification(notification: Dict[str, Any]) -> None:
    """Render a notification with appropriate styling."""
    if notification.get('dismissed', False):
        return
    
    notification_type = notification.get('type', 'info')
    message = notification.get('message', '')
    
    # Choose appropriate Streamlit method based on type
    if notification_type == 'success':
        st.success(message)
    elif notification_type == 'warning':
        st.warning(message)
    elif notification_type == 'error':
        st.error(message)
    else:
        st.info(message)
    
    # Add action buttons if specified
    actions = notification.get('actions', [])
    if actions:
        cols = st.columns(len(actions))
        for i, action in enumerate(actions):
            with cols[i]:
                if st.button(action.get('label', 'Action'), key=f"{notification['id']}_action_{i}"):
                    callback = action.get('callback')
                    if callback and callable(callback):
                        callback()


def dismiss_notification(notification_id: str) -> None:
    """Dismiss a specific notification."""
    if 'notifications' in st.session_state:
        for notification in st.session_state.notifications:
            if notification['id'] == notification_id:
                notification['dismissed'] = True
                break


def clear_all_notifications() -> None:
    """Clear all notifications from session state."""
    if 'notifications' in st.session_state:
        st.session_state.notifications = []


# =============================================================================
# Advanced Chart Customization
# =============================================================================

def create_advanced_candlestick_chart(
    df: pd.DataFrame,
    title: str = "Stock Price Chart",
    height: int = 600,
    volume_subplot: bool = True,
    technical_indicators: Optional[Dict[str, Any]] = None,
    custom_styling: Optional[Dict[str, Any]] = None
) -> go.Figure:
    """
    Create an advanced candlestick chart with technical indicators and custom styling.
    
    Args:
        df: DataFrame with OHLC data
        title: Chart title
        height: Chart height
        volume_subplot: Whether to include volume subplot
        technical_indicators: Dictionary of technical indicators to add
        custom_styling: Custom styling options
      Returns:
        Plotly figure object
    """
    try:
        if not validate_ohlc_dataframe(df)[0]:  # validate_ohlc_dataframe returns (bool, str)
            return _create_error_chart(title, height, "Invalid OHLC data provided")
        
        # Default styling
        default_styling = {
            'increasing_color': '#00ff88',
            'decreasing_color': '#ff4444',
            'volume_color': '#888888',
            'background_color': '#0e1117',
            'grid_color': '#333333',
            'text_color': '#ffffff'
        }
        
        if custom_styling:
            default_styling.update(custom_styling)
        
        # Create subplots
        subplot_titles = [title]
        rows = 1
        
        if volume_subplot:
            subplot_titles.append("Volume")
            rows += 1
        
        if technical_indicators:
            for indicator_name in technical_indicators.keys():
                subplot_titles.append(indicator_name)
                rows += 1
        
        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=subplot_titles,
            row_heights=[0.6] + [0.2] * (rows - 1)
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="OHLC",
                increasing_line_color=default_styling['increasing_color'],
                decreasing_line_color=default_styling['decreasing_color']
            ),
            row=1, col=1
        )
        
        current_row = 2
        
        # Add volume subplot
        if volume_subplot and 'Volume' in df.columns:
            colors = [default_styling['increasing_color'] if df['Close'].iloc[i] >= df['Open'].iloc[i] 
                     else default_styling['decreasing_color'] for i in range(len(df))]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    marker_color=colors,
                    name="Volume",
                    opacity=0.7
                ),
                row=current_row, col=1
            )
            current_row += 1
          # Add technical indicators
        if technical_indicators:
            for indicator_name, indicator_data in technical_indicators.items():
                if isinstance(indicator_data, pd.Series):
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=indicator_data,
                            mode='lines',
                            name=indicator_name,
                            line=dict(width=2)
                        ),
                        row=current_row, col=1
                    )
                    current_row += 1
        
        # Apply custom styling
        fig.update_layout(
            title=title,
            height=height,
            xaxis_rangeslider_visible=False,
            plot_bgcolor=default_styling['background_color'],
            paper_bgcolor=default_styling['background_color'],
            font_color=default_styling['text_color'],
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(
            gridcolor=default_styling['grid_color'],
            showgrid=True
        )
        fig.update_yaxes(
            gridcolor=default_styling['grid_color'],
            showgrid=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating advanced candlestick chart: {e}")
        return _create_error_chart(title, height, f"Error creating chart: {str(e)}")


# =============================================================================
# Enhanced Caching System
# =============================================================================

def create_tiered_cache_key(*args, cache_level: str = "user", **kwargs) -> str:
    """
    Create a hierarchical cache key for multi-level caching.
    
    Args:
        *args: Positional arguments for cache key
        cache_level: Cache level (global, user, session)
        **kwargs: Keyword arguments for cache key
    
    Returns:
        Hierarchical cache key
    """
    base_key = cache_key_builder(*args, **kwargs)
    
    if cache_level == "global":
        return f"global_{base_key}"
    elif cache_level == "user":
        user_id = st.session_state.get('user_id', 'anonymous')
        return f"user_{user_id}_{base_key}"
    elif cache_level == "session":
        session_id = st.session_state.get('session_id', generate_secure_token(8))
        if 'session_id' not in st.session_state:
            st.session_state.session_id = session_id
        return f"session_{session_id}_{base_key}"
    else:
        return base_key


@contextmanager
def cache_context(cache_key: str, ttl: Optional[int] = None):
    """
    Context manager for caching operations with automatic cleanup.
    
    Args:
        cache_key: Cache key
        ttl: Time to live in seconds
    """
    start_time = time.time()
    try:
        yield cache_key
    finally:
        elapsed_time = time.time() - start_time
        logger.debug(f"Cache operation completed in {elapsed_time:.3f}s for key: {cache_key}")


# =============================================================================
# Backward Compatibility Wrapper
# =============================================================================

# Pattern Detection Integration Functions
@monitor_performance("pattern_detection")
def detect_candlestick_patterns(
    df: pd.DataFrame,
    pattern_names: Optional[List[str]] = None,
    confidence_threshold: float = 0.7,
    enable_caching: bool = True,
    parallel: bool = False
) -> List[PatternResult]:
    """
    Detect candlestick patterns using the existing comprehensive patterns system.
    
    This function integrates with the production-grade CandlestickPatterns class
    from patterns.patterns module to avoid code duplication.
    
    Args:
        df: DataFrame with OHLC data
        pattern_names: Specific patterns to detect (None for all)
        confidence_threshold: Minimum confidence for pattern detection
        enable_caching: Whether to enable result caching
        parallel: Whether to use parallel processing
        
    Returns:
        List of PatternResult objects for detected patterns
    """
    try:
        # Create pattern detector with specified configuration
        pattern_detector = create_pattern_detector(
            confidence_threshold=confidence_threshold,
            enable_caching=enable_caching
        )
          # Detect patterns using the existing system
        results = pattern_detector.detect_patterns(
            df, 
            pattern_names=pattern_names,
            parallel=parallel
        )
        
        logger.debug(f"Detected {len(results)} patterns with confidence >= {confidence_threshold}")
        return results
        
    except Exception as e:
        logger.error(f"Error in pattern detection: {e}")
        return []

@monitor_performance("pattern_analysis")
def analyze_patterns_for_chart(
    df: pd.DataFrame,
    selected_patterns: Optional[List[str]] = None,
    window_size: int = 5,
    confidence_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Analyze patterns across a DataFrame for chart visualization.
    
    This function performs sliding window pattern detection and returns
    results in a format suitable for chart annotations.
    
    Args:
        df: DataFrame with OHLC data
        selected_patterns: List of pattern names to detect
        window_size: Size of sliding window for detection
        confidence_threshold: Minimum confidence for pattern detection
        
    Returns:
        List of dictionaries with pattern detection results for charting
    """
    chart_patterns = []
    
    try:
        if df.empty:
            return chart_patterns
        
        # Get pattern detector
        pattern_detector = create_pattern_detector(confidence_threshold=confidence_threshold)
        
        # Get available patterns if none specified
        if selected_patterns is None:
            selected_patterns = pattern_detector.get_pattern_names()
        
        # Sliding window pattern detection
        for i in range(len(df)):
            # Define window boundaries
            window_start = max(0, i - window_size + 1)
            window = df.iloc[window_start:i+1].copy()
            
            # Need minimum data for pattern detection
            if len(window) < 1:
                continue
                
            try:
                # Detect patterns in current window
                detected_results = pattern_detector.detect_patterns(
                    df=window,
                    pattern_names=selected_patterns
                )
                
                # Process detected patterns
                for result in detected_results:
                    if result.detected and result.name in selected_patterns:
                        # Calculate y-position for chart annotation
                        y_position = df['high'].iloc[i] * 1.02  # Slightly above high
                        
                        chart_patterns.append({
                            "index": i,
                            "pattern": result.name,
                            "confidence": result.confidence,
                            "pattern_type": result.pattern_type.value,
                            "strength": result.strength.value,
                            "description": result.description,
                            "y_position": y_position,
                            "timestamp": df.index[i] if hasattr(df.index, 'name') else i
                        })
                        
            except Exception as pattern_error:
                logger.debug(f"Pattern detection error at index {i}: {pattern_error}")
                continue
        
        logger.info(f"Found {len(chart_patterns)} pattern occurrences")
        return chart_patterns
        
    except Exception as e:
        logger.error(f"Error in pattern analysis: {e}")
        return chart_patterns

def get_available_patterns() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all available candlestick patterns.
    
    Returns:
        Dictionary mapping pattern names to their information
    """
    try:
        pattern_detector = create_pattern_detector()
        pattern_names = pattern_detector.get_pattern_names()
        
        pattern_info = {}
        for name in pattern_names:
            try:
                info = pattern_detector.get_pattern_info(name)
                pattern_info[name] = info
            except Exception as e:
                logger.debug(f"Could not get info for pattern {name}: {e}")
                pattern_info[name] = {
                    "name": name,
                    "description": "Pattern information unavailable",
                    "min_rows": 1,
                    "pattern_type": "unknown",
                    "strength": "unknown"
                }
        
        return pattern_info
        
    except Exception as e:
        logger.error(f"Error getting pattern information: {e}")
        return {}

def validate_pattern_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate DataFrame for pattern detection compatibility.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, validation_message)
    """
    try:
        if df.empty:
            return False, "DataFrame is empty"
        
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
        
        # Check for numeric data
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return False, f"Column '{col}' must contain numeric data"
        
        # Check for null values
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            null_cols = null_counts[null_counts > 0].index.tolist()
            return False, f"Null values found in columns: {null_cols}"
        
        # Check OHLC relationships
        invalid_rows = df[
            (df['high'] < df[['open', 'close', 'low']].max(axis=1)) |
            (df['low'] > df[['open', 'close', 'high']].min(axis=1))
        ]
        
        if not invalid_rows.empty:
            return False, f"Found {len(invalid_rows)} rows with invalid OHLC relationships"
        
        return True, "DataFrame is valid for pattern detection"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"