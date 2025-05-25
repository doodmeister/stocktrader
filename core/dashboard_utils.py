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
    """
    Build cache key from arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        str: Cache key
    """
    key_parts = [str(arg) for arg in args]
    key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
    return "_".join(key_parts)

def initialize_dashboard_session_state():
    """Initialize common Streamlit session state variables if they don't exist."""
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = None
    if 'data_cache' not in st.session_state:
        st.session_state.data_cache = None
    if 'error_count' not in st.session_state:
        st.session_state.error_count = 0
    if 'login_status' not in st.session_state:
        st.session_state.login_status = False

# New unified functions
def create_candlestick_chart(
    df: pd.DataFrame, 
    title: str = "Price Chart",
    detections: List[Tuple[int, float]] = None,
    pattern_name: str = None,
    height: int = 500
) -> go.Figure:
    """Create standardized candlestick chart used across all dashboards."""
    try:
        # Normalize column names
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]
        
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Ensure date column
        if 'date' not in df.columns:
            df['date'] = pd.date_range(start='2023-01-01', periods=len(df))
        
        # Create chart with volume subplot if available
        has_volume = 'volume' in df.columns
        specs = [[{"secondary_y": True}]] if has_volume else [[{}]]
        fig = make_subplots(specs=specs)
        
        # Main candlestick
        fig.add_trace(
            go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Price",
                increasing_line_color=CHART_COLORS["bullish"],
                decreasing_line_color=CHART_COLORS["bearish"]
            )
        )
        
        # Add volume if available
        if has_volume:
            fig.add_trace(
                go.Bar(
                    x=df['date'],
                    y=df['volume'],
                    name="Volume",
                    marker_color=CHART_COLORS["neutral"],
                    opacity=0.3,
                    yaxis="y2"
                )
            )
        
        # Add pattern markers if provided
        if detections and pattern_name:
            for idx, confidence in detections:
                if idx < len(df):
                    marker_color = CHART_COLORS["bullish"] if confidence > 0.7 else CHART_COLORS["neutral"]
                    marker_size = 12 + (confidence * 8)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[df['date'].iloc[idx]],
                            y=[df['high'].iloc[idx] * 1.02],
                            mode="markers+text",
                            marker=dict(
                                symbol="triangle-down",
                                size=marker_size,
                                color=marker_color,
                                line=dict(width=2, color="white")
                            ),
                            text=[f"{pattern_name}<br>({confidence:.1%})"],
                            textposition="top center",
                            name=f"Pattern (Conf: {confidence:.1%})",
                            showlegend=True
                        )
                    )
        
        # Layout
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16, family="Arial Black")),
            xaxis_title="Date",
            yaxis_title="Price", 
            height=height,
            template="plotly_white",
            hovermode="x unified",
            xaxis_rangeslider_visible=False,
            margin=dict(l=50, r=50, t=50, b=50),
            legend=dict(
                yanchor="top", y=0.99, xanchor="left", x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            )
        )
        
        if has_volume:
            fig.update_yaxes(title_text="Volume", secondary_y=True)
            fig.update_yaxes(showgrid=False, secondary_y=True)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating chart: {e}")
        # Return empty chart as fallback
        fig = go.Figure()
        fig.add_annotation(
            text=f"Chart generation failed: {str(e)}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=height, template="plotly_white")
        return fig

def validate_ohlc_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate OHLC dataframe structure used across dashboards."""
    try:
        # Normalize column names for checking
        df_columns_lower = [col.lower() for col in df.columns]
        
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df_columns_lower]
        
        if missing_cols:
            return False, f"Missing required columns: {', '.join(missing_cols)}"
        
        # Check for date/timestamp column
        date_cols = ['date', 'timestamp', 'time']
        has_date = any(date_col in df_columns_lower for date_col in date_cols)
        if not has_date:
            return False, "Missing date/timestamp column"
        
        # Validate numeric data
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for required_col in numeric_cols:
            if required_col in df_columns_lower:
                actual_col = df.columns[df_columns_lower.index(required_col)]
                if not pd.api.types.is_numeric_dtype(df[actual_col]):
                    return False, f"Column '{actual_col}' must contain numeric data"
        
        # Check minimum data points
        if len(df) < 3:
            return False, "Minimum 3 data points required"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def safe_file_write(file_path: Path, content: str, create_backup: bool = True) -> Tuple[bool, str, Optional[Path]]:
    """Safely write file with backup and atomic operations."""
    backup_path = None
    temp_file_path = None
    
    try:
        # Create backup if requested and file exists
        if create_backup and file_path.exists():
            backup_path = file_path.with_suffix(f'.backup_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.py')
            shutil.copy2(file_path, backup_path)
        
        # Write to temporary file first
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.tmp', delete=False, 
            encoding='utf-8', dir=file_path.parent
        ) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            import os
            os.fsync(temp_file.fileno())
            temp_file_path = Path(temp_file.name)
        
        # Atomic move
        shutil.move(str(temp_file_path), str(file_path))
        
        # Verify
        if not file_path.exists():
            return False, "File write verification failed", backup_path
        
        return True, "File written successfully", backup_path
        
    except Exception as e:
        # Cleanup temp file
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink(missing_ok=True)
        
        return False, f"Write failed: {str(e)}", backup_path

def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe columns to standard OHLC format."""
    df_copy = df.copy()
    
    # Column mapping
    column_mapping = {
        'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
        'volume': 'volume', 'date': 'date', 'timestamp': 'date', 'time': 'date'
    }
    
    # Create rename dictionary
    rename_dict = {}
    df_columns_lower = [col.lower() for col in df_copy.columns]
    
    for i, col_lower in enumerate(df_columns_lower):
        if col_lower in column_mapping:
            original_col = df_copy.columns[i]
            standard_name = column_mapping[col_lower]
            if original_col != standard_name:
                rename_dict[original_col] = standard_name
    
    # Apply renaming
    if rename_dict:
        df_copy = df_copy.rename(columns=rename_dict)
    
    return df_copy

def render_file_upload_section(
    title: str,
    file_types: List[str], 
    max_size_mb: int = 10,
    help_text: str = None
) -> Optional[Any]:
    """Render standardized file upload section used across dashboards."""
    st.subheader(title)
    
    uploaded_file = st.file_uploader(
        f"Choose file ({', '.join(file_types)})",
        type=file_types,
        accept_multiple_files=False,
        help=help_text or f"Upload a file (max {max_size_mb}MB)"
    )
    
    if uploaded_file:
        # Validate file size
        if uploaded_file.size > max_size_mb * 1024 * 1024:
            st.error(f"File size exceeds {max_size_mb}MB limit")
            return None
        
        # Show file info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("File Size", f"{uploaded_file.size:,} bytes")
        with col2:
            st.metric("File Type", uploaded_file.type)
    
    return uploaded_file

def show_success_with_actions(
    message: str, 
    actions: Dict[str, callable] = None,
    show_balloons: bool = True
) -> None:
    """Show success message with optional action buttons."""
    if show_balloons:
        st.balloons()
    
    st.success(message)
    
    if actions:
        cols = st.columns(len(actions))
        for i, (label, callback) in enumerate(actions.items()):
            if cols[i].button(label, key=f"action_{i}"):
                callback()