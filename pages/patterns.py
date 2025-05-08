# stocktrader/streamlit_patterns.py

from utils.logger import setup_logger
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from patterns.pattern_utils import (
    read_patterns_file,
    write_patterns_file,
    get_pattern_names,
    get_pattern_method,
    get_pattern_source_and_doc,
    validate_python_code
)
from patterns.patterns import PatternDetectionError, CandlestickPatterns
from utils.dashboard_utils import initialize_dashboard_session_state

# ─── Logging Setup ─────────────────────────────────────────────────────────────
logger = setup_logger(__name__)

# ─── Streamlit Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Candlestick Patterns Editor",
    layout="wide"
)

# Path constant (if you ever need it)
PATTERNS_PATH = Path(__file__).resolve().parent / "patterns.py"

# ─── CACHED LOADERS ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_pattern_names() -> List[str]:
    return get_pattern_names()

@st.cache_data(show_spinner=False)
def load_patterns_source() -> str:
    return read_patterns_file()

# ─── Pattern Examples ───────────────────────────────────────────────────────
@st.cache_data
def get_pattern_examples() -> Dict[str, pd.DataFrame]:
    """Generate example data for each pattern"""
    examples = {}
    
    # Hammer example
    df_hammer = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=5),
        'open': [100, 98, 95, 93, 91],
        'high': [101, 99, 96, 94, 94],
        'low': [99, 97, 94, 87, 87],
        'close': [98, 95, 93, 91, 92]
    })
    examples['Hammer'] = df_hammer
    
    # Bullish Engulfing example
    df_bullish_engulfing = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=5),
        'open': [100, 98, 95, 94, 89],
        'high': [101, 99, 96, 95, 94],
        'low': [99, 97, 94, 92, 88],
        'close': [98, 95, 93, 90, 93]
    })
    examples['Bullish Engulfing'] = df_bullish_engulfing
    
    # Morning Star example
    df_morning_star = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=7),
        'open': [100, 98, 95, 93, 91, 89, 91],
        'high': [101, 99, 96, 94, 91, 90, 95],
        'low': [99, 97, 94, 92, 88, 88, 90],
        'close': [98, 95, 93, 91, 88, 89, 94]
    })
    examples['Morning Star'] = df_morning_star
    
    # Piercing Pattern example
    df_piercing = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=5),
        'open': [100, 98, 95, 93, 88],
        'high': [101, 99, 96, 94, 93],
        'low': [99, 97, 94, 92, 87],
        'close': [98, 95, 93, 90, 92]
    })
    examples['Piercing Pattern'] = df_piercing
    
    # Bullish Harami example
    df_bullish_harami = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=5),
        'open': [100, 98, 95, 97, 92],
        'high': [101, 99, 96, 97, 94],
        'low': [99, 97, 94, 91, 91],
        'close': [98, 95, 93, 91, 93]
    })
    examples['Bullish Harami'] = df_bullish_harami
    
    # Three White Soldiers example
    df_three_white_soldiers = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=7),
        'open': [100, 98, 95, 93, 90, 92, 94],
        'high': [101, 99, 96, 94, 94, 97, 99],
        'low': [99, 97, 94, 92, 89, 91, 93],
        'close': [98, 95, 93, 91, 93, 96, 98]
    })
    examples['Three White Soldiers'] = df_three_white_soldiers
    
    # Inverted Hammer example
    df_inverted_hammer = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=5),
        'open': [100, 98, 95, 93, 89],
        'high': [101, 99, 96, 94, 94],
        'low': [99, 97, 94, 92, 88],
        'close': [98, 95, 93, 91, 90]
    })
    examples['Inverted Hammer'] = df_inverted_hammer
    
    # Doji example
    df_doji = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=5),
        'open': [100, 98, 95, 93, 91],
        'high': [101, 99, 96, 94, 94],
        'low': [99, 97, 94, 92, 88],
        'close': [98, 95, 93, 91, 91]
    })
    examples['Doji'] = df_doji
    
    # Morning Doji Star example
    df_morning_doji_star = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=7),
        'open': [100, 98, 95, 93, 91, 91, 91],
        'high': [101, 99, 96, 94, 94, 93, 95],
        'low': [99, 97, 94, 92, 88, 89, 91],
        'close': [98, 95, 93, 91, 88, 91, 94]
    })
    examples['Morning Doji Star'] = df_morning_doji_star
    
    # Bullish Abandoned Baby example
    df_bullish_abandoned_baby = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=7),
        'open': [100, 98, 95, 93, 91, 87, 88],
        'high': [101, 99, 96, 94, 94, 88, 93],
        'low': [99, 97, 94, 92, 86, 86, 87],
        'close': [98, 95, 93, 91, 86, 87, 92]
    })
    examples['Bullish Abandoned Baby'] = df_bullish_abandoned_baby
    
    # Bullish Belt Hold example
    df_bullish_belt_hold = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=5),
        'open': [100, 98, 95, 93, 87],
        'high': [101, 99, 96, 94, 93],
        'low': [99, 97, 94, 92, 87],
        'close': [98, 95, 93, 91, 92]
    })
    examples['Bullish Belt Hold'] = df_bullish_belt_hold
    
    # Three Inside Up example
    df_three_inside_up = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=7),
        'open': [100, 98, 95, 93, 95, 92, 93],
        'high': [101, 99, 96, 94, 95, 94, 96],
        'low': [99, 97, 94, 92, 90, 91, 92],
        'close': [98, 95, 93, 91, 90, 93, 95]
    })
    examples['Three Inside Up'] = df_three_inside_up
    
    # Rising Window example
    df_rising_window = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=5),
        'open': [100, 102, 104, 107, 110],
        'high': [103, 105, 107, 110, 113],
        'low': [99, 101, 103, 106, 109],
        'close': [102, 104, 106, 109, 112]
    })
    examples['Rising Window'] = df_rising_window
    
    return examples

def detect_pattern_in_example(pattern_name: str, df: pd.DataFrame) -> Tuple[bool, int]:
    """
    Detect the specified pattern in the example data.
    Returns a tuple of (found, index) where index is the position where the pattern was found.
    """
    method = get_pattern_method(pattern_name)
    if method is None:
        return False, -1
    
    # Rename columns to match what the pattern detection expects
    df_copy = df.copy()
    df_copy.columns = [col.lower() for col in df_copy.columns]
    
    # We'll use a sliding window approach to find the pattern
    window_size = 3  # Most patterns need 1-3 candles
    
    for i in range(len(df_copy) - window_size + 1):
        window = df_copy.iloc[i:i+window_size]
        try:
            if method(window):
                return True, i + window_size - 1
        except Exception:
            pass
    
    # Try with the full dataframe as a last resort
    try:
        if method(df_copy):
            return True, len(df_copy) - 1
    except Exception:
        pass
    
    return False, -1

def render_pattern_chart(pattern_name: str, df: pd.DataFrame) -> go.Figure:
    """Create a candlestick chart with pattern highlighting"""
    # Standardize columns to lower-case for pattern detection
    df.columns = [col.lower() for col in df.columns]

    # Find where the pattern occurs
    found, pattern_idx = detect_pattern_in_example(pattern_name, df)
    
    fig = go.Figure()
    
    # Add candlesticks
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'], 
            high=df['high'],
            low=df['low'], 
            close=df['close'],
            name="Price"
        )
    )
    
    # Add marker at pattern location if found
    if found and pattern_idx >= 0:
        fig.add_trace(
            go.Scatter(
                x=[df['date'].iloc[pattern_idx]],
                y=[df['high'].iloc[pattern_idx] * 1.02],
                mode="markers+text",
                marker=dict(symbol="triangle-down", size=15, color="green"),
                text=["Pattern"],
                textposition="top center",
                name="Pattern"
            )
        )
        
        # Highlight the candles involved in the pattern
        highlight_start = max(0, pattern_idx - 2)
        for i in range(highlight_start, pattern_idx + 1):
            fig.add_shape(
                type="rect",
                x0=df['date'].iloc[i] - pd.Timedelta(hours=12),
                x1=df['date'].iloc[i] + pd.Timedelta(hours=12),
                y0=df['low'].iloc[i] * 0.99,
                y1=df['high'].iloc[i] * 1.01,
                line=dict(color="rgba(50, 200, 50, 0.5)"),
                fillcolor="rgba(50, 200, 50, 0.1)"
            )
    
    # Layout
    fig.update_layout(
        title=f"{pattern_name} Pattern Example",
        xaxis_title="Date",
        yaxis_title="Price",
        height=400,
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )
    
    return fig

# ─── UI COMPONENTS ────────────────────────────────────────────────────────────

def render_title_and_header():
    st.title("🔧 Candlestick Patterns Editor")
    st.markdown("""
    This tool allows you to view, explore, and modify candlestick pattern definitions 
    used throughout the Stock Trader application.
    """)

def render_current_patterns(patterns: List[str]):
    st.header("Current Patterns")
    if not patterns:
        st.warning("No patterns were detected in the system.")
        return
    st.write(f"Detected **{len(patterns)}** patterns:")
    for name in patterns:
        st.write(f"- {name}")

def render_pattern_explorer(patterns: List[str]):
    st.header("Pattern Explorer")
    if not patterns:
        st.warning("No patterns available to explore.")
        return

    selected = st.selectbox("Select a pattern:", patterns, key="pattern_selector")
    if not selected:
        return

    st.subheader(f"📊 {selected}")
    method = get_pattern_method(selected)
    if method is None:
        st.warning(f"No implementation found for `{selected}`.")
        return

    # Source & doc
    src, doc = get_pattern_source_and_doc(method)
    with st.expander("🔍 Implementation", expanded=False):
        st.code(src, language="python")

    st.markdown("### Pattern Explanation")
    if doc:
        st.write(doc)
    else:
        st.info("No docstring available. Consider adding details in `patterns.py`.")

    st.markdown("### Visual Example")
    # Get example data for this pattern
    examples = get_pattern_examples()
    if selected in examples:
        fig = render_pattern_chart(selected, examples[selected])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No example data available for this pattern.")

def render_export_section():
    st.header("Export patterns.py for Editing")
    code = load_patterns_source()
    st.text_area(
        "Copy the code below, paste into ChatGPT for edits, then re-upload:",
        code,
        height=300,
        key="export_code"
    )
    st.download_button("Download patterns.py", data=code, file_name="patterns.py")

def render_upload_section():
    st.header("Upload Updated patterns.py")
    uploaded = st.file_uploader("Choose updated patterns.py", type="py", key="upload")
    if not uploaded:
        return

    new_code = uploaded.getvalue().decode("utf-8")
    if not validate_python_code(new_code):
        st.error("Invalid Python syntax.")
        return

    if "class CandlestickPatterns" not in new_code:
        if not st.confirm("No `CandlestickPatterns` class found. Overwrite anyway?"):
            return

    err = write_patterns_file(new_code)
    if err:
        st.error(f"Failed to write: {err}")
    else:
        st.success("✅ patterns.py updated. Please refresh.")

def render_visualizer(patterns: List[str]):
    st.header("📈 Pattern Visualizer")
    uploaded = st.file_uploader("Upload OHLC CSV", type="csv", key="viz_upload")
    pattern = st.selectbox("Pattern to highlight", patterns, key="viz_pattern")

    if not uploaded or not pattern:
        return

    df = pd.read_csv(uploaded)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"])
    else:
        st.error("CSV must have a 'date' or 'timestamp' column.")
        return

    # Standardize columns to lower-case for pattern detection
    df.columns = [col.lower() for col in df.columns]

    method = get_pattern_method(pattern)
    df["signal"] = df.apply(lambda row: method(row), axis=1)

    # For plotting, Plotly expects capitalized column names
    df_plot = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "date": "date"
    })

    if not df["signal"].any():
        st.info("No instances of this pattern detected.")
        return

    # Plotly chart
    fig = go.Figure(data=[
        go.Candlestick(
            x=df_plot["date"], open=df_plot["Open"], high=df_plot["High"],
            low=df_plot["Low"], close=df_plot["Close"], name="Price"
        ),
        go.Scatter(
            x=df_plot.loc[df["signal"], "date"],
            y=df_plot.loc[df["signal"], "High"] * 1.01,
            mode="markers",
            marker=dict(symbol="triangle-up", size=12),
            name="Signal"
        )
    ])
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

def render_sidebar():
    with st.sidebar.expander("📚 How to Document Patterns", expanded=False):
        st.markdown("""
        **Docstring tips**:
        1. What the pattern looks like  
        2. Market psychology  
        3. Mathematical conditions  
        4. Plain-English explanation  
        5. Trading implications  
        """)
    if st.sidebar.button("Reload all", use_container_width=True):
        st.experimental_rerun()

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    initialize_dashboard_session_state()
    try:
        render_title_and_header()
        patterns = load_pattern_names()

        render_current_patterns(patterns)
        render_pattern_explorer(patterns)
        render_export_section()
        render_upload_section()
        render_visualizer(patterns)
        render_sidebar()

    except Exception as e:
        logger.exception("Unexpected error")
        st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()