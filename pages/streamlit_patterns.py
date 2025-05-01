import streamlit as st
from patterns import CandlestickPatterns

st.set_page_config(
    page_title="Candlestick Patterns Editor",
    layout="wide"
)

st.title("🔧 Candlestick Patterns Editor")

# --- Section: Inspect Current Patterns ---
st.header("Current Patterns")

def get_pattern_names():
    """Get pattern names from CandlestickPatterns."""
    try:
        return CandlestickPatterns.get_pattern_names()
    except Exception as e:
        st.error(f"Could not get pattern names: {e}")
        return []

pattern_names = get_pattern_names()
st.write(f"Detected **{len(pattern_names)}** patterns:")
for name in pattern_names:
    st.write(f"- {name}")

# --- Section: Export patterns.py for Editing ---
st.header("Export patterns.py for Editing")
code = ""
try:
    with open("patterns.py", "r") as f:
        code = f.read()
except Exception as e:
    st.error(f"Could not read patterns.py: {e}")

st.text_area(
    "Copy the current patterns.py code below and paste it into ChatGPT for updates:",
    code,
    height=350,
    key="export_code"
)

# --- Section: Upload Updated patterns.py ---
st.header("Upload Updated patterns.py")
st.markdown("Upload your modified patterns.py file (it will overwrite the existing one).")
uploaded = st.file_uploader(
    label="Choose Updated patterns.py",
    type="py",
    key="upload_patterns"
)

if uploaded:
    new_code = uploaded.getvalue().decode("utf-8")
    try:
        with open("patterns.py", "w") as f:
            f.write(new_code)
        st.success("✅ patterns.py has been updated successfully!")
    except Exception as e:
        st.error(f"Failed to write to patterns.py: {e}")

st.sidebar.info("Use the sections above to inspect, export, and upload your candlestick pattern definitions.")
