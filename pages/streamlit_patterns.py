import streamlit as st
import inspect
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

# --- Section: Pattern Explorer ---
st.header("Pattern Explorer")

if pattern_names:
    selected_pattern = st.selectbox(
        "Select a pattern to explore:",
        options=pattern_names,
        key="pattern_selector"
    )
    
    if selected_pattern:
        st.subheader(f"📊 {selected_pattern} Pattern")
        
        # Get pattern implementation details
        try:
            # Get the method corresponding to the selected pattern
            pattern_method = getattr(CandlestickPatterns, f"is_{selected_pattern.lower().replace(' ', '_')}", None)
            
            if pattern_method:
                # Get the source code
                source_code = inspect.getsource(pattern_method)
                
                # Display the implementation
                with st.expander("🔍 Pattern Implementation (Technical)", expanded=False):
                    st.code(source_code, language="python")
                
                # Get docstring (explanation)
                doc = inspect.getdoc(pattern_method)
                
                if doc:
                    st.markdown("### Pattern Explanation")
                    st.write(doc)
                else:
                    # Generate simple explanation based on pattern name
                    st.markdown("### Pattern Explanation")
                    st.write(f"The {selected_pattern} pattern is used to identify potential market reversals or continuations.")
                    st.info("💡 Detailed explanation not available. Consider enhancing the pattern's docstring in patterns.py.")
                
                # Visual representation placeholder
                st.markdown("### Visual Example")
                st.info("Visual representation would be shown here. Consider adding a function to generate pattern visualizations.")
            else:
                st.warning(f"Implementation for '{selected_pattern}' pattern not found.")
                
        except Exception as e:
            st.error(f"Error retrieving pattern details: {e}")
else:
    st.warning("No patterns available to explore.")

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
        st.info("To see the updated patterns, please refresh the page.")
    except Exception as e:
        st.error(f"Failed to write to patterns.py: {e}")

# Add guidance for pattern documentation
with st.sidebar.expander("How to Document Patterns", expanded=False):
    st.markdown("""
    ### How to Write Good Pattern Explanations
    
    When updating pattern implementations in `patterns.py`, add detailed docstrings to explain:
    
    1. **What the pattern looks like**
    2. **Market psychology behind the pattern**
    3. **Mathematical conditions** that define the pattern
    4. **Plain English explanation** of what it indicates
    5. **Trading implications** (bullish/bearish signals)
    
    Example format:
    ```python
    def is_doji(self, candle):
        """
        Identifies a Doji candlestick pattern.
        
        A Doji forms when the opening and closing prices are virtually equal.
        It represents market indecision where bulls and bears reached equilibrium.
        
        Mathematical condition:
        - |close - open| <= (high - low) * 0.05
        
        In plain terms: The body of the candle is very small (less than 5% of the total range),
        while the wicks can be of any length.
        
        Trading implication: Potential reversal signal, especially at support/resistance levels.
        """
        # Implementation code...
    ```
    """)
    ### How to Write Good Pattern Explanations
    
    When updating pattern implementations in `patterns.py`, add detailed docstrings to explain:
    
    1. **What the pattern looks like**
    2. **Market psychology behind the pattern**
    3. **Mathematical conditions** that define the pattern
    4. **Plain English explanation** of what it indicates
    5. **Trading implications** (bullish/bearish signals)
    
    Example format:
    ```python
    def is_doji(self, candle):
        """
        Identifies a Doji candlestick pattern.
        
        A Doji forms when the opening and closing prices are virtually equal.
        It represents market indecision where bulls and bears reached equilibrium.
        
        Mathematical condition:
        - |close - open| <= (high - low) * 0.05
        
        In plain terms: The body of the candle is very small (less than 5% of the total range),
        while the wicks can be of any length.
        
        Trading implication: Potential reversal signal, especially at support/resistance levels.
        """
        # Implementation code...
    ```
    """)

st.sidebar.info("Use the sections above to explore, inspect, export, and upload your candlestick pattern definitions.")
