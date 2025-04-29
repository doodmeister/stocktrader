import streamlit as st
import ast

st.set_page_config(
    page_title="Candlestick Patterns Editor",
    layout="wide"
)

st.title("🔧 Candlestick Patterns Editor")

# --- Section: Inspect Current Patterns ---
st.header("Current Patterns")

def get_pattern_names():
    """Parse patterns.py and return the list of pattern names."""
    try:
        with open("patterns.py", "r") as f:
            source = f.read()
    except Exception as e:
        st.error(f"Could not open patterns.py: {e}")
        return []

    names = []
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            # Find the assignment to pattern_checks
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "pattern_checks":
                        # node.value should be a List of Tuples
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Tuple) and elt.elts:
                                name_node = elt.elts[0]
                                if hasattr(name_node, 'value'):
                                    names.append(name_node.value)
                        return names
    except Exception as e:
        st.error(f"Error parsing patterns.py: {e}")
    return names

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
