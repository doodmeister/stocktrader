import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Any, Dict, Optional, Tuple, List
from dataclasses import asdict

from train.model_manager import ModelManager
from patterns.patterns_nn import PatternNN

# --- Logging Setup ---
logger = logging.getLogger("model_visualizer")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

# --- Utility Functions ---

@st.cache_resource(show_spinner=False)
def load_model(model_class: Optional[type], model_file: str) -> Tuple[Any, Dict]:
    """
    Loads a model and its metadata using ModelManager.
    """
    try:
        model_manager = ModelManager()
        model, metadata = model_manager.load_model(model_class, model_file)
        return model, metadata
    except Exception as e:
        logger.error(f"Error loading model {model_file}: {e}")
        st.error(f"Failed to load model: {e}")
        return None, {}

def get_model_type(metadata: Dict, model_file: str) -> str:
    """
    Determines the model type from metadata or file extension.
    """
    if "model_type" in metadata:
        return metadata["model_type"]
    if model_file.endswith(".pth"):
        return "PatternNN"
    return "Classic ML"

def display_patternnn_model(model: Any, metadata: Dict, model_file: str):
    """
    Displays architecture, parameters, weights, and metrics for PatternNN models.
    """
    st.subheader("PatternNN Model Architecture")
    col1, col2 = st.columns(2)
    with col1:
        st.text(str(model))
    with col2:
        st.subheader("Model Parameters")
        st.json(metadata.get("parameters", {}))

    if st.checkbox(f"Show weights for {model_file}"):
        st.subheader("Model Weights")
        for name, param in model.named_parameters():
            st.write(f"**{name}**: shape {tuple(param.shape)}")
            st.write(param.data.cpu().numpy())

    if "metrics" in metadata:
        st.subheader("Performance Metrics")
        st.json(metadata["metrics"])
        return metadata["metrics"]
    return None

def display_classic_ml_model(model: Any, metadata: Dict):
    """
    Displays summary, parameters, feature importances, coefficients, and metrics for classic ML models.
    """
    st.subheader("Classic ML Model")
    col1, col2 = st.columns(2)
    with col1:
        st.text(str(model))
    with col2:
        st.subheader("Model Parameters")
        st.json(metadata.get("parameters", {}))

    if hasattr(model, "feature_importances_"):
        st.subheader("Feature Importances")
        importances = model.feature_importances_
        features = metadata.get("parameters", {}).get("features", [])
        if features and len(features) == len(importances):
            df = pd.DataFrame({"Feature": features, "Importance": importances})
            st.bar_chart(df.set_index("Feature"))
        else:
            st.write(importances)
    elif hasattr(model, "coef_"):
        st.subheader("Model Coefficients")
        st.write(model.coef_)

    if "metrics" in metadata:
        st.subheader("Performance Metrics")
        st.json(metadata["metrics"])
        return metadata["metrics"]
    return None

def compare_metrics(all_metrics: Dict[str, Dict]):
    """
    Visualizes metric comparisons across models.
    """
    if len(all_metrics) > 1:
        st.markdown("## 📊 Metric Comparison Across Models")
        metric_df = pd.DataFrame(all_metrics).T
        for metric in metric_df.columns:
            st.subheader(f"Metric: {metric}")
            try:
                st.bar_chart(metric_df[[metric]])
            except Exception as e:
                logger.warning(f"Could not plot metric '{metric}': {e}")

# --- Main Page Logic ---

def main():
    st.title("🔍 Model Visualizer & Comparator")

    model_manager = ModelManager()
    try:
        model_files = [
            f for f in model_manager.list_models()
            if f.endswith((".pth", ".joblib", ".pkl"))  # Only show model files
        ]
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        st.error("Could not list models. Please check model directory and permissions.")
        return

    if not model_files:
        st.warning("No trained models found. Train and save a model first.")
        return

    selected_model_files = st.multiselect(
        "Select Model Files to Compare",
        options=model_files,
        default=[model_files[0]],
        help="Select one or more models to visualize and compare."
    )

    all_metrics = {}

    for selected_model_file in selected_model_files:
        st.markdown(f"## Model: `{selected_model_file}`")
        # Determine model type by extension
        if selected_model_file.endswith(".pth"):
            model_type = "PatternNN"
            model, metadata = load_model(PatternNN, selected_model_file)
        else:
            model_type = "Classic ML"
            model, metadata = load_model(None, selected_model_file)

        st.write(f"Detected model type: **{model_type}**")

        if model is None or metadata is None:
            continue

        # Convert ModelMetadata to dict for compatibility
        if not isinstance(metadata, dict):
            metadata = asdict(metadata)

        if model_type == "PatternNN":
            metrics = display_patternnn_model(model, metadata, selected_model_file)
        else:
            metrics = display_classic_ml_model(model, metadata)

        if metrics:
            all_metrics[selected_model_file] = metrics

    compare_metrics(all_metrics)

if __name__ == "__main__":
    main()

