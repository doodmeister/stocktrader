# utils/preprocess_input.py

import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Dict

def load_preprocessing_config(config_path: Union[str, Path]) -> Dict:
    """
    Load saved preprocessing configuration.
    """
    with open(config_path, 'r') as f:
        return json.load(f)

def preprocess_input(
    df: pd.DataFrame,
    preprocessing_config: Dict,
    seq_len: int
) -> torch.Tensor:
    """
    Preprocess live input data for model inference.

    Args:
        df: DataFrame with raw OHLCV + candlestick pattern columns.
        preprocessing_config: dict with 'feature_order' and 'normalization'
        seq_len: number of time steps expected by model

    Returns:
        torch.Tensor of shape (1, seq_len, input_size)
    """
    feature_order = preprocessing_config["feature_order"]
    min_vals = np.array(preprocessing_config["normalization"]["min"])
    max_vals = np.array(preprocessing_config["normalization"]["max"])

    if not all(col in df.columns for col in feature_order):
        missing = [c for c in feature_order if c not in df.columns]
        raise ValueError(f"Missing required features: {missing}")

    df = df[feature_order].copy()

    # Normalize using training min/max
    data = df.values
    normed = (data - min_vals) / (max_vals - min_vals + 1e-8)

    # Extract most recent sequence
    if len(normed) < seq_len:
        raise ValueError(f"Not enough data: need {seq_len} rows, got {len(normed)}")

    seq = normed[-seq_len:]
    tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # shape: (1, seq_len, input_size)
    return tensor
