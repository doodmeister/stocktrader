# inference/live_inference.py

import torch
from pathlib import Path
from utils.preprocess_input import load_preprocessing_config, preprocess_input
from train.model_manager import load_latest_model
from patterns.patterns_nn import PatternNN

ACTION_MAP = {0: "hold", 1: "buy", 2: "sell"}

def make_trade_decision(
    df,
    preprocessing_path: str,
    model_dir: str = "models/",
    seq_len: int = 10
) -> str:
    """
    Predict trading action from live OHLCV + pattern-enhanced DataFrame.

    Args:
        df: pandas.DataFrame with OHLCV and candlestick pattern features
        preprocessing_path: Path to saved preprocessing JSON
        model_dir: Directory where trained models are stored
        seq_len: Number of time steps the model expects

    Returns:
        A string: 'buy', 'sell', or 'hold'
    """
    # Load preprocessing configuration
    preprocessing = load_preprocessing_config(preprocessing_path)

    # Preprocess input
    input_tensor = preprocess_input(df, preprocessing, seq_len=seq_len)

    # Load model
    model, metadata = load_latest_model(PatternNN, base_directory=model_dir)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        predicted = torch.argmax(output, dim=1).item()

    return ACTION_MAP.get(predicted, "hold")
