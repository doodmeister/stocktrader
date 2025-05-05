import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from train.ml_training_pipeline import (
    validate_dataframe,
    feature_engineering,
    train_model,
    save_model,
    REQUIRED_COLUMNS,
    FEATURES,
    MIN_SAMPLES,
)

@pytest.fixture
def valid_ohlcv_df(tmp_path):
    """Create a valid OHLCV DataFrame for testing."""
    n = max(MIN_SAMPLES + 10, 120)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "Open": np.random.uniform(100, 200, n),
        "High": np.random.uniform(100, 200, n),
        "Low": np.random.uniform(90, 199, n),
        "Close": np.random.uniform(100, 200, n),
        "Volume": np.random.randint(1000, 10000, n),
        "date": dates,
    })
    return df

def test_validate_dataframe_valid(valid_ohlcv_df):
    # Should not raise
    validate_dataframe(valid_ohlcv_df)

def test_validate_dataframe_missing_column(valid_ohlcv_df):
    df = valid_ohlcv_df.drop(columns=["Open"])
    with pytest.raises(ValueError):
        validate_dataframe(df)

def test_validate_dataframe_nulls(valid_ohlcv_df):
    df = valid_ohlcv_df.copy()
    df.loc[0, "Close"] = None
    with pytest.raises(ValueError):
        validate_dataframe(df)

def test_validate_dataframe_wrong_type(valid_ohlcv_df):
    df = valid_ohlcv_df.copy()
    df["Volume"] = df["Volume"].astype(str)
    with pytest.raises(TypeError):
        validate_dataframe(df)

def test_feature_engineering_output(valid_ohlcv_df):
    df_feat = feature_engineering(valid_ohlcv_df)
    # Should have all feature columns and no nulls
    for col in FEATURES + ["target"]:
        assert col in df_feat.columns
    assert not df_feat.isnull().any().any()
    assert set(df_feat["target"].unique()).issubset({0, 1})

def test_train_model_and_save(tmp_path, valid_ohlcv_df):
    df_feat = feature_engineering(valid_ohlcv_df)
    model = train_model(df_feat)
    assert hasattr(model, "predict")
    # Save model and check file exists
    path = save_model(model, "AAPL", "1d")
    assert Path(path).exists()
    # Clean up
    Path(path).unlink()