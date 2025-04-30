import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from utils.ml_pipeline import MLPipeline
from config import MLConfig
from utils.etrade_candlestick_bot import ETradeClient, PatternNN

@pytest.fixture
def mock_client():
    client = Mock(spec=ETradeClient)
    df_mock = pd.DataFrame({
        'open': np.random.rand(100),
        'high': np.random.rand(100),
        'low': np.random.rand(100),
        'close': np.random.rand(100),
        'volume': np.random.rand(100)
    })
    client.get_candles.return_value = df_mock
    return client

@pytest.fixture
def config():
    return MLConfig(
        seq_len=5,
        epochs=2,
        batch_size=16,
        device="cpu"
    )

def test_prepare_dataset(mock_client, config):
    pipeline = MLPipeline(mock_client, config)
    X_train, X_val, y_train, y_val = pipeline.prepare_dataset()
    
    assert isinstance(X_train, torch.Tensor)
    assert len(X_train.shape) == 3
    assert X_train.shape[1] == config.seq_len
    assert X_train.shape[2] == 5  # OHLCV features

def test_train_and_evaluate(mock_client, config):
    pipeline = MLPipeline(mock_client, config)
    model = PatternNN()
    
    metrics = pipeline.train_and_evaluate(model)
    
    assert 'accuracy' in metrics
    assert 'confusion_matrix' in metrics
    assert 0 <= metrics['accuracy'] <= 1