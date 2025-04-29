import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

from pages.model_training import (
    validate_training_data, 
    MAX_FILE_SIZE_MB,
    REQUIRED_COLUMNS,
    get_model_manager,
    TrainingConfig
)

@pytest.fixture
def valid_training_data():
    """Create valid training dataset fixture."""
    np.random.seed(42)  # For reproducibility
    return pd.DataFrame({
        'open': np.random.random(1000),
        'high': np.random.random(1000),
        'low': np.random.random(1000),
        'close': np.random.random(1000),
        'volume': np.random.randint(1000, 10000, 1000),
        'target': np.random.randint(0, 3, 1000)
    })

@pytest.fixture
def mock_model_manager():
    """Create a mock ModelManager instance."""
    manager = Mock()
    manager.save_model.return_value = Path("models/test_model.h5")
    return manager

class TestDataValidation:
    def test_validate_training_data_valid(self, valid_training_data):
        is_valid, error_msg = validate_training_data(valid_training_data)
        assert is_valid
        assert error_msg is None

    def test_validate_training_data_missing_columns(self):
        df = pd.DataFrame({'open': [1, 2], 'close': [1, 2]})
        is_valid, error_msg = validate_training_data(df)
        assert not is_valid
        assert "Missing required columns" in error_msg
        
    def test_validate_training_data_null_values(self, valid_training_data):
        df = valid_training_data.copy()
        df.loc[0, 'close'] = None
        is_valid, error_msg = validate_training_data(df)
        assert not is_valid
        assert "contains null values" in error_msg

    def test_validate_training_data_wrong_types(self, valid_training_data):
        df = valid_training_data.copy()
        df['volume'] = df['volume'].astype(str)
        is_valid, error_msg = validate_training_data(df)
        assert not is_valid
        assert "must be numeric" in error_msg

    def test_validate_training_data_too_small(self):
        df = pd.DataFrame({
            col: [1] * 100 for col in REQUIRED_COLUMNS
        })
        is_valid, error_msg = validate_training_data(df)
        assert not is_valid
        assert "too small" in error_msg

    def test_validate_training_data_empty_dataframe(self):
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        is_valid, error_msg = validate_training_data(df)
        assert not is_valid
        assert "too small" in error_msg

    def test_validate_training_data_negative_values(self, valid_training_data):
        df = valid_training_data.copy()
        df.loc[0, 'volume'] = -100
        is_valid, error_msg = validate_training_data(df)
        assert not is_valid
        assert "invalid values" in error_msg

class TestModelManager:
    @patch('pages.model_training.ModelManager')
    def test_get_model_manager_caching(self, mock_manager_class):
        """Test that get_model_manager properly caches the instance."""
        manager1 = get_model_manager()
        manager2 = get_model_manager()
        
        mock_manager_class.assert_called_once()
        assert manager1 is manager2

    def test_model_saving(self, mock_model_manager, valid_training_data):
        """Test model saving functionality with metadata."""
        model = Mock()
        metrics = {'accuracy': 0.95, 'loss': 0.1}
        
        with patch('pages.model_training.get_model_manager', return_value=mock_model_manager):
            save_path = mock_model_manager.save_model(
                model,
                metadata={
                    'version': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'metrics': metrics
                }
            )
            
            assert save_path == Path("models/test_model.h5")
            mock_model_manager.save_model.assert_called_once()

def test_file_size_validation():
    """Test the file size validation constant."""
    assert MAX_FILE_SIZE_MB == 100
    assert isinstance(MAX_FILE_SIZE_MB, int)

def test_training_config_validation():
    valid_config = TrainingConfig(epochs=10, batch_size=32, learning_rate=0.001)
    is_valid, _ = valid_config.validate()
    assert is_valid

def test_data_validation():
    # Add test cases for different validation scenarios
    pass

if __name__ == '__main__':
    pytest.main(['-v'])