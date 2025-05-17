# ml_config.py
from pathlib import Path
from typing import List
try:
    from pydantic_settings import BaseSettings
except ImportError:
    raise ImportError(
        "You need to `pip install pydantic-settings` to use MLConfig on Pydantic v2"
    )

from pydantic import Field

class MLConfig(BaseSettings):
    # --- Platform/Integration Settings (not used in classic ML training) ---
    ETRADE_CONSUMER_KEY:       str
    ETRADE_CONSUMER_SECRET:    str
    ETRADE_OAUTH_TOKEN:        str
    ETRADE_OAUTH_TOKEN_SECRET: str
    ETRADE_ACCOUNT_ID:         str
    ETRADE_USE_SANDBOX:        bool = True

    smtp_port: str = Field(default="587")
    
    max_positions: str = Field(default="5")
    max_loss_percent: str = Field(default="0.02") 
    profit_target_percent: str = Field(default="0.03")
    max_daily_loss: str = Field(default="0.05")

    # --- Deep Learning Parameters (not used in classic ML training) ---
    seq_len: int = Field(default=10)
    epochs: int = Field(default=10)
    batch_size: int = Field(default=32)
    learning_rate: float = Field(default=0.001)
    device: str = Field(default="cpu")

    # --- Shared/Data/General Parameters ---
    test_size: float = Field(default=0.2)
    random_state: int = Field(default=42)
    model_dir: Path = Field(default=Path("models"))
    symbols: List[str] = Field(default_factory=lambda: ["AAPL"])

    # --- Classic ML Parameters (use these for classic ML training) ---
    n_estimators: int = Field(default=100)
    max_depth: int = Field(default=10)
    min_samples_split: int = Field(default=2)
    cv_folds: int = Field(default=3)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"