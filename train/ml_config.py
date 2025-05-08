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
    # E-Trade credentials
    ETRADE_CONSUMER_KEY:       str
    ETRADE_CONSUMER_SECRET:    str
    ETRADE_OAUTH_TOKEN:        str
    ETRADE_OAUTH_TOKEN_SECRET: str
    ETRADE_ACCOUNT_ID:         str
    ETRADE_USE_SANDBOX:        bool = True

    # SMTP settings
    smtp_port: str = Field(default="587")
    
    # Trading parameters
    max_positions: str = Field(default="5")
    max_loss_percent: str = Field(default="0.02") 
    profit_target_percent: str = Field(default="0.03")
    max_daily_loss: str = Field(default="0.05")

    # ML parameters
    seq_len: int = Field(default=10)
    epochs: int = Field(default=10)
    batch_size: int = Field(default=32)
    learning_rate: float = Field(default=0.001)
    test_size: float = Field(default=0.2)
    random_state: int = Field(default=42)
    device: str = Field(default="cpu")
    model_dir: Path = Field(default=Path("models"))
    symbols: List[str]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"