"""Dashboard configuration and settings."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# Try the new pydantic-settings package first…
try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older Pydantic versions
    from pydantic import BaseSettings

# Field is still in pydantic
from pydantic import Field


@dataclass
class DashboardConfig:
    """Configuration settings for the stock trading dashboard."""
    DEFAULT_SYMBOLS: str = "AAPL,MSFT"
    VALID_INTERVALS: Tuple[str, ...] = ("1d", "1h", "30m", "15m", "5m", "1m")
    MAX_INTRADAY_DAYS: int = 60
    REFRESH_INTERVAL: int = 300
    DATA_DIR: Path = Path("data")
    MODEL_DIR: Path = Path("models")
    CACHE_TTL: int = 3600
    LOG_FILE: str = "dashboard.log"
    LOG_MAX_SIZE: int = 1024 * 1024  # 1MB
    LOG_BACKUP_COUNT: int = 5