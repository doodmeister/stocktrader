"""Dashboard configuration and settings."""

# filepath: c:\dev\stocktrader\data\config.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

@dataclass
class DashboardConfig:
    """Configuration settings for the stock trading dashboard."""
    DEFAULT_SYMBOLS: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL"])
    VALID_INTERVALS: List[str] = field(default_factory=lambda: ["1d", "1h", "15m", "5m", "1m"])
    MAX_INTRADAY_DAYS: int = 60
    REFRESH_INTERVAL: int = 300
    DATA_DIR: Path = Path("data")
    MODEL_DIR: Path = Path("models")
    CACHE_TTL: int = 3600
    LOG_FILE: str = str(Path("logs") / "dashboard.log")  # Updated to use logs folder
    LOG_MAX_SIZE: int = 1024 * 1024  # 1MB
    LOG_BACKUP_COUNT: int = 5