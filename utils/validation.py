"""Validation Utilities

Helper functions to validate and sanitize user input.
"""

import re
from train.trainer import TrainingConfig
from train.config import TrainingConfig

def sanitize_input(text: str) -> str:
    """Sanitize stock symbol input (basic alphanumeric)."""
    sanitized = re.sub(r'[^A-Za-z0-9\.\-]', '', text)
    return sanitized.upper()

def validate_training_params(cfg: TrainingConfig) -> None:
    if cfg.epochs <= 0: raise ValueError("epochs must be > 0")
    if not (0 < cfg.validation_split < 1): raise ValueError("validation_split must be in (0,1)")
    # …any other checks…
