from typing import List
from functools import lru_cache
from pydantic import Field, field_validator

# Handle BaseSettings for both Pydantic v1 and v2
try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings


class Settings(BaseSettings):
    # E*TRADE API
    etrade_consumer_key: str = Field(..., env="ETRADE_CONSUMER_KEY")
    etrade_consumer_secret: str = Field(..., env="ETRADE_CONSUMER_SECRET")
    etrade_oauth_token: str = Field(..., env="ETRADE_OAUTH_TOKEN")
    etrade_oauth_token_secret: str = Field(..., env="ETRADE_OAUTH_TOKEN_SECRET")
    etrade_account_id: str = Field(..., env="ETRADE_ACCOUNT_ID")
    etrade_use_sandbox: bool = Field(True, env="ETRADE_USE_SANDBOX")

    # Notifications
    smtp_server: str = Field(default="", env="SMTP_SERVER")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    smtp_user: str = Field(default="", env="SMTP_USER")
    smtp_pass: str = Field(default="", env="SMTP_PASS")

    twilio_account_sid: str = Field(default="", env="TWILIO_ACCOUNT_SID")
    twilio_auth_token: str = Field(default="", env="TWILIO_AUTH_TOKEN")
    twilio_from_number: str = Field(default="", env="TWILIO_FROM_NUMBER")

    slack_webhook_url: str = Field(default="", env="SLACK_WEBHOOK_URL")

    # Trading Configuration
    max_positions: int = Field(5, env="MAX_POSITIONS")
    max_loss_percent: float = Field(0.02, env="MAX_LOSS_PERCENT")
    profit_target_percent: float = Field(0.03, env="PROFIT_TARGET_PERCENT")
    max_daily_loss: float = Field(0.05, env="MAX_DAILY_LOSS")
    
    # Use str type with default and validate later
    symbols_str: str = Field(default="AAPL", env="SYMBOLS")
    
    # Project paths and limits
    required_columns: List[str] = Field(default=["open", "high", "low", "close", "volume", "target"])
    max_file_size_mb: int = Field(100)
    models_dir: str = Field("models")
    min_samples: int = Field(1000)

    # Add a validator to convert symbols_str to symbols list
    @field_validator("symbols_str")
    def parse_symbols(cls, v):
        if not v:
            return ["AAPL"]  # Default if empty
        return [s.strip() for s in v.split(",") if s.strip()]
    
    @property
    def symbols(self) -> List[str]:
        """Access symbols as a property"""
        return self.parse_symbols(self.symbols_str)

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
