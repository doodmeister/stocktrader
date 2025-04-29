import streamlit as st
import pandas as pd
import logging
from pydantic import BaseSettings, ValidationError
from typing import Dict, Any
from datetime import datetime

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Settings(BaseSettings):
    etrade_api_key: str
    etrade_api_secret: str
    etrade_account_id: str
    default_refresh_secs: int = 60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@st.cache_resource
def get_settings() -> Settings:
    """Load settings from environment or .env file."""
    try:
        settings = Settings()
        return settings
    except ValidationError as e:
        logger.exception("Settings validation error")
        st.error(f"Configuration error: {e}")
        st.stop()

@st.cache_resource
def init_etrade_client(settings: Settings) -> Any:
    """Initialize and return the E*Trade API client."""
    try:
        # Replace with actual ETrade client initialization
        from etrade_api import ETradeClient
        client = ETradeClient(
            api_key=settings.etrade_api_key,
            api_secret=settings.etrade_api_secret,
            account_id=settings.etrade_account_id
        )
        return client
    except Exception as e:
        logger.exception("Failed to initialize E*Trade client")
        st.error(f"API client init error: {e}")
        st.stop()


def validate_live_data(data: pd.DataFrame) -> None:
    """Ensure the data has all required columns."""
    required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"Missing fields from API response: {', '.join(missing)}")


def main():
    st.title("Live Trading Dashboard")
    settings = get_settings()
    client = init_etrade_client(settings)

    st.sidebar.header("Settings")
    symbol = st.sidebar.text_input("Ticker Symbol", value="AAPL").strip().upper()
    refresh_secs = st.sidebar.number_input(
        "Refresh Interval (seconds)",
        min_value=10,
        value=settings.default_refresh_secs
    )
    if not symbol:
        st.error("Please enter a valid ticker symbol.")
        st.stop()

    placeholder = st.empty()
    while True:
        try:
            logger.info("Fetching live data for %s", symbol)
            raw = client.get_live_data(symbol)
            df = pd.DataFrame(raw)
            validate_live_data(df)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            with placeholder.container():
                st.subheader(f"Live Price Chart: {symbol}")
                st.line_chart(df[['open', 'high', 'low', 'close']])
                st.bar_chart(df['volume'])
                st.markdown(f"**Last Updated:** {datetime.utcnow().isoformat()} UTC")

        except Exception as e:
            logger.exception("Live data fetch error for %s", symbol)
            st.error(f"Failed to fetch or display data: {e}")

        st.experimental_rerun() if st.button("Refresh Now") else None
        st.time.sleep(refresh_secs)

if __name__ == "__main__":
    main()
