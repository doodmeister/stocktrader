"""Validate .env and core configuration for E*Trade Candlestick Bot."""

from pathlib import Path

REQUIRED_ENV_VARS = [
    "ETRADE_CONSUMER_KEY",
    "ETRADE_CONSUMER_SECRET",
    "ETRADE_ACCOUNT_ID",
]

class DashboardConfig:
    # ... your config attributes and methods ...
    pass

def main():
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env file not found. Please create it from .env.example.")
        exit(1)

    # Load .env variables
    with env_path.open() as f:
        lines = f.readlines()
    env = {}
    for line in lines:
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.strip().split("=", 1)
            env[k.strip()] = v.strip()

    missing = [var for var in REQUIRED_ENV_VARS if not env.get(var)]
    if missing:
        print(f"❌ Missing required environment variables: {', '.join(missing)}")
        exit(1)

    print("✅ .env configuration looks OK.")

if __name__ == "__main__":
    main()