from pydantic import BaseSettings

class Settings(BaseSettings):
    REQUIRED_COLUMNS: list = ['open', 'high', 'low', 'close', 'volume', 'target']
    MAX_FILE_SIZE_MB: int = 100
    MODELS_DIR: str = "models"
    MIN_SAMPLES: int = 1000
    
    class Config:
        env_file = ".env"

def get_settings():
    return Settings()