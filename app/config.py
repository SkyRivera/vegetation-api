# app/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API
    API_VERSION: str = "1.1.0"
    API_TITLE: str = "Vegetation Classification API"
    API_DESCRIPTION: str = "API para clasificar vegetación usando Random Forest (4 categorías, 11 features)"

    # Servidor
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    WORKERS: int = 4

    # Imágenes
    MAX_IMAGE_SIZE_MB: int = 10
    ALLOWED_IMAGE_FORMATS: list = ["jpg", "jpeg", "png"]
    IMAGE_TARGET_SIZE: tuple = (400, 400)

    # Modelo
    MODEL_PATH: str = "models/vegetation_classifier_latest.joblib"
    MODEL_NAME: str = "vegetation_classifier"

    # PostgreSQL
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "vegetation_db"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgre5"

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
