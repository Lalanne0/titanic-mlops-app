"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings


# LESSON POINT: Environment-Based Configuration
# Using Pydantic Settings allows us to define configuration schemas that automatically load
# values from system environment variables (or a local .env file). It parses types and ensures
# that critical configs (like the MLflow URI) are present, providing defaults for easier local dev.
class Settings(BaseSettings):
    # MLflow
    MLFLOW_TRACKING_URI: str = "http://mlflow:5000"
    MLFLOW_S3_ENDPOINT_URL: str = "http://minio:9000"
    AWS_ACCESS_KEY_ID: str = "minioadmin"
    AWS_SECRET_ACCESS_KEY: str = "minioadmin_secret_1234"

    # Model
    MODEL_NAME: str = "titanic-survivor"
    DATA_PATH: str = "/app/data/raw.csv"
    MODEL_PATH: str = "/app/data/model.joblib"
    REPORTS_PATH: str = "/app/reports"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
