"""Model loading from MLflow registry (with local joblib fallback) and prediction."""

import logging

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from .config import settings
from .schemas import PassengerInput
from .training import FEATURE_COLS

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages the lifecycle of the prediction model."""

    def __init__(self) -> None:
        self.model = None
        self.model_version: str | None = None
        self.run_id: str | None = None
        self.is_loaded: bool = False

    # --- Model Loading ---
    # LESSON POINT: Model Loading Decoupling and Fail-Safe Fallbacks
    # In a robust production setting, model training and model serving are decoupled.
    # The ModelManager contacts the centralized MLflow registry to download the active
    # model version. If the MLflow tracking server is offline or unreachable, the service
    # gracefully falls back to loading a local `model.joblib` file. This prevents downtime
    # during network issues or registry outages.
    def load_model(self) -> None:
        """Load the latest model from MLflow Model Registry.

        Falls back to a local joblib file if MLflow is unavailable or the
        registry is empty.
        """
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

        # 1) Try MLflow registry
        try:
            client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
            versions = client.search_model_versions(f"name='{settings.MODEL_NAME}'")
            if versions:
                latest = max(versions, key=lambda v: int(v.version))
                model_uri = f"models:/{settings.MODEL_NAME}/{latest.version}"
                self.model = mlflow.sklearn.load_model(model_uri)
                self.model_version = str(latest.version)
                self.run_id = latest.run_id
                self.is_loaded = True
                logger.info("Loaded model v%s from MLflow (run %s)", latest.version, latest.run_id)
                return
        except Exception as exc:
            logger.warning("Could not load from MLflow registry: %s", exc)

        # 2) Fallback to local file
        try:
            self.model = joblib.load(settings.MODEL_PATH)
            self.model_version = "local"
            self.run_id = None
            self.is_loaded = True
            logger.info("Loaded model from local file %s", settings.MODEL_PATH)
        except Exception as exc:
            logger.error("No model available: %s", exc)
            raise

    # --- Prediction ---
    # LESSON POINT: Consistent Feature Alignment
    # Before feeding prediction inputs into the model, the data must be formatted
    # exactly like the training dataset. We convert the Pydantic schema data into a
    # Pandas DataFrame and enforce the exact column names and order (`FEATURE_COLS`).
    # Any misalignment here would result in scikit-learn throwing value errors.
    def predict(self, passenger: PassengerInput) -> tuple[bool, float]:
        """Return (survived: bool, probability: float) for a single passenger."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        input_df = pd.DataFrame(
            [
                {
                    "Pclass": passenger.Pclass,
                    "Sex": passenger.Sex,
                    "Age": passenger.Age,
                    "SibSp": passenger.SibSp,
                    "Parch": passenger.Parch,
                    "Fare": passenger.Fare,
                    "Embarked": passenger.Embarked,
                }
            ],
            columns=FEATURE_COLS,  # keep column order consistent
        )

        prediction = self.model.predict(input_df)[0]
        probabilities = self.model.predict_proba(input_df)[0]

        survived = bool(prediction)
        survival_probability = float(probabilities[1])

        return survived, survival_probability
