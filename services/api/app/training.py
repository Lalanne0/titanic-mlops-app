"""Training pipeline: preprocessing, model fitting, MLflow logging & registration."""

import logging
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import settings

logger = logging.getLogger(__name__)

# --- Feature definitions ---
NUMERIC_FEATURES = ["Age", "Fare", "SibSp", "Parch", "Pclass"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Survived"


# LESSON POINT: Data Preprocessing Pipelines
# Using a scikit-learn Pipeline and ColumnTransformer ensures that preprocessing steps
# (imputation, scaling, encoding) are grouped into a single asset. Applying them together
# avoids "data leakage" (accidentally using information from the test set during training)
# and guarantees that incoming live prediction requests are preprocessed exactly the same way.
def create_preprocessing_pipeline() -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for Titanic features."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def train_model(data_path: str | None = None) -> dict:
    """
    Full training pipeline:
      1. Load & split data
      2. Fit preprocessing + LogisticRegression
      3. Evaluate metrics
      4. Log everything to MLflow and register the model
      5. Save a local joblib fallback
      6. Save reference data for monitoring

    Returns a dict with metrics, model version, and run ID.
    """
    if data_path is None:
        data_path = settings.DATA_PATH

    # --- Load data ---
    df = pd.read_csv(data_path)
    logger.info("Loaded dataset: %d rows", len(df))

    X = df[FEATURE_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- Build pipeline ---
    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessing_pipeline()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    # --- MLflow tracking ---
    # LESSON POINT: MLflow Experiment Tracking and Registry
    # MLflow tracks experiment runs, parameters, and metrics. Logging these helps developers
    # compare model performance over time. The `log_model` call automatically registers the
    # fitted pipeline in the MLflow Model Registry, assigning it a unique version.
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("titanic-survival")

    with mlflow.start_run(run_name="titanic-training") as run:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
        }

        # Log parameters
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "max_iter": 1000,
                "test_size": 0.2,
                "n_features": len(FEATURE_COLS),
                "n_samples": len(df),
                "numeric_features": ",".join(NUMERIC_FEATURES),
                "categorical_features": ",".join(CATEGORICAL_FEATURES),
            }
        )

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log & register model
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=settings.MODEL_NAME,
            skops_trusted_types=["numpy.dtype"],
        )

        run_id = run.info.run_id
        logger.info("MLflow run %s - accuracy=%.4f", run_id, metrics["accuracy"])

    # --- Local fallback ---
    os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, settings.MODEL_PATH)

    # --- Save reference data for Evidently ---
    # LESSON POINT: Reference Data for Drift Detection
    # To detect if the live prediction data distribution is drifting over time, monitoring tools
    # need a baseline to compare against. Saving the training data split as `reference.csv` provides
    # that baseline distribution for statistical tests.
    reference = X_train.copy()
    reference[TARGET] = y_train.values
    reference_path = os.path.join(os.path.dirname(settings.DATA_PATH), "reference.csv")
    reference.to_csv(reference_path, index=False)

    # --- Get latest registered version ---
    client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
    versions = client.search_model_versions(f"name='{settings.MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))

    return {
        "status": "success",
        **metrics,
        "model_version": str(latest.version),
        "run_id": run_id,
        "n_samples": len(df),
    }
