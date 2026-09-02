"""
Titanic Survival Prediction API
====================================
FastAPI service with MLflow tracking, Evidently monitoring,
user feedback loop, and data-drift simulation.
"""

import asyncio
import logging
import os
import shutil
from contextlib import asynccontextmanager

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .chat import ChatRequest, ChatResponse, handle_chat
from .config import settings
from .drift import simulate_drift
from .llm_traces import trace_store
from .model import ModelManager
from .monitoring import generate_drift_report
from .schemas import FeedbackInput, PassengerInput, PredictionResponse, RetrainResponse
from .training import train_model

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# --- Global model manager ---
# LESSON POINT: Singleton Pattern for Model Management
# We instantiate a single ModelManager globally. This class encapsulates the state
# of the loaded model (such as model version, runs, and underlying scikit-learn pipeline).
# By doing this, we keep the model in memory across different requests without reloading it
# from the disk or model registry every time /predict is called.
model_manager = ModelManager()


# =======================================================
#  Startup / shutdown lifecycle
# =======================================================
async def _wait_for_mlflow(max_retries: int = 30, delay: float = 2.0) -> None:
    """Block until the MLflow tracking server is reachable."""
    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{settings.MLFLOW_TRACKING_URI}/health", timeout=5)
                if resp.status_code == 200:
                    logger.info("MLflow tracking server is ready")
                    return
        except Exception:
            pass
        logger.info("Waiting for MLflow tracking server... (%d/%d)", attempt, max_retries)
        await asyncio.sleep(delay)
    raise RuntimeError("MLflow did not become available in time")


def _ensure_data_initialized() -> None:
    """Copy the bundled dataset to the Docker volume on first boot."""
    if not os.path.exists(settings.DATA_PATH):
        os.makedirs(os.path.dirname(settings.DATA_PATH), exist_ok=True)
        shutil.copy("/app/raw_original.csv", settings.DATA_PATH)
        logger.info("Initialized working dataset from bundled raw_original.csv")


# LESSON POINT: Lifespan Event Handler in FastAPI
# FastAPI's lifespan context manager manages the application startup and shutdown phases.
# In MLOps pipelines, startup is the ideal time to run initial sanity checks, verify the
# availability of external systems (like MLflow), prepare directories, and load the
# machine learning model. Running this logic before the app yields prevents incoming requests
# from hitting an uninitialized or broken API.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown logic."""
    logger.info("Starting Titanic MLOps API...")

    # 1) Ensure dataset is available in the volume
    _ensure_data_initialized()

    # 2) Wait for MLflow
    await _wait_for_mlflow()

    # 3) Load model - auto-train if nothing is registered yet
    try:
        model_manager.load_model()
        logger.info("Model loaded successfully (version %s)", model_manager.model_version)
    except Exception:
        logger.info("No registered model found. Running initial training pipeline...")
        try:
            result = await asyncio.to_thread(train_model)
            logger.info("Initial model training completed. Accuracy: %.4f", result["accuracy"])
            model_manager.load_model()
        except Exception as exc:
            logger.warning("Could not auto-train model: %s", exc)

    yield  # --- app is running ---

    logger.info("Shutting down Titanic MLOps API")


# =======================================================
#  FastAPI app
# =======================================================
app = FastAPI(
    title="Would You Survive the Titanic?",
    description=(
        "An MLOps-powered prediction service.\n\n"
        "- Predict survival based on passenger features\n"
        "- Submit feedback when the prediction is wrong\n"
        "- Retrain the model on demand\n"
        "- Simulate data drift and observe its effects\n"
    ),
    version="1.0.0",
    root_path="/api",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =======================================================
#  Endpoints
# =======================================================


# --- Health ---
@app.get("/health", tags=["ops"])
def health():
    """Liveness probe - always returns 200 if the process is up."""
    return {
        "status": "healthy",
        "model_loaded": model_manager.is_loaded,
        "model_version": model_manager.model_version,
    }


# --- Predict ---
# LESSON POINT: API Inference Request Handling
# In FastAPI, we use Pydantic schemas (PassengerInput) to define the expected structure
# and validation rules of incoming JSON requests. FastAPI automatically validates the input
# (e.g. checks that passenger class is between 1 and 3, and sex is male or female).
# Once validated, we pass the data to the ModelManager, which returns the prediction results.
@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(passenger: PassengerInput):
    """Predict whether a passenger would survive the Titanic."""
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet - call POST /retrain first.")

    survived, probability = model_manager.predict(passenger)

    return PredictionResponse(
        survived=survived,
        probability=round(probability, 4),
        model_version=model_manager.model_version or "unknown",
    )


# --- Feedback ("Got it wrong?") ---
# LESSON POINT: Implementing a User Feedback Loop
# The /feedback endpoint allows users to report incorrect predictions. By collecting
# corrected labels, we append new rows to our working CSV dataset. In a real MLOps
# pipeline, this new data is essential to monitor performance degradation, and it serves
# as the training set for future model iterations.
@app.post("/feedback", tags=["feedback"])
def submit_feedback(feedback: FeedbackInput):
    """Record user feedback - appends a corrected row to the dataset.

    Use this when the model prediction was wrong:
    'Got it wrong? Let us know!'
    """
    p = feedback.passenger_data
    df = pd.read_csv(settings.DATA_PATH)
    next_id = int(df["PassengerId"].max()) + 1

    new_row = pd.DataFrame(
        [
            {
                "PassengerId": next_id,
                "Survived": int(feedback.actual_survived),
                "Pclass": p.Pclass,
                "Name": f"Feedback, Passenger {next_id}",
                "Sex": p.Sex,
                "Age": p.Age,
                "SibSp": p.SibSp,
                "Parch": p.Parch,
                "Ticket": f"FEEDBACK{next_id}",
                "Fare": p.Fare,
                "Cabin": "",
                "Embarked": p.Embarked,
            }
        ]
    )
    new_row.to_csv(settings.DATA_PATH, mode="a", header=False, index=False)

    return {
        "status": "feedback_recorded",
        "message": "Thank you! Your feedback has been recorded. You can now trigger /retrain.",
        "new_dataset_size": len(df) + 1,
    }


# --- Retrain ---
# LESSON POINT: On-Demand Retraining
# Triggering model retraining runs the training pipeline on the updated dataset (which includes
# original data + user feedback). The new model is logged to MLflow, registered, and then the
# ModelManager dynamically hot-swaps the active model in-memory without restarting the server.
@app.post("/retrain", response_model=RetrainResponse, tags=["training"])
def retrain():
    """Retrain the model on the current dataset, log to MLflow, and reload."""
    result = train_model()
    model_manager.load_model()

    # Generate a drift report post-training
    try:
        report_name = generate_drift_report()
        result["drift_report"] = report_name
    except Exception as exc:
        result["drift_report"] = f"Could not generate report: {exc}"

    return RetrainResponse(**result)


# --- Drift simulation ---
# LESSON POINT: Data Drift Simulation and Monitoring
# Data drift occurs when the statistical properties of features change over time.
# To demonstrate how monitoring tools detect this, this endpoint injects synthetically modified
# passenger profiles (e.g. highly anomalous age/fare distributions and inverted survival rules)
# into the active dataset and generates a drift report. Retraining is intentionally left as a
# separate step so the user can first observe the drift in the Evidently dashboard before
# deciding to retrain.
@app.post("/simulate-drift", tags=["drift"])
def trigger_drift_simulation(
    n_samples: int = Query(100, ge=10, le=1000, description="Number of drifted samples to inject"),
):
    """Inject synthetic data with inverted survival patterns and generate a drift report.

    This endpoint:
    1. Generates *n_samples* passengers with inverted survival logic
    2. Appends them to the dataset
    3. Generates an Evidently drift report (comparing against the training baseline)

    Retraining is NOT triggered automatically so you can observe the drift first.
    Call POST /retrain afterwards to update the model.
    """
    drift_result = simulate_drift(n_samples=n_samples)

    report_name = None
    try:
        report_name = generate_drift_report()
    except Exception as exc:
        report_name = f"Could not generate report: {exc}"

    return {
        "drift_injection": drift_result,
        "drift_report": report_name,
    }


# --- Model info ---
@app.get("/model-info", tags=["ops"])
def model_info():
    """Return metadata about the currently loaded model."""
    if not model_manager.is_loaded:
        return {"status": "no_model_loaded"}
    return {
        "model_name": settings.MODEL_NAME,
        "model_version": model_manager.model_version,
        "model_run_id": model_manager.run_id,
        "mlflow_tracking_uri": settings.MLFLOW_TRACKING_URI,
    }


# --- Dataset info ---
@app.get("/dataset-info", tags=["ops"])
def dataset_info():
    """Return basic statistics about the current dataset."""
    try:
        df = pd.read_csv(settings.DATA_PATH)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "total_rows": len(df),
        "columns": list(df.columns),
        "survival_rate": round(float(df["Survived"].mean()), 4),
        "class_distribution": df["Pclass"].value_counts().to_dict(),
        "sex_distribution": df["Sex"].value_counts().to_dict(),
        "age_stats": {
            "mean": round(float(df["Age"].mean()), 1),
            "median": round(float(df["Age"].median()), 1),
            "missing": int(df["Age"].isna().sum()),
        },
    }


# --- Reset dataset ---
@app.post("/reset-data", tags=["ops"])
def reset_data():
    """Restore the working dataset to its original bundled state.

    This removes all user feedback and injected drift samples.
    The model remains loaded until you retrain.
    """
    original_path = "/app/raw_original.csv"
    if not os.path.exists(original_path):
        raise HTTPException(status_code=500, detail="Original dataset not found in container")

    shutil.copy(original_path, settings.DATA_PATH)

    # Remove reference.csv so it gets regenerated on next train
    reference_path = os.path.join(os.path.dirname(settings.DATA_PATH), "reference.csv")
    if os.path.exists(reference_path):
        os.remove(reference_path)

    df = pd.read_csv(settings.DATA_PATH)

    return {
        "status": "dataset_reset",
        "message": "Working dataset restored to original. Retrain the model to use the clean data.",
        "total_rows": len(df),
    }


# --- Chat (LLMOps layer) ---
# LESSON POINT: LLMOps vs MLOps Separation
# The /chat endpoint adds a language model interaction layer on top of the existing
# prediction API. The LLM calls predict_survival as a tool -- it never generates
# predictions itself. This keeps the deterministic sklearn model as the source of
# truth while the LLM handles natural-language understanding, grounding, and safety.
# Traces for LLM calls are stored separately from the sklearn metrics in MLflow.
@app.post("/chat", response_model=ChatResponse, tags=["copilot"])
async def chat(request: ChatRequest):
    """Send a natural-language message to the Titanic Decision Copilot.

    The assistant uses the predict_survival tool for any numerical predictions.
    When LLM_API_KEY is not set, falls back to mock mode.
    """
    return await handle_chat(request.message, model_manager)


@app.get("/traces", tags=["copilot"])
def list_traces():
    """Return recent LLM interaction traces (most recent first)."""
    return {"traces": trace_store.list_all()}
