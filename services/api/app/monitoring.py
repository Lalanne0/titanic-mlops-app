"""Evidently AI monitoring - generate data drift & performance reports."""

import logging
import os
from datetime import datetime, timezone

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from .config import settings
from .training import FEATURE_COLS, TARGET

logger = logging.getLogger(__name__)

ANALYSIS_COLS = FEATURE_COLS + [TARGET]


# LESSON POINT: Data Drift and Covariate Shift Detection
# Data drift occurs when the input distribution of live production data differs significantly
# from the training data. Here, we compare the "reference" dataset (the clean baseline split we
# saved during model training) against the "current" dataset (the active CSV, which gathers new
# user inputs and feedback). Evidently AI runs statistical tests to determine if features
# (like age or fare distributions) have drifted, which helps prevent model performance decay.
def generate_drift_report() -> str:
    """Compare reference (training split) vs. current dataset.

    Returns the filename of the generated HTML report.
    Raises FileNotFoundError if no reference data is available.
    """
    reference_path = os.path.join(os.path.dirname(settings.DATA_PATH), "reference.csv")
    if not os.path.exists(reference_path):
        raise FileNotFoundError("No reference data found. Train the model first.")

    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(settings.DATA_PATH)

    # Keep only the columns we care about
    ref = reference_df[[c for c in ANALYSIS_COLS if c in reference_df.columns]].copy()
    cur = current_df[[c for c in ANALYSIS_COLS if c in current_df.columns]].copy()

    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=ref, current_data=cur)

    # Save to shared reports volume
    os.makedirs(settings.REPORTS_PATH, exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"drift_report_{timestamp}.html"
    filepath = os.path.join(settings.REPORTS_PATH, filename)
    snapshot.save_html(filepath)

    logger.info("Drift report saved: %s", filename)
    return filename
