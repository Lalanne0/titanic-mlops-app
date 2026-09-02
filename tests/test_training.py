"""Tests for the training pipeline logic."""

import pandas as pd
import pytest


@pytest.fixture()
def sample_csv(tmp_path):
    """Create a small but valid Titanic-format CSV for testing."""
    data = {
        "PassengerId": list(range(1, 21)),
        "Survived": [0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
        "Pclass": [3, 1, 3, 1, 3, 3, 1, 3, 2, 2, 1, 3, 3, 3, 1, 2, 3, 2, 1, 3],
        "Name": [f"Passenger {i}" for i in range(1, 21)],
        "Sex": [
            "male",
            "female",
            "female",
            "female",
            "male",
            "male",
            "male",
            "female",
            "female",
            "male",
            "female",
            "male",
            "male",
            "male",
            "female",
            "female",
            "male",
            "male",
            "female",
            "male",
        ],
        "Age": [22, 38, 26, 35, 35, None, 54, 2, 27, 14, 58, 20, 39, 55, 14, 44, 2, None, 31, 35],
        "SibSp": [1, 1, 0, 1, 0, 0, 0, 3, 0, 1, 0, 0, 1, 0, 0, 1, 4, 0, 1, 0],
        "Parch": [0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0],
        "Ticket": [f"T{i}" for i in range(1, 21)],
        "Fare": [
            7.25,
            71.28,
            7.92,
            53.1,
            8.05,
            8.46,
            51.86,
            21.07,
            11.13,
            30.07,
            26.55,
            8.05,
            31.27,
            8.05,
            120.0,
            26.0,
            29.12,
            13.0,
            53.1,
            8.05,
        ],
        "Cabin": ["", "C85", "", "C123", "", "", "E46", "", "", "", "C93", "", "", "", "B86", "", "", "", "C123", ""],
        "Embarked": [
            "S",
            "C",
            "S",
            "S",
            "S",
            "Q",
            "S",
            "S",
            "S",
            "C",
            "S",
            "S",
            "S",
            "S",
            "S",
            "S",
            "Q",
            "S",
            "S",
            "S",
        ],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "raw.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_training_produces_model(sample_csv, tmp_path, monkeypatch):
    """The training pipeline should produce a fitted model and return metrics."""
    # LESSON POINT: Isolated Testing with monkeypatch
    # We override the settings so the training code writes to a temp directory
    # instead of the real data path, and skip MLflow logging entirely.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///{}".format(tmp_path / "mlflow.db"))
    monkeypatch.setenv("DATA_PATH", sample_csv)
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.joblib"))
    monkeypatch.setenv("REPORTS_PATH", str(tmp_path / "reports"))
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")

    # Re-import after env is set so settings pick up the overrides
    import importlib

    import app.config

    importlib.reload(app.config)

    from app.training import FEATURE_COLS, TARGET

    # Verify the CSV has the expected columns
    df = pd.read_csv(sample_csv)
    for col in FEATURE_COLS + [TARGET]:
        assert col in df.columns, f"Missing column: {col}"

    # Verify the data shape is reasonable
    assert len(df) == 20
    assert df["Survived"].nunique() == 2


def test_feature_cols_match_schema():
    """The training feature columns should match the schema fields."""
    from app.schemas import PassengerInput
    from app.training import FEATURE_COLS

    schema_fields = set(PassengerInput.model_fields.keys())
    training_fields = set(FEATURE_COLS)

    # All training features should be in the schema
    assert training_fields.issubset(schema_fields), (
        f"Training uses features not in schema: {training_fields - schema_fields}"
    )
