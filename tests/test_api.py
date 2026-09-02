"""Tests for the FastAPI application endpoints using TestClient."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Create a TestClient with isolated config pointing to temp paths."""
    import pandas as pd

    # Create a small test dataset
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
        "Age": [22, 38, 26, 35, 35, 30, 54, 2, 27, 14, 58, 20, 39, 55, 14, 44, 2, 30, 31, 35],
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
    csv_path = str(tmp_path / "raw.csv")
    pd.DataFrame(data).to_csv(csv_path, index=False)

    # Also create the "original" dataset for reset-data
    pd.DataFrame(data).to_csv(str(tmp_path / "raw_original.csv"), index=False)

    monkeypatch.setenv("DATA_PATH", csv_path)
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.joblib"))
    monkeypatch.setenv("REPORTS_PATH", str(tmp_path / "reports"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("MODEL_NAME", "test-model")

    # Reload config to pick up test env
    import importlib

    import app.config

    importlib.reload(app.config)

    # Import the app after config is set
    from app.main import app

    return TestClient(app)


class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_reports_model_status(self, client):
        response = client.get("/health")
        data = response.json()
        assert "model_loaded" in data


class TestDatasetInfo:
    def test_dataset_info_returns_stats(self, client):
        response = client.get("/dataset-info")
        assert response.status_code == 200
        data = response.json()
        assert "total_rows" in data
        assert "survival_rate" in data
        assert data["total_rows"] == 20

    def test_dataset_info_has_distributions(self, client):
        response = client.get("/dataset-info")
        data = response.json()
        assert "sex_distribution" in data
        assert "class_distribution" in data
        assert "age_stats" in data


class TestModelInfo:
    def test_model_info_when_no_model(self, client):
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "no_model_loaded"


class TestPredict:
    def test_predict_without_model_returns_503(self, client):
        response = client.post(
            "/predict",
            json={
                "Pclass": 1,
                "Sex": "female",
                "Age": 29,
                "SibSp": 0,
                "Parch": 0,
                "Fare": 100.0,
                "Embarked": "S",
            },
        )
        assert response.status_code == 503

    def test_predict_with_invalid_input_returns_422(self, client):
        response = client.post(
            "/predict",
            json={
                "Pclass": 5,
                "Sex": "invalid",
                "Fare": -10,
            },
        )
        assert response.status_code == 422
