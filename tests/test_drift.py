"""Tests for the drift simulation module."""

import pandas as pd
import pytest

from app.drift import simulate_drift


@pytest.fixture()
def drift_csv(tmp_path):
    """Create a minimal CSV that drift injection can append to."""
    data = {
        "PassengerId": [1, 2, 3],
        "Survived": [0, 1, 1],
        "Pclass": [3, 1, 2],
        "Name": ["A", "B", "C"],
        "Sex": ["male", "female", "male"],
        "Age": [22.0, 38.0, 26.0],
        "SibSp": [1, 1, 0],
        "Parch": [0, 0, 0],
        "Ticket": ["T1", "T2", "T3"],
        "Fare": [7.25, 71.28, 7.92],
        "Cabin": ["", "C85", ""],
        "Embarked": ["S", "C", "S"],
    }
    path = tmp_path / "raw.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    return str(path)


@pytest.fixture(autouse=True)
def _patch_data_path(drift_csv, monkeypatch):
    """Patch settings.DATA_PATH in the drift module so simulate_drift
    reads/writes to the test CSV rather than the default path."""
    import app.drift as drift_mod

    monkeypatch.setattr(drift_mod.settings, "DATA_PATH", drift_csv)


def test_drift_injection_appends_rows(drift_csv):
    """simulate_drift should append exactly n_samples rows to the CSV."""
    n = 50
    result = simulate_drift(n_samples=n)

    assert result["n_injected"] == n

    df = pd.read_csv(drift_csv)
    assert len(df) == 3 + n


def test_drift_injection_has_correct_columns(drift_csv):
    """Injected rows should have the same columns as the original data."""
    original_df = pd.read_csv(drift_csv)
    original_cols = set(original_df.columns)

    simulate_drift(n_samples=10)

    after_df = pd.read_csv(drift_csv)
    assert set(after_df.columns) == original_cols


def test_drift_injection_inverts_patterns(drift_csv):
    """Drifted data should contain inverted survival patterns."""
    simulate_drift(n_samples=200)

    df = pd.read_csv(drift_csv)
    drifted = df[df["PassengerId"] > 3]

    # 1st class males should sometimes survive (inverted from original pattern)
    first_class_males = drifted[(drifted["Pclass"] == 1) & (drifted["Sex"] == "male")]
    if len(first_class_males) > 0:
        assert first_class_males["Survived"].sum() > 0, "Expected some 1st-class males to survive in drifted data"
