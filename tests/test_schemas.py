"""Tests for Pydantic schema validation."""

import pytest
from pydantic import ValidationError

from app.schemas import FeedbackInput, PassengerInput, PredictionResponse, RetrainResponse


class TestPassengerInput:
    def test_valid_input(self):
        p = PassengerInput(Pclass=1, Sex="female", Age=29, SibSp=0, Parch=0, Fare=100.0, Embarked="S")
        assert p.Pclass == 1
        assert p.Sex == "female"
        assert p.Fare == 100.0

    def test_minimal_input(self):
        """Only required fields: Pclass, Sex, Fare."""
        p = PassengerInput(Pclass=3, Sex="male", Fare=7.25)
        assert p.Age is None
        assert p.SibSp == 0
        assert p.Embarked == "S"

    def test_invalid_pclass(self):
        with pytest.raises(ValidationError):
            PassengerInput(Pclass=4, Sex="male", Fare=10.0)

    def test_invalid_sex(self):
        with pytest.raises(ValidationError):
            PassengerInput(Pclass=1, Sex="other", Fare=10.0)

    def test_negative_fare(self):
        with pytest.raises(ValidationError):
            PassengerInput(Pclass=1, Sex="male", Fare=-5.0)

    def test_age_out_of_range(self):
        with pytest.raises(ValidationError):
            PassengerInput(Pclass=1, Sex="male", Fare=10.0, Age=200)

    def test_invalid_embarked(self):
        with pytest.raises(ValidationError):
            PassengerInput(Pclass=1, Sex="male", Fare=10.0, Embarked="X")


class TestPredictionResponse:
    def test_valid_response(self):
        r = PredictionResponse(survived=True, probability=0.85, model_version="1")
        assert r.survived is True
        assert r.probability == 0.85


class TestFeedbackInput:
    def test_valid_feedback(self):
        passenger = PassengerInput(Pclass=1, Sex="female", Fare=100.0)
        f = FeedbackInput(passenger_data=passenger, actual_survived=True)
        assert f.actual_survived is True


class TestRetrainResponse:
    def test_valid_retrain(self):
        r = RetrainResponse(
            status="success",
            accuracy=0.8,
            precision=0.75,
            recall=0.7,
            f1_score=0.72,
            model_version="2",
            run_id="abc123",
            n_samples=891,
        )
        assert r.model_version == "2"
        assert r.drift_report is None
