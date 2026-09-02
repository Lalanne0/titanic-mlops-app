"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field


# LESSON POINT: API Request Validation using Pydantic
# Pydantic is a data validation library that FastAPI uses under the hood. By defining a schema like
# PassengerInput, we validate request payloads before they reach our machine learning code.
# The `Field` parameters define constraints (like age constraints, matching regexes for port/sex),
# automatically returning a 422 Unprocessable Entity error if a client sends invalid values.
class PassengerInput(BaseModel):
    """Input features for a Titanic passenger."""

    Pclass: int = Field(..., ge=1, le=3, description="Ticket class: 1=1st, 2=2nd, 3=3rd")
    Sex: str = Field(..., pattern="^(male|female)$", description="male or female")
    Age: float | None = Field(None, ge=0, le=120, description="Age in years")
    SibSp: int = Field(0, ge=0, description="Number of siblings/spouses aboard")
    Parch: int = Field(0, ge=0, description="Number of parents/children aboard")
    Fare: float = Field(..., ge=0, description="Passenger fare")
    Embarked: str | None = Field("S", pattern="^[SCQ]$", description="Port: S=Southampton, C=Cherbourg, Q=Queenstown")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Pclass": 1,
                    "Sex": "female",
                    "Age": 29,
                    "SibSp": 0,
                    "Parch": 0,
                    "Fare": 100.0,
                    "Embarked": "S",
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Prediction result returned to the user."""

    survived: bool
    probability: float
    model_version: str


class FeedbackInput(BaseModel):
    """User feedback when the prediction was wrong."""

    passenger_data: PassengerInput
    actual_survived: bool = Field(..., description="Did the passenger actually survive?")


class RetrainResponse(BaseModel):
    """Result of a model retraining run."""

    status: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    model_version: str
    run_id: str
    n_samples: int
    drift_report: str | None = None
