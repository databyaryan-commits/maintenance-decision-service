
"""Serve predictions from the frozen maintenance model."""

import logging
from functools import lru_cache
from importlib.metadata import version
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from maintain.data import MODEL_FEATURES
from maintain.features import validate_prediction_features
from maintain.frozen import verify_frozen_configuration
from maintain.holdout import get_failure_scores


logger = logging.getLogger(__name__)

app = FastAPI(
    title="Maintenance Decision Service",
    description=(
        "A local ML service for classifying equipment failure "
        "observations and applying a validation-selected review threshold."
    ),
    version="0.1.0",
)


class EquipmentInput(BaseModel):
    """The six measurements expected by the trained model."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )

    machine_type: Literal["L", "M", "H"] = Field(
        alias="Type",
        description="Product quality category.",
    )

    air_temperature: float = Field(
        alias="Air temperature [K]",
        ge=200,
        le=400,
        description="Air temperature in Kelvin.",
    )

    process_temperature: float = Field(
        alias="Process temperature [K]",
        ge=200,
        le=450,
        description="Process temperature in Kelvin.",
    )

    rotational_speed: int = Field(
        alias="Rotational speed [rpm]",
        gt=0,
        le=20000,
        description="Rotational speed in revolutions per minute.",
    )

    torque: float = Field(
        alias="Torque [Nm]",
        ge=0,
        le=1000,
        description="Torque in newton-metres.",
    )

    tool_wear: int = Field(
        alias="Tool wear [min]",
        ge=0,
        le=100000,
        description="Accumulated tool wear in minutes.",
    )


class PredictionResponse(BaseModel):
    """The score and action returned by the prediction endpoint."""

    score: float
    threshold: float
    action: Literal["review", "normal"]
    model_version: str
    explanation: str


class ModelInfo(BaseModel):
    """Information about the model currently used for predictions."""

    model_name: str
    model_version: str
    threshold: float
    decision_rule: str
    features: list[str]
    dataset_sha256: str


def model_version(config):
    """Identify the package version and exact trained model artifact."""

    package_version = version("maintenance-decision-service")
    model_hash = config["model"]["sha256"][:12]

    return f"{package_version}+{model_hash}"


@lru_cache(maxsize=1)
def load_service_model():
    """Verify and load our locally trained model once per process."""

    try:
        config, model_path = verify_frozen_configuration()

        # Joblib files use Python pickle internally. Only load the
        # model artifact created and verified within this project.
        model = joblib.load(model_path)

        if not hasattr(model, "predict_proba"):
            raise ValueError(
                "Saved model does not support probability predictions."
            )

        return config, model

    except Exception as error:
        logger.exception("Could not load the frozen model")

        raise HTTPException(
            status_code=503,
            detail="Frozen model unavailable or failed verification.",
        ) from error


@app.get("/health")
def health():
    """Report whether the saved model is ready to serve."""

    load_service_model()

    return {
        "status": "ok",
        "model_loaded": True,
    }


@app.get("/model-info", response_model=ModelInfo)
def get_model_info():
    """Return the settings used by the prediction service."""

    config, _ = load_service_model()

    return ModelInfo(
        model_name=config["model"]["name"],
        model_version=model_version(config),
        threshold=config["decision"]["threshold"],
        decision_rule=config["decision"]["rule"],
        features=config["features"],
        dataset_sha256=config["dataset"]["sha256"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: EquipmentInput):
    """Apply the fitted model and frozen review threshold."""

    config, model = load_service_model()

    # Keep the original source column names and their fitted order.
    measurements = pd.DataFrame(
        [request.model_dump(by_alias=True)]
    )

    features = validate_prediction_features(measurements)
    features = features.loc[:, MODEL_FEATURES]

    # Reuse the same score calculation used for holdout evaluation.
    score = float(get_failure_scores(model, features)[0])

    threshold = float(config["decision"]["threshold"])

    # A score equal to the threshold also triggers a review.
    action = "review" if score >= threshold else "normal"

    if action == "review":
        explanation = (
            "The model score meets or exceeds the frozen threshold. "
            "This observation is flagged for review."
        )
    else:
        explanation = (
            "The model score is below the frozen threshold. "
            "No review is flagged by this decision rule."
        )

    return PredictionResponse(
        score=score,
        threshold=threshold,
        action=action,
        model_version=model_version(config),
        explanation=explanation,
    )
