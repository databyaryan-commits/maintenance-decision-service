
"""Tests for the maintenance prediction API."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from maintain import api
from maintain.data import MODEL_FEATURES
from maintain.features import validate_prediction_features
from maintain.holdout import get_failure_scores


class ExampleModel:
    """A predictable model for testing the API without training."""

    classes_ = np.array([0, 1])
    feature_names_in_ = np.array(MODEL_FEATURES)

    def predict_proba(self, features):
        """Return fixed scores based on torque for these tests."""

        assert list(features.columns) == MODEL_FEATURES

        scores = []

        for torque in features["Torque [Nm]"]:
            if torque >= 60:
                scores.append(0.8)
            elif torque == 50:
                scores.append(0.3)
            else:
                scores.append(0.2)

        scores = np.asarray(scores, dtype=float)

        return np.column_stack([1 - scores, scores])


def example_request():
    """A valid set of equipment measurements."""

    return {
        "Type": "L",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0,
    }


@pytest.fixture
def service(monkeypatch):
    """Provide a test model without touching the real saved artifact."""

    model = ExampleModel()

    config = {
        "model": {
            "name": "random_forest",
            "sha256": "1234567890abcdef",
        },
        "decision": {
            "threshold": 0.3,
            "rule": "score >= threshold",
        },
        "features": MODEL_FEATURES.copy(),
        "dataset": {
            "sha256": "test-dataset-checksum",
        },
    }

    monkeypatch.setattr(
        api,
        "load_service_model",
        lambda: (config, model),
    )

    return TestClient(api.app), model


def test_health_returns_ready_status(service):
    """The health endpoint should report a loaded model."""

    client, _ = service
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
    }


def test_model_info_returns_frozen_settings(service):
    """Model information should match the selected configuration."""

    client, _ = service
    response = client.get("/model-info")

    assert response.status_code == 200

    details = response.json()

    assert details["model_name"] == "random_forest"
    assert details["threshold"] == pytest.approx(0.3)
    assert details["decision_rule"] == "score >= threshold"
    assert details["features"] == MODEL_FEATURES
    assert details["model_version"].endswith("+1234567890ab")


def test_valid_prediction_returns_normal(service):
    """A score below the threshold should produce a normal decision."""

    client, _ = service
    response = client.post("/predict", json=example_request())

    assert response.status_code == 200

    result = response.json()

    assert result["score"] == pytest.approx(0.2)
    assert result["threshold"] == pytest.approx(0.3)
    assert result["action"] == "normal"
    assert result["explanation"]


def test_valid_prediction_returns_review(service):
    """A score above the threshold should trigger review."""

    client, _ = service

    request = example_request()
    request["Torque [Nm]"] = 60.0

    response = client.post("/predict", json=request)

    assert response.status_code == 200
    assert response.json()["score"] == pytest.approx(0.8)
    assert response.json()["action"] == "review"


def test_score_equal_to_threshold_triggers_review(service):
    """The frozen decision rule includes threshold equality."""

    client, _ = service

    request = example_request()
    request["Torque [Nm]"] = 50.0

    response = client.post("/predict", json=request)

    assert response.status_code == 200
    assert response.json()["score"] == pytest.approx(0.3)
    assert response.json()["action"] == "review"


def test_api_rejects_failure_mode_column(service):
    """Failure-mode labels must not be accepted as predictors."""

    client, _ = service

    request = example_request()
    request["TWF"] = 1

    response = client.post("/predict", json=request)

    assert response.status_code == 422

    errors = response.json()["detail"]

    assert any(
        error["type"] == "extra_forbidden"
        and error["loc"] == ["body", "TWF"]
        for error in errors
    )


def test_api_rejects_missing_measurement(service):
    """All six model measurements are required."""

    client, _ = service

    request = example_request()
    del request["Tool wear [min]"]

    response = client.post("/predict", json=request)

    assert response.status_code == 422

    errors = response.json()["detail"]

    assert any(
        error["type"] == "missing"
        and error["loc"] == ["body", "Tool wear [min]"]
        for error in errors
    )


def test_api_rejects_negative_rotational_speed(service):
    """Rotational speed must be greater than zero."""

    client, _ = service

    request = example_request()
    request["Rotational speed [rpm]"] = -100

    response = client.post("/predict", json=request)

    assert response.status_code == 422

    errors = response.json()["detail"]

    assert any(
        error["type"] == "greater_than"
        and error["loc"] == ["body", "Rotational speed [rpm]"]
        for error in errors
    )


def test_api_rejects_unknown_equipment_type(service):
    """Only the documented L, M and H categories are permitted."""

    client, _ = service

    request = example_request()
    request["Type"] = "X"

    response = client.post("/predict", json=request)

    assert response.status_code == 422


def test_api_rejects_numeric_strings(service):
    """Measurements should use JSON numbers rather than strings."""

    client, _ = service

    request = example_request()
    request["Torque [Nm]"] = "42.8"

    response = client.post("/predict", json=request)

    assert response.status_code == 422


def test_api_matches_batch_scoring(service):
    """The API and batch scoring path should return the same score."""

    client, model = service
    request = example_request()

    response = client.post("/predict", json=request)

    assert response.status_code == 200

    # Use the same feature checks and scoring function as evaluation.
    measurements = pd.DataFrame([request])
    features = validate_prediction_features(measurements)

    batch_score = float(
        get_failure_scores(model, features)[0]
    )

    assert response.json()["score"] == pytest.approx(
        batch_score,
        abs=1e-12,
    )


def test_health_reports_missing_model(monkeypatch):
    """The API should return 503 when model verification fails."""

    def missing_model():
        raise FileNotFoundError("Model file is missing.")

    monkeypatch.setattr(
        api,
        "verify_frozen_configuration",
        missing_model,
    )

    # Clear any previously loaded model before testing failure.
    api.load_service_model.cache_clear()

    try:
        client = TestClient(api.app)
        response = client.get("/health")
    finally:
        api.load_service_model.cache_clear()

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Frozen model unavailable or failed verification."
    )
