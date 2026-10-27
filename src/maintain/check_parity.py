
"""Check that API decisions match the saved model's batch predictions."""

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from maintain.api import app, load_service_model
from maintain.data import MODEL_FEATURES
from maintain.features import validate_prediction_features


# Use the same observations for both prediction paths.
EXAMPLES = [
    {
        "Type": "L",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0,
    },
    {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1450,
        "Torque [Nm]": 60.0,
        "Tool wear [min]": 125,
    },
]


def check_prediction_parity():
    """Compare API responses against direct model predictions."""

    # Recheck the frozen configuration before loading the actual model.
    load_service_model.cache_clear()
    config, model = load_service_model()

    if list(model.classes_) != [0, 1]:
        raise ValueError("Unexpected model classes.")

    threshold = float(config["decision"]["threshold"])

    # TestClient sends requests through FastAPI in the current process.
    # We don't need a separate running web server for this check.
    with TestClient(app) as client:
        health = client.get("/health")

        if health.status_code != 200:
            raise AssertionError("API health check failed.")

        metadata = client.get("/model-info")

        if metadata.status_code != 200:
            raise AssertionError("Model information request failed.")

        if metadata.json()["threshold"] != threshold:
            raise AssertionError("API threshold differs from the frozen one.")

        print("\nREAL-MODEL PREDICTION PARITY")
        print("-" * 55)
        print(f"Model: {config['model']['name']}")
        print(f"Frozen threshold: {threshold:.8f}")

        for number, observation in enumerate(EXAMPLES, start=1):
            response = client.post("/predict", json=observation)

            if response.status_code != 200:
                raise AssertionError(
                    f"API request {number} failed: {response.text}"
                )

            api_result = response.json()

            # Recreate the feature table using the original column names.
            batch_input = pd.DataFrame([observation])
            batch_input = validate_prediction_features(batch_input)
            batch_input = batch_input.loc[:, MODEL_FEATURES]

            # Run the actual fitted sklearn pipeline directly.
            batch_score = float(
                model.predict_proba(batch_input)[0, 1]
            )

            api_score = float(api_result["score"])

            if not np.isclose(
                api_score,
                batch_score,
                rtol=0,
                atol=1e-12,
            ):
                raise AssertionError(
                    f"Prediction {number} has different scores."
                )

            expected_action = (
                "review" if batch_score >= threshold else "normal"
            )

            if api_result["action"] != expected_action:
                raise AssertionError(
                    f"Prediction {number} has different actions."
                )

            if api_result["threshold"] != threshold:
                raise AssertionError(
                    f"Prediction {number} has a different threshold."
                )

            if (
                api_result["model_version"]
                != metadata.json()["model_version"]
            ):
                raise AssertionError(
                    f"Prediction {number} has a different model version."
                )

            print(f"\nObservation {number}")
            print(f"API score:   {api_score:.12f}")
            print(f"Batch score: {batch_score:.12f}")
            print(f"Difference:  {abs(api_score - batch_score):.12f}")
            print(f"Decision:    {expected_action}")
            print("Result: PASS")

    print("\nBoth API and batch predictions match.")
    print("The frozen model and threshold remain unchanged.")


if __name__ == "__main__":
    check_prediction_parity()
