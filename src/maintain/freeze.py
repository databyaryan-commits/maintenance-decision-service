
"""Freeze the model and threshold selected on validation data."""

import json
import math
import platform
from hashlib import sha256

import pandas as pd
import sklearn
import yaml

from maintain.data import MODEL_FEATURES, PROJECT_ROOT


RESULT_DIR = PROJECT_ROOT / "artifacts" / "example"
CONFIG_PATH = PROJECT_ROOT / "configs" / "decision.yaml"


def calculate_checksum(file_path):
    """Calculate a file checksum without loading it all into memory."""

    digest = sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def freeze_validation_selection():
    """Record the selected model without evaluating the test block."""

    selection_path = RESULT_DIR / "validation_selection.json"
    results_path = RESULT_DIR / "validation_decisions.csv"
    metadata_path = RESULT_DIR / "split_metadata.json"

    with selection_path.open(encoding="utf-8") as file:
        selection = json.load(file)

    with metadata_path.open(encoding="utf-8") as file:
        dataset_metadata = json.load(file)

    # A decision can only be frozen from our validation experiment.
    if selection["selection_stage"] != "validation_only":
        raise ValueError("Model selection must use validation data.")

    if selection["test_evaluated"] is not False:
        raise ValueError("The test block has already been evaluated.")

    model_name = selection["selected_model"]

    if model_name not in {
        "dummy",
        "logistic_regression",
        "random_forest",
    }:
        raise ValueError(f"Unknown selected model: {model_name}")

    threshold = float(selection["selected_threshold"])

    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Selected threshold must be between 0 and 1.")

    # Verify the saved decision against our comparison results.
    results = pd.read_csv(results_path)

    matching = results.loc[
        (results["model"] == model_name)
        & (results["policy"] == "validation_selected")
    ]

    if len(matching) != 1:
        raise ValueError("Could not identify the selected validation result.")

    result = matching.iloc[0]

    if not math.isclose(
        float(result["threshold"]),
        threshold,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("Selected threshold does not match the results.")

    fp_cost = int(selection["false_positive_penalty"])
    fn_cost = int(selection["false_negative_penalty"])

    penalty = (
        fp_cost * int(result["false_positives"])
        + fn_cost * int(result["false_negatives"])
    )

    if penalty != int(selection["validation_metrics"]["penalty_units"]):
        raise ValueError("Saved validation penalty does not match.")

    model_relative_path = f"models/{model_name}.joblib"
    model_path = PROJECT_ROOT / model_relative_path

    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    # The checksum identifies the exact model file we selected.
    model_checksum = calculate_checksum(model_path)

    frozen_config = {
        "model": {
            "name": model_name,
            "artifact": model_relative_path,
            "sha256": model_checksum,
        },
        "dataset": {
            "source": "data/raw/ai4i2020.csv",
            "sha256": dataset_metadata["sha256"],
        },
        "features": MODEL_FEATURES,
        "decision": {
            "threshold": threshold,
            "rule": "score >= threshold",
            "false_positive_penalty": fp_cost,
            "false_negative_penalty": fn_cost,
        },
        "selection": {
            "source": "artifacts/example/validation_selection.json",
            "split": "validation",
            "status": "frozen",
            "test_used_for_selection": False,
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
        },
    }

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Do not silently overwrite a previously frozen decision.
    if CONFIG_PATH.exists():
        raise FileExistsError(
            f"Frozen configuration already exists: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        yaml.safe_dump(frozen_config, file, sort_keys=False)

    print("\nMODEL CONFIGURATION FROZEN")
    print("-" * 55)
    print(f"Model: {model_name}")
    print(f"Threshold: {threshold:.8f}")
    print(f"Validation penalty: {penalty}")
    print(f"Model SHA-256: {model_checksum}")
    print(f"Configuration: {CONFIG_PATH}")
    print("\nNo test data was evaluated.")
    print("Configuration freeze completed successfully.")


if __name__ == "__main__":
    freeze_validation_selection()
