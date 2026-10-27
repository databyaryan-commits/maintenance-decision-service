
"""Check frozen model settings before evaluation or prediction."""

import json
import math
from pathlib import Path

import yaml

from maintain.data import MODEL_FEATURES, PROJECT_ROOT
from maintain.freeze import calculate_checksum


def read_json(path):
    """Read an experiment file and report if it is missing."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Required experiment file missing: {path}"
        )

    with path.open(encoding="utf-8") as file:
        return json.load(file)


def verify_frozen_configuration(project_root=PROJECT_ROOT):
    """Verify the selected settings and exact saved model file."""

    root = Path(project_root).resolve()
    config_path = root / "configs" / "decision.yaml"

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Frozen configuration missing: {config_path}"
        )

    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Frozen configuration must be a YAML mapping."
        )

    model = config.get("model", {})
    decision = config.get("decision", {})
    selection = config.get("selection", {})
    dataset = config.get("dataset", {})

    name = model.get("name")

    if name not in (
        "dummy",
        "logistic_regression",
        "random_forest",
    ):
        raise ValueError("Unrecognised frozen model name.")

    expected_model_path = f"models/{name}.joblib"

    if model.get("artifact") != expected_model_path:
        raise ValueError(
            "Frozen model path does not match its name."
        )

    if config.get("features") != MODEL_FEATURES:
        raise ValueError(
            "Frozen features differ from the approved feature list."
        )

    threshold = decision.get("threshold")

    if (
        type(threshold) not in (int, float)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError(
            "Frozen threshold must be a finite number in [0, 1]."
        )

    if decision.get("rule") != "score >= threshold":
        raise ValueError("Unexpected decision rule.")

    if (
        decision.get("false_positive_penalty") != 10
        or decision.get("false_negative_penalty") != 500
    ):
        raise ValueError(
            "Primary penalty assumptions have changed."
        )

    if (
        selection.get("status") != "frozen"
        or selection.get("split") != "validation"
        or selection.get("test_used_for_selection") is not False
        or selection.get("source")
        != "artifacts/example/validation_selection.json"
    ):
        raise ValueError(
            "The configuration is not a frozen validation selection."
        )

    # Check the frozen values against the saved validation decision.
    saved_selection = read_json(
        root / "artifacts" / "example" / "validation_selection.json"
    )

    if (
        saved_selection.get("selection_stage") != "validation_only"
        or saved_selection.get("test_evaluated") is not False
        or saved_selection.get("selected_model") != name
        or saved_selection.get("selected_threshold") != threshold
        or saved_selection.get("false_positive_penalty") != 10
        or saved_selection.get("false_negative_penalty") != 500
    ):
        raise ValueError(
            "Frozen settings disagree with validation selection."
        )

    # The dataset recorded at selection must still match our split.
    saved_metadata = read_json(
        root / "artifacts" / "example" / "split_metadata.json"
    )

    if (
        dataset.get("source") != "data/raw/ai4i2020.csv"
        or dataset.get("sha256") != saved_metadata.get("sha256")
    ):
        raise ValueError(
            "Frozen dataset metadata does not match the split."
        )

    model_path = root / expected_model_path

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Selected model file missing: {model_path}"
        )

    # Check the artifact bytes before the model is loaded.
    if calculate_checksum(model_path) != model.get("sha256"):
        raise ValueError(
            "Model checksum mismatch; model file has changed."
        )

    return config, model_path


if __name__ == "__main__":
    settings, saved_model = verify_frozen_configuration()

    print("Frozen configuration verified.")
    print(f"Model: {settings['model']['name']}")
    print(f"Threshold: {settings['decision']['threshold']:.8f}")
    print(f"Model file: {saved_model}")
    print("No test observations were loaded.")
