
"""Test frozen settings without needing the real trained model."""

import json
from hashlib import sha256

import pytest
import yaml

from maintain.data import MODEL_FEATURES
from maintain.frozen import verify_frozen_configuration


@pytest.fixture
def example_project(tmp_path):
    """Build a temporary project with a small fake model file."""

    model_dir = tmp_path / "models"
    model_dir.mkdir()

    model_file = model_dir / "random_forest.joblib"
    model_file.write_bytes(b"small test artifact")

    reports = tmp_path / "artifacts" / "example"
    reports.mkdir(parents=True)

    selection = {
        "selection_stage": "validation_only",
        "test_evaluated": False,
        "selected_model": "random_forest",
        "selected_threshold": 0.28362210,
        "false_positive_penalty": 10,
        "false_negative_penalty": 500,
    }

    (reports / "validation_selection.json").write_text(
        json.dumps(selection),
        encoding="utf-8",
    )

    (reports / "split_metadata.json").write_text(
        json.dumps({"sha256": "a" * 64}),
        encoding="utf-8",
    )

    configs = tmp_path / "configs"
    configs.mkdir()

    frozen = {
        "model": {
            "name": "random_forest",
            "artifact": "models/random_forest.joblib",
            "sha256": sha256(model_file.read_bytes()).hexdigest(),
        },
        "dataset": {
            "source": "data/raw/ai4i2020.csv",
            "sha256": "a" * 64,
        },
        "features": MODEL_FEATURES.copy(),
        "decision": {
            "threshold": 0.28362210,
            "rule": "score >= threshold",
            "false_positive_penalty": 10,
            "false_negative_penalty": 500,
        },
        "selection": {
            "source": "artifacts/example/validation_selection.json",
            "split": "validation",
            "status": "frozen",
            "test_used_for_selection": False,
        },
    }

    (configs / "decision.yaml").write_text(
        yaml.safe_dump(frozen),
        encoding="utf-8",
    )

    return tmp_path


def change_configuration(project, section, field, new_value):
    """Change a saved setting to check that validation catches it."""

    path = project / "configs" / "decision.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    config[section][field] = new_value

    path.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )


def test_valid_frozen_configuration(example_project):
    """A consistent frozen configuration should pass."""

    config, model_path = verify_frozen_configuration(
        example_project
    )

    assert config["model"]["name"] == "random_forest"
    assert model_path.is_file()


def test_rejects_modified_model(example_project):
    """Changing the model file must invalidate its checksum."""

    model_path = (
        example_project / "models" / "random_forest.joblib"
    )
    model_path.write_bytes(b"changed test artifact")

    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_frozen_configuration(example_project)


def test_rejects_missing_model(example_project):
    """A missing saved model should produce an explicit error."""

    (
        example_project / "models" / "random_forest.joblib"
    ).unlink()

    with pytest.raises(
        FileNotFoundError,
        match="model file missing",
    ):
        verify_frozen_configuration(example_project)


def test_rejects_changed_threshold(example_project):
    """The frozen threshold must match validation selection."""

    change_configuration(
        example_project,
        "decision",
        "threshold",
        0.5,
    )

    with pytest.raises(ValueError, match="validation selection"):
        verify_frozen_configuration(example_project)


def test_rejects_extra_feature(example_project):
    """Failure-mode columns cannot enter the frozen feature list."""

    path = example_project / "configs" / "decision.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    config["features"].append("TWF")

    path.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="approved feature list"):
        verify_frozen_configuration(example_project)


def test_rejects_unfrozen_selection(example_project):
    """A draft configuration cannot be treated as frozen."""

    change_configuration(
        example_project,
        "selection",
        "status",
        "draft",
    )

    with pytest.raises(
        ValueError,
        match="frozen validation selection",
    ):
        verify_frozen_configuration(example_project)


def test_rejects_dataset_checksum_mismatch(example_project):
    """The selected dataset must match the recorded split."""

    change_configuration(
        example_project,
        "dataset",
        "sha256",
        "b" * 64,
    )

    with pytest.raises(ValueError, match="dataset metadata"):
        verify_frozen_configuration(example_project)
