
"""Tests for the model's permitted input features."""

import pandas as pd
import pytest

from maintain.data import EXCLUDED_COLUMNS, MODEL_FEATURES
from maintain.features import (
    select_training_features,
    validate_prediction_features,
)


def example_features():
    """A small, valid equipment observation for testing."""

    return {
        "Type": "L",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0,
    }


def test_training_removes_leakage_columns():
    """Training inputs must exclude identifiers and failure labels."""

    row = example_features()

    for column in EXCLUDED_COLUMNS:
        row[column] = 1

    source = pd.DataFrame([row])
    selected = select_training_features(source)

    assert list(selected.columns) == MODEL_FEATURES
    assert not set(EXCLUDED_COLUMNS).intersection(selected.columns)


def test_training_requires_all_model_features():
    """A missing source feature must cause an explicit error."""

    source = pd.DataFrame([example_features()])
    source = source.drop(columns=["Torque [Nm]"])

    with pytest.raises(ValueError, match="Missing model features"):
        select_training_features(source)


@pytest.mark.parametrize("forbidden_column", EXCLUDED_COLUMNS)
def test_prediction_rejects_forbidden_columns(forbidden_column):
    """Identifiers, targets and failure modes cannot enter inference."""

    row = example_features()
    row[forbidden_column] = 1

    with pytest.raises(ValueError, match="Unexpected"):
        validate_prediction_features(pd.DataFrame([row]))


def test_prediction_rejects_missing_features():
    """Every prediction needs all six documented inputs."""

    row = example_features()
    del row["Tool wear [min]"]

    with pytest.raises(ValueError, match="Missing"):
        validate_prediction_features(pd.DataFrame([row]))


def test_prediction_rejects_unknown_features():
    """An unrecognised field must not be silently accepted."""

    row = example_features()
    row["Operator notes"] = "recent inspection"

    with pytest.raises(ValueError, match="Unexpected"):
        validate_prediction_features(pd.DataFrame([row]))


def test_valid_prediction_features_are_accepted():
    """A correctly formed feature table should pass this check."""

    values = pd.DataFrame([example_features()])
    checked = validate_prediction_features(values)

    assert list(checked.columns) == MODEL_FEATURES
    pd.testing.assert_frame_equal(checked, values)
