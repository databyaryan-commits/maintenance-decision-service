
"""Small, repeatable checks for final holdout calculations."""

import numpy as np
import pandas as pd
import pytest

from maintain import holdout
from maintain.data import MODEL_FEATURES


COSTS = {
    "false_positive_penalty": 10,
    "false_negative_penalty": 500,
}


def example_features():
    """Two equipment observations with the expected input columns."""

    return pd.DataFrame({
        "Type": ["L", "M"],
        "Air temperature [K]": [298.1, 299.2],
        "Process temperature [K]": [308.6, 309.1],
        "Rotational speed [rpm]": [1551, 1400],
        "Torque [Nm]": [42.8, 50.1],
        "Tool wear [min]": [0, 12],
    })


class ExampleModel:
    """Return fixed scores so the checks don't require model training."""

    def __init__(self, scores, classes=(0, 1), feature_names=None):
        self.scores = np.asarray(scores, dtype=float)
        self.classes_ = np.asarray(classes)
        self.feature_names_in_ = np.asarray(
            MODEL_FEATURES if feature_names is None else feature_names
        )

    def predict_proba(self, features):
        assert len(features) == len(self.scores)
        return np.column_stack([1 - self.scores, self.scores])


def test_policy_counts_and_penalty():
    """Count one missed failure and one false alarm correctly."""

    actual = np.array([0, 0, 1, 1])
    scores = np.array([0.8, 0.1, 0.2, 0.9])

    result = holdout.measure_policy(
        "random_forest", "default_0.5", actual, scores, 0.5, COSTS
    )

    assert result["true_positives"] == 1
    assert result["true_negatives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["false_positive_rate"] == pytest.approx(0.5)
    assert result["penalty_units"] == 510


def test_score_equal_to_threshold_triggers_review():
    """The documented rule is review when score >= threshold."""

    actual = np.array([1, 0])
    scores = np.array([0.3, 0.2])

    result = holdout.measure_policy(
        "random_forest", "validation_selected",
        actual, scores, 0.3, COSTS
    )

    assert result["true_positives"] == 1
    assert result["true_negatives"] == 1
    assert result["penalty_units"] == 0


def test_model_returns_positive_class_scores():
    """The score for class 1 must be used, not class 0."""

    features = example_features()
    model = ExampleModel([0.2, 0.8])

    scores = holdout.get_failure_scores(model, features)

    np.testing.assert_allclose(scores, [0.2, 0.8])


def test_model_rejects_unexpected_class_order():
    """A reversed class order would change the meaning of the score."""

    model = ExampleModel([0.2, 0.8], classes=(1, 0))

    with pytest.raises(ValueError, match="Expected binary classes"):
        holdout.get_failure_scores(model, example_features())


def test_model_rejects_changed_feature_names():
    """The fitted feature schema must match the approved inputs."""

    model = ExampleModel(
        [0.2, 0.8],
        feature_names=MODEL_FEATURES[:-1],
    )

    with pytest.raises(ValueError, match="feature names do not match"):
        holdout.get_failure_scores(model, example_features())


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf")])
def test_model_rejects_nonfinite_scores(bad_score):
    """Prediction scores must be finite numbers."""

    model = ExampleModel([0.2, bad_score])

    with pytest.raises(ValueError, match="non-finite scores"):
        holdout.get_failure_scores(model, example_features())


@pytest.mark.parametrize("bad_score", [-0.1, 1.1])
def test_model_rejects_scores_outside_probability_range(bad_score):
    """Classification scores must stay between zero and one."""

    model = ExampleModel([0.2, bad_score])

    with pytest.raises(ValueError, match="between 0 and 1"):
        holdout.get_failure_scores(model, example_features())


def test_holdout_rejects_changed_dataset(tmp_path, monkeypatch):
    """Do not evaluate if source bytes differ from the frozen record."""

    changed_data = tmp_path / "ai4i2020.csv"
    changed_data.write_text("changed dataset", encoding="utf-8")

    monkeypatch.setattr(holdout, "DATA_PATH", changed_data)

    with pytest.raises(ValueError, match="Dataset checksum has changed"):
        holdout.load_holdout_data({
            "dataset": {"sha256": "0" * 64}
        })
