
"""Tests for the holdout error-analysis report."""

import numpy as np
import pandas as pd
import pytest

from maintain import error_analysis


@pytest.fixture
def example_errors():
    """Small set of decisions with known outcomes."""

    return pd.DataFrame({
        "Type": ["L", "L", "L", "L", "M", "M"],
        "actual": [1, 1, 0, 0, 0, 0],
        "outcome": ["TP", "FN", "FP", "TN", "TN", "TN"],
    })


def test_error_counts_by_equipment_type(example_errors):
    """Check the error counts and rates for Type L."""

    summary = error_analysis.summarise_groups(
        example_errors, "Type"
    )

    type_l = summary.loc[summary["Type"] == "L"].iloc[0]

    assert type_l["records"] == 4
    assert type_l["actual_failures"] == 2
    assert type_l["true_positives"] == 1
    assert type_l["false_positives"] == 1
    assert type_l["false_negatives"] == 1
    assert type_l["failure_recall"] == pytest.approx(0.5)
    assert type_l["false_positive_rate"] == pytest.approx(0.5)


def test_recall_is_undefined_without_failures(example_errors):
    """Do not report zero recall for a group with no failures."""

    summary = error_analysis.summarise_groups(
        example_errors, "Type"
    )

    type_m = summary.loc[summary["Type"] == "M"].iloc[0]

    assert type_m["actual_failures"] == 0
    assert np.isnan(type_m["failure_recall"])
    assert type_m["false_positive_rate"] == pytest.approx(0)


def test_confusion_matrix_creates_png(
    example_errors, tmp_path, monkeypatch
):
    """The report should save a readable PNG image file."""

    monkeypatch.setattr(
        error_analysis, "REPORT_DIR", tmp_path
    )

    error_analysis.save_confusion_matrix(example_errors)

    image_path = tmp_path / "holdout_confusion_matrix.png"

    assert image_path.is_file()
    assert image_path.stat().st_size > 0

    # Every valid PNG starts with this eight-byte signature.
    assert image_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def example_saved_report(tmp_path, monkeypatch):
    """Create temporary holdout files without loading the real model."""

    monkeypatch.setattr(
        error_analysis, "REPORT_DIR", tmp_path
    )

    config = {
        "model": {"name": "random_forest"},
        "decision": {"threshold": 0.3},
    }

    monkeypatch.setattr(
        error_analysis,
        "verify_frozen_configuration",
        lambda: (config, None),
    )

    features = pd.DataFrame({
        "Type": ["L", "M"],
    })

    actual = np.array([0, 1])
    identifiers = np.array([10, 11])

    monkeypatch.setattr(
        error_analysis,
        "load_holdout_data",
        lambda settings: (
            features,
            actual,
            identifiers,
            0.1,
        ),
    )

    predictions = pd.DataFrame({
        "UDI": [10, 11],
        "actual": [0, 1],
        "random_forest_score": [0.2, 0.9],
        "selected_action": ["normal", "review"],
    })

    predictions.to_csv(
        tmp_path / "holdout_predictions.csv",
        index=False,
    )

    metrics = pd.DataFrame([{
        "model": "random_forest",
        "policy": "validation_selected",
        "true_positives": 1,
        "false_positives": 0,
        "true_negatives": 1,
        "false_negatives": 0,
    }])

    metrics.to_csv(
        tmp_path / "holdout_metrics.csv",
        index=False,
    )

    return tmp_path


def change_prediction(report_dir, column, value):
    """Change a saved prediction to simulate a reporting mistake."""

    path = report_dir / "holdout_predictions.csv"
    predictions = pd.read_csv(path)

    predictions.loc[0, column] = value
    predictions.to_csv(path, index=False)


def test_rejects_wrong_prediction_ids(example_saved_report):
    """Prediction rows must match the saved test identifiers."""

    change_prediction(example_saved_report, "UDI", 999)

    with pytest.raises(ValueError, match="identifiers"):
        error_analysis.analyse_holdout_errors()


def test_rejects_wrong_actual_labels(example_saved_report):
    """The recorded labels must match the original holdout."""

    change_prediction(example_saved_report, "actual", 1)

    with pytest.raises(ValueError, match="labels"):
        error_analysis.analyse_holdout_errors()


def test_rejects_wrong_saved_actions(example_saved_report):
    """Saved decisions must agree with the frozen threshold."""

    change_prediction(
        example_saved_report, "selected_action", "review"
    )

    with pytest.raises(ValueError, match="Saved actions"):
        error_analysis.analyse_holdout_errors()


def test_rejects_inconsistent_error_counts(example_saved_report):
    """The error report must match the original evaluation metrics."""

    path = example_saved_report / "holdout_metrics.csv"
    metrics = pd.read_csv(path)

    metrics.loc[0, "false_positives"] = 1
    metrics.to_csv(path, index=False)

    with pytest.raises(ValueError, match="FP count disagrees"):
        error_analysis.analyse_holdout_errors()


@pytest.mark.parametrize(
    ("bad_score", "expected_message"),
    [
        (-0.1, "outside"),
        (1.1, "outside"),
        (float("nan"), "Non-finite"),
    ],
)
def test_rejects_invalid_saved_scores(
    example_saved_report, bad_score, expected_message
):
    """Reject impossible or missing scores in saved predictions."""

    change_prediction(
        example_saved_report,
        "random_forest_score",
        bad_score,
    )

    with pytest.raises(ValueError, match=expected_message):
        error_analysis.analyse_holdout_errors()
