
"""Checks for the evaluation charts and their source data."""

import shutil

import pandas as pd
import pytest
import yaml

from maintain import plot_penalties, plot_performance, plot_results
from maintain.data import PROJECT_ROOT


SAVED_REPORTS = PROJECT_ROOT / "artifacts" / "example"

INPUT_FILES = [
    "split_manifest.csv",
    "validation_predictions.csv",
    "holdout_predictions.csv",
    "validation_decisions.csv",
    "holdout_metrics.csv",
    "baseline_metrics.csv",
]


@pytest.fixture
def report_folder(tmp_path, monkeypatch):
    """Work with temporary copies instead of modifying saved results."""

    for filename in INPUT_FILES:
        shutil.copyfile(
            SAVED_REPORTS / filename,
            tmp_path / filename,
        )

    for module in (
        plot_results,
        plot_penalties,
        plot_performance,
    ):
        monkeypatch.setattr(module, "REPORT_DIR", tmp_path)

    return tmp_path


@pytest.mark.parametrize(
    ("filename", "split_name", "failures"),
    [
        ("validation_predictions.csv", "validation", 45),
        ("holdout_predictions.csv", "test", 39),
    ],
)
def test_prediction_files_match_the_saved_splits(
    report_folder, filename, split_name, failures
):
    """Both prediction files must contain the correct observations."""

    manifest = pd.read_csv(report_folder / "split_manifest.csv")

    actual, scores = plot_results.load_predictions(
        filename,
        split_name,
        failures,
        manifest,
    )

    assert len(actual) == 2000
    assert len(scores) == 2000
    assert int(actual.sum()) == failures


def test_rejects_changed_prediction_ids(report_folder):
    """A reordered or replaced observation must be detected."""

    path = report_folder / "validation_predictions.csv"
    predictions = pd.read_csv(path)

    predictions.loc[0, "UDI"] = 999999
    predictions.to_csv(path, index=False)

    manifest = pd.read_csv(report_folder / "split_manifest.csv")

    with pytest.raises(ValueError, match="Incorrect records"):
        plot_results.load_predictions(
            "validation_predictions.csv",
            "validation",
            45,
            manifest,
        )


@pytest.mark.parametrize(
    ("bad_score", "error_message"),
    [
        (-0.1, "Scores outside"),
        (float("nan"), "Invalid prediction scores"),
    ],
)
def test_rejects_invalid_plot_scores(
    report_folder, bad_score, error_message
):
    """Plots should not accept missing or impossible model scores."""

    path = report_folder / "validation_predictions.csv"
    predictions = pd.read_csv(path)

    predictions.loc[0, "random_forest_score"] = bad_score
    predictions.to_csv(path, index=False)

    manifest = pd.read_csv(report_folder / "split_manifest.csv")

    with pytest.raises(ValueError, match=error_message):
        plot_results.load_predictions(
            "validation_predictions.csv",
            "validation",
            45,
            manifest,
        )


def test_penalty_totals_match_saved_results(report_folder):
    """Recalculate penalties using the frozen cost assumptions."""

    config = yaml.safe_load(
        plot_penalties.CONFIG_PATH.read_text(encoding="utf-8")
    )
    costs = config["decision"]

    validation = pd.read_csv(
        report_folder / "validation_decisions.csv"
    )
    test = pd.read_csv(
        report_folder / "holdout_metrics.csv"
    )

    validation_result = plot_penalties.get_policy_result(
        validation, "random_forest", "validation_selected", costs
    )

    test_result = plot_penalties.get_policy_result(
        test, "random_forest", "validation_selected", costs
    )

    assert int(validation_result["penalty_units"]) == 3030
    assert int(test_result["penalty_units"]) == 4410


def test_rejects_incorrect_saved_penalty(report_folder):
    """A plotting function should not trust an incorrect penalty."""

    results = pd.read_csv(
        report_folder / "validation_decisions.csv"
    )

    selected = (
        (results["model"] == "random_forest")
        & (results["policy"] == "validation_selected")
    )

    results.loc[selected, "penalty_units"] = 9999

    costs = {
        "false_positive_penalty": 10,
        "false_negative_penalty": 500,
    }

    with pytest.raises(ValueError, match="Saved penalty"):
        plot_penalties.get_policy_result(
            results, "random_forest", "validation_selected", costs
        )


def test_rejects_missing_policy(report_folder):
    """A missing policy must not be silently substituted."""

    results = pd.read_csv(
        report_folder / "holdout_metrics.csv"
    )

    costs = {
        "false_positive_penalty": 10,
        "false_negative_penalty": 500,
    }

    with pytest.raises(ValueError, match="Expected one result"):
        plot_penalties.get_policy_result(
            results, "random_forest", "unknown_policy", costs
        )


def test_performance_metrics_match_recorded_counts(report_folder):
    """Validate the final chart values against actual error counts."""

    metrics, threshold = (
        plot_performance.load_performance_results()
    )

    assert threshold == pytest.approx(0.2836220952642361)

    assert metrics["Average precision"] == pytest.approx(
        (0.5442, 0.6764),
        abs=0.0001,
    )

    assert metrics["Precision"] == pytest.approx(
        (42 / 195, 32 / 123)
    )

    assert metrics["Recall"] == pytest.approx(
        (42 / 45, 32 / 39)
    )

    assert metrics["False-positive rate"] == pytest.approx(
        (153 / 1955, 91 / 1961)
    )


def test_rejects_changed_frozen_threshold(
    report_folder, tmp_path, monkeypatch
):
    """The chart must not quietly use a different threshold."""

    config = yaml.safe_load(
        plot_performance.CONFIG_PATH.read_text(encoding="utf-8")
    )

    config["decision"]["threshold"] = 0.5

    changed_config = tmp_path / "changed_decision.yaml"
    changed_config.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        plot_performance,
        "CONFIG_PATH",
        changed_config,
    )

    with pytest.raises(ValueError, match="threshold has changed"):
        plot_performance.load_performance_results()


@pytest.mark.parametrize(
    ("module", "function_name", "output_name"),
    [
        (
            plot_results,
            "plot_precision_recall",
            "precision_recall_comparison.png",
        ),
        (
            plot_penalties,
            "plot_decision_penalties",
            "decision_penalty_comparison.png",
        ),
        (
            plot_performance,
            "plot_performance_comparison",
            "validation_test_performance.png",
        ),
    ],
)
def test_charts_generate_valid_pngs(
    report_folder, module, function_name, output_name
):
    """Each plotting script should produce a non-empty PNG."""

    plot_function = getattr(module, function_name)
    plot_function()

    output_path = report_folder / output_name

    assert output_path.is_file()

    image_bytes = output_path.read_bytes()

    # PNG files begin with a standard eight-byte signature.
    assert image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(image_bytes) > 2000
