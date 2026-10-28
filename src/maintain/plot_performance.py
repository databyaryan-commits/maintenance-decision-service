
"""Compare validation and test performance of the frozen model."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import PercentFormatter

from maintain.data import PROJECT_ROOT


REPORT_DIR = PROJECT_ROOT / "artifacts" / "example"
CONFIG_PATH = PROJECT_ROOT / "configs" / "decision.yaml"


def find_result(results, model_name, policy_name=None):
    """Return one saved result for the requested model and policy."""

    matching = results.loc[results["model"] == model_name]

    if policy_name is not None:
        matching = matching.loc[
            matching["policy"] == policy_name
        ]

    if len(matching) != 1:
        raise ValueError(
            f"Expected one result for {model_name}, {policy_name}"
        )

    return matching.iloc[0]


def load_performance_results():
    """Collect metrics recorded during validation and holdout."""

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if (
        config["model"]["name"] != "random_forest"
        or config["selection"]["status"] != "frozen"
    ):
        raise ValueError("Expected the frozen Random Forest.")

    threshold = float(config["decision"]["threshold"])

    validation_models = pd.read_csv(
        REPORT_DIR / "baseline_metrics.csv"
    )

    validation_decisions = pd.read_csv(
        REPORT_DIR / "validation_decisions.csv"
    )

    holdout_results = pd.read_csv(
        REPORT_DIR / "holdout_metrics.csv"
    )

    validation_model = find_result(
        validation_models,
        "random_forest",
    )

    validation_policy = find_result(
        validation_decisions,
        "random_forest",
        "validation_selected",
    )

    test_policy = find_result(
        holdout_results,
        "random_forest",
        "validation_selected",
    )

    if (
        validation_model["split"] != "validation"
        or test_policy["split"] != "test"
    ):
        raise ValueError("Unexpected evaluation split.")

    # Both blocks must use the exact threshold selected on validation.
    for result in (validation_policy, test_policy):
        if not np.isclose(
            float(result["threshold"]),
            threshold,
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError("Decision threshold has changed.")

    metrics = {
        "Average precision": (
            float(validation_model["average_precision"]),
            float(test_policy["average_precision"]),
        ),
        "Precision": (
            float(validation_policy["precision"]),
            float(test_policy["precision"]),
        ),
        "Recall": (
            float(validation_policy["recall"]),
            float(test_policy["recall"]),
        ),
        "False-positive rate": (
            float(validation_policy["false_positive_rate"]),
            float(test_policy["false_positive_rate"]),
        ),
    }

    for metric_name, values in metrics.items():
        if not all(np.isfinite(value) and 0 <= value <= 1
                   for value in values):
            raise ValueError(
                f"Invalid values for {metric_name}"
            )

    return metrics, threshold


def plot_performance_comparison():
    """Create a grouped chart from the saved evaluation results."""

    metrics, threshold = load_performance_results()

    names = list(metrics)
    validation_values = [metrics[name][0] for name in names]
    test_values = [metrics[name][1] for name in names]

    print("\nVALIDATION VS TEST PERFORMANCE")
    print("-" * 55)
    print("Model: random_forest")
    print(f"Frozen threshold: {threshold:.8f}")

    for name, (validation, test) in metrics.items():
        print(f"\n{name}")
        print(f"Validation: {validation:.2%}")
        print(f"Test: {test:.2%}")

    positions = np.arange(len(names))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6.5))

    validation_bars = ax.bar(
        positions - bar_width / 2,
        validation_values,
        bar_width,
        label="Validation",
    )

    test_bars = ax.bar(
        positions + bar_width / 2,
        test_values,
        bar_width,
        label="Test",
    )

    # Show the actual values above each bar.
    ax.bar_label(
        validation_bars,
        labels=[f"{value:.1%}" for value in validation_values],
        padding=4,
        fontsize=9,
    )

    ax.bar_label(
        test_bars,
        labels=[f"{value:.1%}" for value in test_values],
        padding=4,
        fontsize=9,
    )

    ax.set_title(
        "Frozen Random Forest — Validation vs Test",
        fontsize=15,
        pad=18,
    )

    ax.set_ylabel("Metric value")
    ax.set_xticks(positions, names)
    ax.set_ylim(0, 1.10)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")

    # Average precision measures ranking quality. The other
    # metrics describe decisions at our selected threshold.
    fig.text(
        0.5,
        0.045,
        "Precision, recall and false-positive rate use the "
        "frozen threshold.",
        ha="center",
        fontsize=9,
    )

    fig.text(
        0.5,
        0.018,
        "Higher is better except for false-positive rate. "
        "AP is threshold-independent.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.09, 1, 1])

    output_path = (
        REPORT_DIR / "validation_test_performance.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nChart saved: {output_path}")
    print("No model settings were changed.")
    print("Performance comparison completed successfully.")


if __name__ == "__main__":
    plot_performance_comparison()
