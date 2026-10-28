
"""Compare decision penalties using saved validation and test results."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from maintain.data import PROJECT_ROOT


REPORT_DIR = PROJECT_ROOT / "artifacts" / "example"
CONFIG_PATH = PROJECT_ROOT / "configs" / "decision.yaml"

POLICIES = [
    ("dummy", "default_0.5", "Dummy baseline"),
    ("random_forest", "default_0.5", "RF: threshold 0.5"),
    ("random_forest", "validation_selected", "RF: selected threshold"),
]


def get_policy_result(results, model_name, policy_name, costs):
    """Read one policy and verify its recorded penalty."""

    matching = results.loc[
        (results["model"] == model_name)
        & (results["policy"] == policy_name)
    ]

    if len(matching) != 1:
        raise ValueError(
            f"Expected one result for {model_name}: {policy_name}"
        )

    result = matching.iloc[0]

    false_positives = int(result["false_positives"])
    false_negatives = int(result["false_negatives"])

    calculated_penalty = (
        costs["false_positive_penalty"] * false_positives
        + costs["false_negative_penalty"] * false_negatives
    )

    if calculated_penalty != int(result["penalty_units"]):
        raise ValueError(
            f"Saved penalty does not match: {model_name}, {policy_name}"
        )

    return result


def plot_decision_penalties():
    """Plot the three decision policies without changing the model."""

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if (
        config["model"]["name"] != "random_forest"
        or config["selection"]["status"] != "frozen"
    ):
        raise ValueError("Expected the frozen Random Forest.")

    costs = config["decision"]
    threshold = float(costs["threshold"])

    validation = pd.read_csv(
        REPORT_DIR / "validation_decisions.csv"
    )

    test = pd.read_csv(
        REPORT_DIR / "holdout_metrics.csv"
    )

    if not (test["split"] == "test").all():
        raise ValueError("Expected test-only holdout metrics.")

    labels = []
    validation_penalties = []
    test_penalties = []

    print("\nDECISION PENALTY COMPARISON")
    print("-" * 55)
    print(f"False-positive penalty: {costs['false_positive_penalty']}")
    print(f"False-negative penalty: {costs['false_negative_penalty']}")
    print(f"Frozen threshold: {threshold:.8f}")

    for model_name, policy_name, label in POLICIES:
        validation_result = get_policy_result(
            validation, model_name, policy_name, costs
        )

        test_result = get_policy_result(
            test, model_name, policy_name, costs
        )

        expected_threshold = (
            threshold if policy_name == "validation_selected" else 0.5
        )

        for result in (validation_result, test_result):
            if not np.isclose(
                float(result["threshold"]),
                expected_threshold,
                rtol=0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"Unexpected threshold for {label}"
                )

        labels.append(label)
        validation_penalties.append(
            int(validation_result["penalty_units"])
        )
        test_penalties.append(
            int(test_result["penalty_units"])
        )

        print(f"\n{label}")
        print(f"Validation penalty: {validation_penalties[-1]:,}")
        print(f"Test penalty: {test_penalties[-1]:,}")

    # Show the two evaluation blocks side by side.
    positions = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    validation_bars = ax.bar(
        positions - width / 2,
        validation_penalties,
        width,
        label="Validation",
    )

    test_bars = ax.bar(
        positions + width / 2,
        test_penalties,
        width,
        label="Test",
    )

    ax.bar_label(validation_bars, fmt="%.0f", padding=4)
    ax.bar_label(test_bars, fmt="%.0f", padding=4)

    ax.set_title(
        "Maintenance Decisions — Assumed Error Penalties",
        fontsize=15,
        pad=18,
    )

    ax.set_ylabel("Penalty units (lower is better)")
    ax.set_xticks(positions, labels)

    ax.set_ylim(
        0,
        max(validation_penalties + test_penalties) * 1.17,
    )

    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend()

    fig.text(
        0.5,
        0.015,
        "Assumed costs: false positive = 10, false negative = 500. "
        "Not actual financial savings.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    output_path = (
        REPORT_DIR / "decision_penalty_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nChart saved: {output_path}")
    print("Frozen model and threshold remain unchanged.")
    print("Penalty comparison completed successfully.")


if __name__ == "__main__":
    plot_decision_penalties()
