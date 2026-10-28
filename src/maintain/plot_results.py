
"""Create precision-recall plots from saved model predictions."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve

from maintain.data import PROJECT_ROOT


REPORT_DIR = PROJECT_ROOT / "artifacts" / "example"
CONFIG_PATH = PROJECT_ROOT / "configs" / "decision.yaml"

SPLITS = {
    "Validation": ("validation_predictions.csv", "validation", 45),
    "Test": ("holdout_predictions.csv", "test", 39),
}


def load_predictions(filename, split_name, expected_failures, manifest):
    """Read saved predictions and check that they belong to the right split."""

    predictions = pd.read_csv(REPORT_DIR / filename)

    required_columns = {"UDI", "actual", "random_forest_score"}

    if not required_columns.issubset(predictions.columns):
        raise ValueError(f"Required columns missing from {filename}")

    expected_ids = (
        manifest.loc[manifest["split"] == split_name, "UDI"]
        .reset_index(drop=True)
    )

    actual_ids = predictions["UDI"].reset_index(drop=True)

    if (
        len(predictions) != 2000
        or not actual_ids.equals(expected_ids)
    ):
        raise ValueError(f"Incorrect records in {filename}")

    if not predictions["actual"].isin([0, 1]).all():
        raise ValueError(f"Invalid failure labels in {filename}")

    actual = predictions["actual"].to_numpy(dtype=int)
    scores = predictions["random_forest_score"].to_numpy(dtype=float)

    if int(actual.sum()) != expected_failures:
        raise ValueError(f"Unexpected failure count in {filename}")

    if not np.isfinite(scores).all():
        raise ValueError(f"Invalid prediction scores in {filename}")

    if ((scores < 0) | (scores > 1)).any():
        raise ValueError(f"Scores outside [0, 1] in {filename}")

    return actual, scores


def plot_precision_recall():
    """Compare the frozen Random Forest across validation and test blocks."""

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if (
        config["model"]["name"] != "random_forest"
        or config["selection"]["status"] != "frozen"
    ):
        raise ValueError("Expected the frozen Random Forest configuration.")

    threshold = float(config["decision"]["threshold"])

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid frozen decision threshold.")

    manifest = pd.read_csv(REPORT_DIR / "split_manifest.csv")

    fig, ax = plt.subplots(figsize=(9, 6))

    print("\nPRECISION-RECALL COMPARISON")
    print("-" * 55)
    print(f"Model: random_forest")
    print(f"Frozen threshold: {threshold:.8f}")

    for name, (filename, split, failures) in SPLITS.items():
        actual, scores = load_predictions(
            filename, split, failures, manifest
        )

        precision, recall, _ = precision_recall_curve(
            actual, scores
        )

        average_precision = average_precision_score(
            actual, scores
        )

        # Plot the full precision-recall relationship.
        line, = ax.step(
            recall,
            precision,
            where="post",
            linewidth=2,
            label=f"{name} (AP = {average_precision:.4f})",
        )

        # Mark the decision point selected during validation.
        review = scores >= threshold

        true_positives = int(
            np.sum((actual == 1) & review)
        )

        predicted_reviews = int(review.sum())
        actual_failures = int(actual.sum())

        selected_precision = (
            true_positives / predicted_reviews
            if predicted_reviews else 0.0
        )

        selected_recall = (
            true_positives / actual_failures
            if actual_failures else 0.0
        )

        ax.scatter(
            selected_recall,
            selected_precision,
            s=85,
            color=line.get_color(),
            edgecolors="white",
            linewidths=1.5,
            zorder=4,
            label=f"{name} at frozen threshold",
        )

        print(f"\n{name}")
        print(f"Average precision: {average_precision:.4f}")
        print(f"Precision at threshold: {selected_precision:.2%}")
        print(f"Recall at threshold: {selected_recall:.2%}")
        print(f"Failures detected: {true_positives}")

    ax.set_title(
        "Random Forest — Precision-Recall Performance",
        fontsize=15,
        pad=16,
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)

    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()

    output_path = REPORT_DIR / "precision_recall_comparison.png"

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nChart saved: {output_path}")
    print("Model and decision threshold remain unchanged.")
    print("Plot generation completed successfully.")


if __name__ == "__main__":
    plot_precision_recall()
