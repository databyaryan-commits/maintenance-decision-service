
"""Investigate mistakes made by the frozen maintenance model."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from maintain.data import DATA_PATH, MODEL_FEATURES, PROJECT_ROOT
from maintain.frozen import verify_frozen_configuration
from maintain.holdout import load_holdout_data


REPORT_DIR = PROJECT_ROOT / "artifacts" / "example"


def summarise_groups(data, column):
    """Count correct and incorrect decisions within each group."""

    rows = []

    for name, group in data.groupby(column, observed=True, sort=True):
        failures = int(group["actual"].sum())
        normal = len(group) - failures

        true_positive = int((group["outcome"] == "TP").sum())
        false_positive = int((group["outcome"] == "FP").sum())
        false_negative = int((group["outcome"] == "FN").sum())

        rows.append({
            column: str(name),
            "records": len(group),
            "actual_failures": failures,
            "true_positives": true_positive,
            "false_positives": false_positive,
            "false_negatives": false_negative,
            "failure_recall": (
                true_positive / failures if failures else np.nan
            ),
            "false_positive_rate": (
                false_positive / normal if normal else np.nan
            ),
        })

    return pd.DataFrame(rows)


def save_confusion_matrix(data):
    """Create a labelled confusion matrix for the saved decisions."""

    tn = int((data["outcome"] == "TN").sum())
    fp = int((data["outcome"] == "FP").sum())
    fn = int((data["outcome"] == "FN").sum())
    tp = int((data["outcome"] == "TP").sum())

    matrix = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(matrix, cmap="Blues")

    labels = ["Normal", "Failure"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)

    ax.set_xlabel("Predicted decision")
    ax.set_ylabel("Recorded outcome")
    ax.set_title("Frozen Random Forest — Test Confusion Matrix")

    for row in range(2):
        for col in range(2):
            value = matrix[row, col]
            text_colour = (
                "white" if value > matrix.max() / 2 else "black"
            )

            ax.text(
                col,
                row,
                f"{value:,}",
                ha="center",
                va="center",
                fontsize=16,
                color=text_colour,
            )

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR / "holdout_confusion_matrix.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def analyse_holdout_errors():
    """Describe errors without changing the frozen model or threshold."""

    config, _ = verify_frozen_configuration()

    features, actual, identifiers, _ = load_holdout_data(config)

    predictions = pd.read_csv(
        REPORT_DIR / "holdout_predictions.csv"
    )

    if len(predictions) != len(actual):
        raise ValueError("Prediction and test record counts differ.")

    if not np.array_equal(
        predictions["UDI"].to_numpy(), identifiers
    ):
        raise ValueError("Prediction identifiers do not match test rows.")

    if not np.array_equal(
        predictions["actual"].to_numpy(dtype=int), actual
    ):
        raise ValueError("Prediction labels do not match test data.")

    scores = predictions["random_forest_score"].to_numpy(dtype=float)

    if not np.isfinite(scores).all():
        raise ValueError("Non-finite prediction scores found.")

    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Prediction scores are outside [0, 1].")

    threshold = float(config["decision"]["threshold"])
    predicted_failure = scores >= threshold

    expected_actions = np.where(
        predicted_failure, "review", "normal"
    )

    if not np.array_equal(
        predictions["selected_action"].to_numpy(), expected_actions
    ):
        raise ValueError("Saved actions do not match the frozen threshold.")

    # Build a descriptive table without fitting another model.
    records = features.reset_index(drop=True).copy()
    records.insert(0, "UDI", identifiers)

    records["actual"] = actual
    records["score"] = scores

    records["outcome"] = np.select(
        [
            (actual == 1) & predicted_failure,
            (actual == 0) & predicted_failure,
            (actual == 1) & ~predicted_failure,
        ],
        ["TP", "FP", "FN"],
        default="TN",
    )

    counts = records["outcome"].value_counts()

    # Check that the errors agree with our previous holdout report.
    saved_metrics = pd.read_csv(
        REPORT_DIR / "holdout_metrics.csv"
    )

    selected = saved_metrics.loc[
        (saved_metrics["model"] == config["model"]["name"])
        & (saved_metrics["policy"] == "validation_selected")
    ]

    if len(selected) != 1:
        raise ValueError("Expected exactly one selected holdout result.")

    expected_counts = {
        "TP": int(selected.iloc[0]["true_positives"]),
        "FP": int(selected.iloc[0]["false_positives"]),
        "TN": int(selected.iloc[0]["true_negatives"]),
        "FN": int(selected.iloc[0]["false_negatives"]),
    }

    for outcome, expected in expected_counts.items():
        if int(counts.get(outcome, 0)) != expected:
            raise ValueError(
                f"{outcome} count disagrees with holdout_metrics.csv"
            )

    # Report mistakes separately for the three equipment types.
    by_type = summarise_groups(records, "Type")

    by_type.to_csv(
        REPORT_DIR / "errors_by_type.csv",
        index=False,
    )

    # Use training-only quartiles to define feature ranges.
    # These ranges are descriptive and are not used to retune the model.
    numeric_features = [
        name for name in MODEL_FEATURES if name != "Type"
    ]

    training = pd.read_csv(
        DATA_PATH, usecols=numeric_features
    ).iloc[:6000]

    range_tables = []

    for feature in numeric_features:
        quartiles = training[feature].quantile(
            [0.25, 0.5, 0.75]
        ).tolist()

        boundaries = [
            -np.inf,
            *sorted(set(quartiles)),
            np.inf,
        ]

        grouped = records.copy()
        grouped["feature_range"] = pd.cut(
            grouped[feature],
            bins=boundaries,
            include_lowest=True,
        ).astype(str)

        summary = summarise_groups(grouped, "feature_range")
        summary.insert(0, "feature", feature)
        range_tables.append(summary)

    by_range = pd.concat(range_tables, ignore_index=True)

    by_range.to_csv(
        REPORT_DIR / "errors_by_range.csv",
        index=False,
    )

    # Keep all missed failures and ten high-scoring false alarms.
    missed = records.loc[
        records["outcome"] == "FN"
    ].sort_values("score")

    false_alarms = records.loc[
        records["outcome"] == "FP"
    ].sort_values("score", ascending=False).head(10)

    examples = pd.concat(
        [missed, false_alarms],
        ignore_index=True,
    )

    examples[
        ["UDI", "outcome", "actual", "score", *MODEL_FEATURES]
    ].to_csv(
        REPORT_DIR / "error_examples.csv",
        index=False,
    )

    save_confusion_matrix(records)

    print("\nHOLDOUT ERROR ANALYSIS")
    print("-" * 55)

    for name, label in [
        ("TP", "Correctly detected failures"),
        ("FN", "Missed failures"),
        ("FP", "False alarms"),
        ("TN", "Correctly identified normal records"),
    ]:
        print(f"{label}: {counts.get(name, 0)}")

    print("\nERRORS BY EQUIPMENT TYPE")
    print(
        by_type[
            [
                "Type",
                "records",
                "actual_failures",
                "false_positives",
                "false_negatives",
                "failure_recall",
            ]
        ].to_string(index=False)
    )

    if not missed.empty:
        example = missed.iloc[0]

        print("\nEXAMPLE MISSED FAILURE")
        print(f"UDI: {int(example['UDI'])}")
        print(f"Equipment type: {example['Type']}")
        print(f"Failure score: {example['score']:.6f}")
        print(f"Frozen threshold: {threshold:.6f}")

    print("\nSaved files:")
    print("- errors_by_type.csv")
    print("- errors_by_range.csv")
    print("- error_examples.csv")
    print("- holdout_confusion_matrix.png")

    print("\nNo model settings were changed.")
    print("Error analysis completed successfully.")


if __name__ == "__main__":
    analyse_holdout_errors()
