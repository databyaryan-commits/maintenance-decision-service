
"""Compare validation thresholds using assumed classification costs."""

import json

import numpy as np
import pandas as pd

from maintain.data import PROJECT_ROOT


RESULT_DIR = PROJECT_ROOT / "artifacts" / "example"

FALSE_POSITIVE_COST = 10
FALSE_NEGATIVE_COST = 500

MODEL_NAMES = [
    "dummy",
    "logistic_regression",
    "random_forest",
]


def calculate_decision_metrics(actual, scores, threshold):
    """Measure the errors and penalty at one operating threshold."""

    review = scores >= threshold
    failures = actual == 1

    true_positives = int(np.sum(review & failures))
    false_positives = int(np.sum(review & ~failures))
    true_negatives = int(np.sum(~review & ~failures))
    false_negatives = int(np.sum(~review & failures))

    predicted_reviews = true_positives + false_positives
    actual_failures = true_positives + false_negatives
    actual_normal = true_negatives + false_positives

    precision = (
        true_positives / predicted_reviews
        if predicted_reviews else 0.0
    )

    recall = (
        true_positives / actual_failures
        if actual_failures else 0.0
    )

    false_positive_rate = (
        false_positives / actual_normal
        if actual_normal else 0.0
    )

    penalty = (
        FALSE_POSITIVE_COST * false_positives
        + FALSE_NEGATIVE_COST * false_negatives
    )

    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "penalty_units": penalty,
    }


def choose_validation_threshold(actual, scores):
    """Find the lowest-penalty threshold using validation data."""

    # Predictions only change when we cross an observed score.
    # Include both boundaries to cover extreme review policies.
    candidates = np.unique(
        np.concatenate([scores, [0.0, 1.0]])
    )

    outcomes = [
        calculate_decision_metrics(actual, scores, threshold)
        for threshold in candidates
    ]

    # Prefer fewer false alarms, then a higher threshold,
    # whenever multiple thresholds have the same penalty.
    return min(
        outcomes,
        key=lambda result: (
            result["penalty_units"],
            result["false_positives"],
            -result["threshold"],
        ),
    )


def evaluate_validation_decisions():
    """Compare models and select a validation-only decision policy."""

    predictions_path = RESULT_DIR / "validation_predictions.csv"
    manifest_path = RESULT_DIR / "split_manifest.csv"

    predictions = pd.read_csv(predictions_path)
    manifest = pd.read_csv(manifest_path)

    validation_ids = (
        manifest.loc[manifest["split"] == "validation", "UDI"]
        .reset_index(drop=True)
    )

    if len(predictions) != 2000:
        raise ValueError("Expected 2,000 validation predictions.")

    if not predictions["UDI"].reset_index(drop=True).equals(
        validation_ids
    ):
        raise ValueError("Validation prediction IDs do not match the split.")

    if not predictions["actual"].isin([0, 1]).all():
        raise ValueError("Validation labels must be binary.")

    actual = predictions["actual"].to_numpy(dtype=int)

    if int(actual.sum()) != 45:
        raise ValueError("Unexpected validation failure count.")

    comparison_rows = []
    selected_policies = []

    print("\nVALIDATION DECISION COMPARISON")
    print("-" * 55)
    print(f"False positive penalty: {FALSE_POSITIVE_COST}")
    print(f"False negative penalty: {FALSE_NEGATIVE_COST}")

    for model_name in MODEL_NAMES:
        score_column = f"{model_name}_score"

        if score_column not in predictions.columns:
            raise ValueError(f"Missing predictions: {score_column}")

        scores = predictions[score_column].to_numpy(dtype=float)

        if not np.isfinite(scores).all():
            raise ValueError(f"Invalid scores from {model_name}.")

        if ((scores < 0) | (scores > 1)).any():
            raise ValueError(f"Scores outside [0, 1]: {model_name}")

        default_result = calculate_decision_metrics(
            actual, scores, threshold=0.5
        )

        selected_result = choose_validation_threshold(
            actual, scores
        )

        for policy_name, metrics in [
            ("default_0.5", default_result),
            ("validation_selected", selected_result),
        ]:
            comparison_rows.append({
                "model": model_name,
                "policy": policy_name,
                **metrics,
            })

        selected_policies.append({
            "model": model_name,
            **selected_result,
        })

        print(f"\nModel: {model_name}")
        print(f"Default 0.5 penalty: {default_result['penalty_units']}")
        print(f"Selected threshold: {selected_result['threshold']:.8f}")
        print(f"Selected penalty: {selected_result['penalty_units']}")
        print(f"Recall: {selected_result['recall']:.2%}")
        print(f"False positives: {selected_result['false_positives']}")
        print(f"False negatives: {selected_result['false_negatives']}")

    # Model selection uses the same primary penalty objective.
    # The remaining fields make any ties deterministic.
    winner = min(
        selected_policies,
        key=lambda result: (
            result["penalty_units"],
            result["false_positives"],
            result["model"],
        ),
    )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        RESULT_DIR / "validation_decisions.csv",
        index=False,
    )

    selection = {
        "selection_stage": "validation_only",
        "status": "pending_review",
        "selected_model": winner["model"],
        "selected_threshold": winner["threshold"],
        "decision_rule": "review if score >= threshold",
        "false_positive_penalty": FALSE_POSITIVE_COST,
        "false_negative_penalty": FALSE_NEGATIVE_COST,
        "validation_metrics": {
            key: value
            for key, value in winner.items()
            if key != "model"
        },
        "test_evaluated": False,
    }

    with (RESULT_DIR / "validation_selection.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(selection, file, indent=2)

    print("\nSELECTED VALIDATION POLICY")
    print("-" * 55)
    print(f"Model: {winner['model']}")
    print(f"Threshold: {winner['threshold']:.8f}")
    print(f"Penalty: {winner['penalty_units']}")
    print(f"Recall: {winner['recall']:.2%}")
    print(f"False positive rate: {winner['false_positive_rate']:.2%}")
    print("\nSelection saved for review.")
    print("No test predictions have been evaluated.")


if __name__ == "__main__":
    evaluate_validation_decisions()
