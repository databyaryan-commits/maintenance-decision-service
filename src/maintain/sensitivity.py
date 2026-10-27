
"""Check how assumed error costs affect our validation decision."""

import json
from math import isclose

import pandas as pd

from maintain.data import PROJECT_ROOT


RESULT_DIR = PROJECT_ROOT / "artifacts" / "example"

FALSE_POSITIVE_COST = 10
FALSE_NEGATIVE_COSTS = [100, 500, 1000]


def find_policy(results, model_name, policy_name):
    """Find the recorded validation results for one policy."""

    matching = results.loc[
        (results["model"] == model_name)
        & (results["policy"] == policy_name)
    ]

    if len(matching) != 1:
        raise ValueError(
            f"Expected one result for {model_name}: {policy_name}"
        )

    return matching.iloc[0]


def calculate_penalty(result, false_negative_cost):
    """Apply assumed costs to the recorded classification errors."""

    false_positives = int(result["false_positives"])
    false_negatives = int(result["false_negatives"])

    return (
        FALSE_POSITIVE_COST * false_positives
        + false_negative_cost * false_negatives
    )


def compare_penalty_scenarios():
    """Compare the frozen primary policy against threshold 0.5."""

    results = pd.read_csv(
        RESULT_DIR / "validation_decisions.csv"
    )

    with (RESULT_DIR / "validation_selection.json").open(
        encoding="utf-8"
    ) as file:
        selection = json.load(file)

    if selection["selection_stage"] != "validation_only":
        raise ValueError("Expected a validation-only selection.")

    if selection["test_evaluated"] is not False:
        raise ValueError("The test block must remain untouched.")

    if (
        selection["false_positive_penalty"] != 10
        or selection["false_negative_penalty"] != 500
    ):
        raise ValueError("Primary penalty assumptions have changed.")

    model_name = selection["selected_model"]

    default = find_policy(results, model_name, "default_0.5")
    selected = find_policy(
        results, model_name, "validation_selected"
    )

    # Use the full saved threshold, not the rounded terminal display.
    threshold = float(selection["selected_threshold"])

    if not isclose(
        float(selected["threshold"]),
        threshold,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("Selected threshold does not match the results.")

    for policy in (default, selected):
        total = sum(
            int(policy[column])
            for column in [
                "true_positives",
                "false_positives",
                "true_negatives",
                "false_negatives",
            ]
        )

        if total != 2000:
            raise ValueError("Unexpected validation record count.")

    comparison = []

    print("\nPENALTY SENSITIVITY ANALYSIS")
    print("-" * 55)
    print(f"Selected model: {model_name}")
    print(f"Selected threshold: {threshold:.8f}")
    print(f"False-positive penalty: {FALSE_POSITIVE_COST}")

    for false_negative_cost in FALSE_NEGATIVE_COSTS:
        default_penalty = calculate_penalty(
            default, false_negative_cost
        )

        selected_penalty = calculate_penalty(
            selected, false_negative_cost
        )

        difference = default_penalty - selected_penalty

        comparison.append({
            "model": model_name,
            "false_positive_cost": FALSE_POSITIVE_COST,
            "false_negative_cost": false_negative_cost,
            "default_threshold": 0.5,
            "selected_threshold": threshold,
            "default_penalty": default_penalty,
            "selected_penalty": selected_penalty,
            "penalty_reduction": difference,
        })

        print(f"\nFalse-negative penalty: {false_negative_cost}")
        print(f"Default threshold penalty: {default_penalty}")
        print(f"Selected threshold penalty: {selected_penalty}")
        print(f"Penalty reduction: {difference}")

    pd.DataFrame(comparison).to_csv(
        RESULT_DIR / "penalty_sensitivity.csv",
        index=False,
    )

    print("\nResults saved to penalty_sensitivity.csv")
    print("Primary model and threshold remain unchanged.")
    print("The test block has not been evaluated.")


if __name__ == "__main__":
    compare_penalty_scenarios()
