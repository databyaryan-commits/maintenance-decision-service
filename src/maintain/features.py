
"""Keep model inputs separate from identifiers and failure labels."""

import pandas as pd

from maintain.data import MODEL_FEATURES


def select_training_features(source: pd.DataFrame) -> pd.DataFrame:
    """Extract only approved features from the original dataset."""

    missing = [
        name for name in MODEL_FEATURES
        if name not in source.columns
    ]

    if missing:
        raise ValueError(f"Missing model features: {missing}")

    # The source also contains identifiers, targets and failure modes.
    # None of those columns should reach the fitted model.
    return source.loc[:, MODEL_FEATURES].copy()


def validate_prediction_features(values: pd.DataFrame) -> pd.DataFrame:
    """Reject prediction inputs that do not match our feature contract."""

    missing = [
        name for name in MODEL_FEATURES
        if name not in values.columns
    ]

    unexpected = [
        name for name in values.columns
        if name not in MODEL_FEATURES
    ]

    if missing or unexpected:
        raise ValueError(
            f"Invalid prediction features. "
            f"Missing: {missing}. "
            f"Unexpected: {unexpected}."
        )

    return values.loc[:, MODEL_FEATURES].copy()
