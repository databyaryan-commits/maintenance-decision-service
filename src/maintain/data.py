
"""Load and inspect the AI4I predictive maintenance dataset."""

from hashlib import sha256
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"

TARGET_COLUMN = "Machine failure"

MODEL_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

EXCLUDED_COLUMNS = [
    "UDI",
    "Product ID",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def inspect_dataset():
    """Check the source data before using it for modelling."""

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)

    expected_columns = set(MODEL_FEATURES + EXCLUDED_COLUMNS)
    actual_columns = set(data.columns)

    missing_columns = sorted(expected_columns - actual_columns)
    unexpected_columns = sorted(actual_columns - expected_columns)

    if missing_columns or unexpected_columns:
        raise ValueError(
            f"Dataset schema mismatch. "
            f"Missing: {missing_columns}. "
            f"Unexpected: {unexpected_columns}."
        )

    if len(data) != 10_000:
        raise ValueError(f"Expected 10,000 rows, found {len(data)}.")

    missing_values = int(data.isna().sum().sum())

    if missing_values:
        raise ValueError(f"Dataset contains {missing_values} missing values.")

    if data["UDI"].duplicated().any():
        raise ValueError("Duplicate row identifiers found.")

    if not set(data[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("Machine failure must contain only 0 and 1.")

    if not set(data["Type"].unique()).issubset({"L", "M", "H"}):
        raise ValueError("Unexpected product type found.")

    # Record the exact bytes of the downloaded CSV.
    checksum = sha256(DATA_PATH.read_bytes()).hexdigest()

    print("\nDATASET INSPECTION")
    print("-" * 50)
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print(f"Missing values: {missing_values}")
    print(f"SHA-256: {checksum}")

    print("\nCOLUMN NAMES")
    for column in data.columns:
        print(f"- {column}")

    print("\nFAILURE DISTRIBUTION")

    # Keep the original row order for the planned block holdout.
    splits = [
        ("Training", 0, 6000),
        ("Validation", 6000, 8000),
        ("Testing", 8000, 10000),
    ]

    for name, start, end in splits:
        block = data.iloc[start:end]
        failures = int(block[TARGET_COLUMN].sum())
        failure_rate = failures / len(block)

        print(
            f"{name}: {len(block)} rows, "
            f"{failures} failures ({failure_rate:.2%})"
        )

    print("\nALLOWED MODEL FEATURES")
    for feature in MODEL_FEATURES:
        print(f"- {feature}")

    print("\nDataset inspection completed successfully.")


if __name__ == "__main__":
    inspect_dataset()
