
"""Create reproducible train, validation and test splits."""

import json
from hashlib import sha256

import pandas as pd

from maintain.data import (
    DATA_PATH,
    MODEL_FEATURES,
    PROJECT_ROOT,
    TARGET_COLUMN,
)


OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "example"

# Keep these boundaries fixed throughout the experiment.
SPLIT_RANGES = {
    "train": (0, 6000),
    "validation": (6000, 8000),
    "test": (8000, 10000),
}


def create_split_manifest():
    """Save the original row-order split and its metadata."""

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)

    if len(data) != 10_000:
        raise ValueError("Expected exactly 10,000 observations.")

    if not {"UDI", TARGET_COLUMN}.issubset(data.columns):
        raise ValueError("Required identifier or target column is missing.")

    if not set(MODEL_FEATURES).issubset(data.columns):
        raise ValueError("One or more model input features are missing.")

    if data["UDI"].isna().any() or data["UDI"].duplicated().any():
        raise ValueError("Each observation must have a unique UDI.")

    if not data[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Machine failure must contain only 0 and 1.")

    # Save the exact identity of the source file.
    dataset_hash = sha256(DATA_PATH.read_bytes()).hexdigest()

    split_labels = []
    split_summary = {}

    for split_name, (start, end) in SPLIT_RANGES.items():
        block = data.iloc[start:end]
        failure_count = int(block[TARGET_COLUMN].sum())

        split_labels.extend([split_name] * len(block))

        split_summary[split_name] = {
            "start_index": start,
            "end_index_exclusive": end,
            "records": len(block),
            "failures": failure_count,
        }

    # UDI is stored for auditing, never used as a model input.
    manifest = pd.DataFrame({
        "row_index": range(len(data)),
        "UDI": data["UDI"].to_numpy(),
        "split": split_labels,
    })

    metadata = {
        "dataset": "UCI AI4I 2020",
        "source_file": "data/raw/ai4i2020.csv",
        "sha256": dataset_hash,
        "split_method": "Original row-order block holdout",
        "is_chronological": False,
        "target": TARGET_COLUMN,
        "model_features": MODEL_FEATURES,
        "splits": split_summary,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest.to_csv(
        OUTPUT_DIR / "split_manifest.csv",
        index=False,
    )

    with (OUTPUT_DIR / "split_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    print("\nSPLIT MANIFEST CREATED")
    print("-" * 50)

    for name, details in split_summary.items():
        print(
            f"{name}: {details['records']} records, "
            f"{details['failures']} failures"
        )

    print(f"\nDataset SHA-256: {dataset_hash}")
    print(f"Files saved in: {OUTPUT_DIR}")
    print("\nSplit creation completed successfully.")


if __name__ == "__main__":
    create_split_manifest()
