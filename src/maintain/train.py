
"""Train baseline classifiers on the maintenance benchmark."""

import json
from hashlib import sha256

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from maintain.data import (
    DATA_PATH,
    MODEL_FEATURES,
    PROJECT_ROOT,
    TARGET_COLUMN,
)
from maintain.features import select_training_features


REPORT_DIR = PROJECT_ROOT / "artifacts" / "example"
MODEL_DIR = PROJECT_ROOT / "models"


def load_training_blocks():
    """Load training and validation rows from our saved split."""

    data = pd.read_csv(DATA_PATH)
    manifest = pd.read_csv(REPORT_DIR / "split_manifest.csv")

    with (REPORT_DIR / "split_metadata.json").open(
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    # Stop if the source file differs from the one we inspected.
    current_hash = sha256(DATA_PATH.read_bytes()).hexdigest()

    if current_hash != metadata["sha256"]:
        raise ValueError("Dataset checksum does not match the split metadata.")

    if len(data) != len(manifest):
        raise ValueError("Dataset and split manifest have different lengths.")

    if manifest["row_index"].tolist() != list(range(len(data))):
        raise ValueError("Split manifest row positions have changed.")

    if not data["UDI"].reset_index(drop=True).equals(
        manifest["UDI"].reset_index(drop=True)
    ):
        raise ValueError("Dataset identifiers do not match the split manifest.")

    expected_splits = (
        ["train"] * 6000
        + ["validation"] * 2000
        + ["test"] * 2000
    )

    if manifest["split"].tolist() != expected_splits:
        raise ValueError("Split assignments differ from the frozen protocol.")

    # Only training and validation records are used in this stage.
    train_data = data.iloc[:6000].copy()
    validation_data = data.iloc[6000:8000].copy()

    return train_data, validation_data


def make_logistic_pipeline():
    """Build preprocessing and logistic regression as one pipeline."""

    numeric_features = [
        feature for feature in MODEL_FEATURES
        if feature != "Type"
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        (
            "type",
            OneHotEncoder(handle_unknown="ignore"),
            ["Type"],
        ),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
        ),
    ])


def make_random_forest_pipeline():
    """Build a small tree ensemble with training-only preprocessing."""

    numeric_features = [
        feature for feature in MODEL_FEATURES
        if feature != "Type"
    ]

    # Trees do not require standard scaling, but missing-value
    # handling and category encoding stay inside the pipeline.
    preprocessing = ColumnTransformer([
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_features,
        ),
        (
            "type",
            OneHotEncoder(handle_unknown="ignore"),
            ["Type"],
        ),
    ])

    forest = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", forest),
    ])


def train_baselines():
    """Fit three models and compare their validation performance."""

    train_data, validation_data = load_training_blocks()

    x_train = select_training_features(train_data)
    y_train = train_data[TARGET_COLUMN]

    x_validation = select_training_features(validation_data)
    y_validation = validation_data[TARGET_COLUMN]

    models = {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": make_logistic_pipeline(),
        "random_forest": make_random_forest_pipeline(),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    predictions = pd.DataFrame({
        "UDI": validation_data["UDI"].to_numpy(),
        "actual": y_validation.to_numpy(),
    })

    results = []

    print("\nMODEL COMPARISON")
    print("-" * 55)
    print(f"Training records: {len(x_train)}")
    print(f"Validation records: {len(x_validation)}")
    print(f"Validation failures: {int(y_validation.sum())}")

    for name, model in models.items():
        # Fit each model using the training block only.
        model.fit(x_train, y_train)

        scores = model.predict_proba(x_validation)[:, 1]
        average_precision = average_precision_score(
            y_validation, scores
        )

        # Keep 0.5 as a reference point, not our final threshold.
        predicted_labels = (scores >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_validation,
            predicted_labels,
            labels=[0, 1],
        ).ravel()

        results.append({
            "model": name,
            "split": "validation",
            "records": len(y_validation),
            "average_precision": average_precision,
            "threshold": 0.5,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        })

        predictions[f"{name}_score"] = scores

        # Keep fitted models locally for the decision experiment.
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")

        print(f"\nModel: {name}")
        print(f"Average precision: {average_precision:.4f}")
        print(f"True positives: {tp}")
        print(f"False positives: {fp}")
        print(f"True negatives: {tn}")
        print(f"False negatives: {fn}")

    pd.DataFrame(results).to_csv(
        REPORT_DIR / "baseline_metrics.csv",
        index=False,
    )

    predictions.to_csv(
        REPORT_DIR / "validation_predictions.csv",
        index=False,
    )

    print("\nSaved validation results and local model files.")
    print("The test block has not been evaluated.")
    print("\nModel comparison completed successfully.")


if __name__ == "__main__":
    train_baselines()
