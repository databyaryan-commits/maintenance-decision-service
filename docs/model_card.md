# Model Card — Maintenance Decision Service

## 1. Overview

Maintenance Decision Service is an independent machine learning project that classifies equipment observations and recommends whether they should be flagged for review.

The project uses the synthetic AI4I 2020 Predictive Maintenance Dataset from the UCI Machine Learning Repository.

The main objective is to demonstrate how model evaluation, class imbalance, decision thresholds and assumed error costs influence classification decisions.

The system compares a dummy baseline, logistic regression and Random Forest. The selected Random Forest is exposed through a local FastAPI service using the same fitted preprocessing pipeline and frozen decision threshold used during evaluation.

**Project version:** 0.1.0 (development)

**Selected model:** Random Forest Classifier

**Task:** Binary classification

**Current deployment:** Local API prototype

**Status:** Model evaluation and API implementation complete; final reproducibility and release checks pending.

## 2. Dataset

**Source:** [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

**DOI:** https://doi.org/10.24432/C5HS5C

**License:** Creative Commons Attribution 4.0 (CC BY 4.0)

The dataset contains 10,000 synthetic equipment observations and 14 source columns.

The target is `Machine failure`:

- `0`: No machine failure recorded
- `1`: Machine failure recorded

There are 339 recorded failures across the dataset, giving an overall positive-class prevalence of 3.39%.

The dataset is synthetic and does not represent the maintenance history of an actual industrial site.

### Input features

Only six fields are used for prediction:

| Feature | Unit or values |
|---|---|
| Type | L, M, H |
| Air temperature | Kelvin |
| Process temperature | Kelvin |
| Rotational speed | rpm |
| Torque | Nm |
| Tool wear | Minutes |

The following columns are explicitly excluded from model inputs:

`UDI`, `Product ID`, `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF` and `RNF`.

The exclusion prevents target-derived labels and identifiers from entering the prediction pipeline.

## 3. Evaluation protocol

The dataset was divided into contiguous blocks based on its original row order.

| Split | Observations | Failures | Failure rate |
|---|---:|---:|---:|
| Training | 6,000 | 255 | 4.25% |
| Validation | 2,000 | 45 | 2.25% |
| Test | 2,000 | 39 | 1.95% |

This is a row-order block holdout, not a verified chronological split.

Training data was used to fit models and preprocessing transformations.

Validation data was used to compare models and select the operating threshold.

The final test block was evaluated after the model and threshold had been frozen. Test results were not used to change the selected policy.

Split membership and dataset checksums are recorded in `artifacts/example/`.

## 4. Models compared

Three classifiers were evaluated:

1. **DummyClassifier:** Predicts using the training class prevalence and provides a baseline.
2. **Logistic Regression:** Uses training-fitted preprocessing, numeric scaling and one-hot encoding.
3. **Random Forest:** Uses training-fitted preprocessing and a small tree ensemble.

The Random Forest was configured with:

| Parameter | Value |
|---|---|
| Number of trees | 150 |
| Maximum depth | 8 |
| Minimum samples per leaf | 5 |
| Class weighting | balanced_subsample |
| Random state | 42 |
| Parallel jobs | 1 |

Numeric features are median-imputed and equipment type is one-hot encoded inside the fitted pipeline.

The same fitted pipeline is used for evaluation and API prediction.

### Validation model comparison

| Model | Average precision |
|---|---:|
| Dummy | 0.0225 |
| Logistic Regression | 0.4161 |
| Random Forest | 0.5442 |

The Random Forest achieved the highest validation average precision.

Average precision measures ranking quality across the precision-recall relationship. It does not directly measure performance at one decision threshold.

## 5. Decision threshold selection

The primary decision experiment assigns assumed penalties of:

- False positive: 10 units
- False negative: 500 units

For each model, candidate thresholds were compared on the validation block.

The threshold was selected by minimising:

`Total penalty = (10 × FP) + (500 × FN)`

The selected policy was:

- **Model:** Random Forest
- **Threshold:** 0.2836220952642361
- **Decision rule:** Flag for review when score is greater than or equal to the threshold.

The full-precision threshold is stored in `configs/decision.yaml`.

### Validation results

| Metric | Default threshold 0.5 | Selected threshold |
|---|---:|---:|
| True positives | 29 | 42 |
| False positives | 30 | 153 |
| False negatives | 16 | 3 |
| Recall | 64.44% | 93.33% |
| Precision | 49.15% | 21.54% |
| Assumed penalty | 8,300 | 3,030 |

The validation penalty decreased by approximately 63.5% under the primary cost assumptions.

However, the selected threshold substantially increased the number of false alarms.

These penalty values are hypothetical classification costs, not measured business expenses or financial savings.

### Penalty sensitivity

The same selected threshold was evaluated under alternative false-negative penalties.

| False-negative penalty | Default penalty | Selected penalty |
|---|---:|---:|
| 100 | 1,900 | 1,830 |
| 500 | 8,300 | 3,030 |
| 1,000 | 16,300 | 4,530 |

The selected threshold remained unchanged during this comparison.

These results show how strongly the apparent benefit of a policy can depend on the assumed consequences of missed failures.

They do not establish the optimal threshold for the alternative penalty scenarios.

## 6. Final holdout results

The frozen Random Forest was evaluated on 2,000 held-out observations containing 39 recorded failures.

### Predictive performance

| Metric | Test result |
|---|---:|
| Average precision | 0.6764 |
| Brier score | 0.0189 |
| Recall | 82.05% |
| Precision | 26.02% |
| False-positive rate | 4.64% |
| True positives | 32 |
| False positives | 91 |
| True negatives | 1,870 |
| False negatives | 7 |

### Decision-policy comparison

| Policy | False positives | False negatives | Penalty units |
|---|---:|---:|---:|
| Dummy baseline | 0 | 39 | 19,500 |
| Random Forest, threshold 0.5 | 22 | 13 | 6,720 |
| Random Forest, selected threshold | 91 | 7 | 4,410 |

The selected threshold reduced the test penalty by approximately 34.4% compared with the default 0.5 threshold.

This improvement came with more false alarms: 91 compared with 22.

The test recall was lower than validation recall, although average precision increased.

These results should be interpreted cautiously because the test set contains only 39 failure cases.

### Confusion matrix

![Frozen model confusion matrix](../artifacts/example/holdout_confusion_matrix.png)

## 7. Error analysis

The selected policy missed seven failures and flagged 91 normal observations for review.

### Errors by equipment type

| Type | Observations | Failures | Missed failures | Recall |
|---|---:|---:|---:|---:|
| H | 221 | 0 | 0 | Not defined |
| L | 1,184 | 32 | 6 | 81.25% |
| M | 595 | 7 | 1 | 85.71% |

Six of the seven missed failures occurred in Type L observations.

However, Type L also accounts for most failures in the test block, so this does not by itself establish that the model is less reliable for Type L.

There were no recorded failures among Type H observations in the test set. Failure recall for that category cannot be assessed from this holdout.

### Example missed failure

One missed failure involved observation UDI 8507.

- Equipment type: L
- Model score: 0.030423
- Selected threshold: approximately 0.283622
- Actual recorded outcome: Failure
- Predicted decision: Normal

The model assigned this observation a score well below the review threshold.

This is a concrete example of a false negative. The result does not establish the physical cause of the failure or explain why the model assigned a low score.

Additional error examples and feature-range summaries are available in `artifacts/example/`.

## 8. Prediction service

The fitted model is served using FastAPI.

Three endpoints are implemented:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Check model availability |
| `GET /model-info` | Retrieve model and configuration metadata |
| `POST /predict` | Evaluate equipment measurements |

The prediction endpoint accepts the six documented input fields.

Invalid categories, missing measurements, unexpected fields and malformed numerical values are rejected.

A successful response contains:

- Model score
- Frozen threshold
- Review or normal action
- Model version
- Short explanation of the decision rule

The explanation describes how the score is compared with the threshold. It is not a causal explanation of equipment failure.

API and direct batch predictions have been checked using two observations with agreement within an absolute tolerance of `1e-12`.

This is a local engineering prototype, not a production deployment.

## 9. Verification and reproducibility

The project currently includes 52 passing automated tests.

Checks cover:

- Feature leakage exclusions
- Missing and unexpected input fields
- Frozen configuration integrity
- Trained-model checksum validation
- Threshold decision boundaries
- Holdout calculation consistency
- Error-analysis reporting
- API request validation
- API and batch scoring consistency

The trained model artifact is excluded from Git history. It must be generated locally from the documented dataset and training pipeline.

The frozen model checksum identifies the specific artifact used for the reported evaluation.

The repository's clean-clone reproduction workflow, automated CI checks and final release instructions are still being completed.

### Reference files

- `configs/decision.yaml` — Selected model configuration
- `artifacts/example/split_metadata.json` — Dataset and split metadata
- `artifacts/example/validation_decisions.csv` — Validation policy comparison
- `artifacts/example/holdout_metrics.csv` — Final test results
- `artifacts/example/holdout_predictions.csv` — Saved test predictions
- `artifacts/example/errors_by_type.csv` — Type-level error analysis
- `artifacts/example/errors_by_range.csv` — Feature-range error analysis

## 10. Limitations and intended use

This model is intended for educational benchmarking, experimental evaluation and demonstration of machine learning engineering practices.

It is not suitable for making autonomous maintenance decisions about real equipment.

Key limitations include:

- The underlying dataset is synthetic.
- The records do not establish a future failure-prediction horizon.
- The row-order split is not confirmed to represent deployment chronology.
- The positive class is rare, especially in the test block.
- Reported performance may not generalise to real industrial systems.
- Classification penalties are assumptions rather than measured operating costs.
- The model's numerical scores have not been established as calibrated probabilities.
- The current system does not include production monitoring, drift detection or operational safety validation.

The reported Brier score alone is not sufficient to establish reliable probability calibration.

Before considering real-world use, the system would require representative operational data, a clearly defined prediction horizon, independently verified failure labels, realistic intervention costs, calibration assessment, external validation and appropriate safety oversight.

No real downtime reductions, financial savings or maintenance benefits are claimed.

## 11. Conclusion

The project demonstrates how model selection and operating thresholds can lead to different classification outcomes.

The Random Forest achieved stronger validation ranking performance than the two baseline models, and the validation-selected threshold reduced assumed penalties on the frozen test block compared with a default threshold of 0.5.

The resulting policy also generated more false alarms and missed seven recorded failures.

Those trade-offs, along with the model's limitations, are part of the evaluation rather than results to hide.

The main contribution is a reproducible classification and decision-making workflow supported by documented model selection, explicit assumptions, automated checks and a validated local prediction API.