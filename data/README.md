
# Dataset

## Source

This project uses the AI4I 2020 Predictive Maintenance Dataset from the UCI Machine Learning Repository.

- Dataset: AI4I 2020 Predictive Maintenance Dataset
- Source: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- DOI: https://doi.org/10.24432/C5HS5C
- License: Creative Commons Attribution 4.0 (CC BY 4.0)
- Expected records: 10,000
- Download date: 2026-10-27
- SHA-256 checksum: dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e

The data is synthetic. It was designed to resemble industrial maintenance observations, but it does not represent actual equipment operated by a particular company.

## Prediction task

The goal is to classify whether a machine failure is recorded for an observation.

Target: `Machine failure`

- 0: No machine failure recorded
- 1: Machine failure recorded

This is a classification problem, not a forecast of when a machine will fail. The data does not establish a future prediction window.

## Allowed model inputs

Only the following six source fields may be used:

| Field | Description | Unit |
|---|---|---|
| Type | Product quality variant (L, M or H) | Category |
| Air temperature | Ambient air temperature | Kelvin |
| Process temperature | Operating process temperature | Kelvin |
| Rotational speed | Rotational speed | rpm |
| Torque | Applied torque | Nm |
| Tool wear | Accumulated tool wear | minutes |

Any encoding or scaling will be fitted using training data only.

## Excluded fields

The following fields must never enter the model as predictors:

- UDI: Unique row identifier, excluded from model inputs
- Product ID: Product identifier
- Machine failure: Prediction target
- TWF: Tool wear failure label
- HDF: Heat dissipation failure label
- PWF: Power failure label
- OSF: Overstrain failure label
- RNF: Random failure label

The five failure-mode labels directly describe the outcome we are trying to predict. Including them would leak target information into the model.

## Evaluation plan

We will preserve the dataset's original row order and use three non-overlapping blocks:

- Training: First 60% (6,000 rows)
- Validation: Next 20% (2,000 rows)
- Testing: Final 20% (2,000 rows)

We will check the failure counts in each block before training and save the exact split membership.

This is a row-order block holdout, not a verified chronological split.

The test block will remain untouched during model and threshold selection.


## Verified dataset inspection

The dataset was inspected before model development.

| Check | Result |
|---|---|
| Total observations | 10,000 |
| Source columns | 14 |
| Allowed predictor columns | 6 |
| Missing values | 0 |
| Recorded machine failures | 339 |
| Overall failure rate | 3.39% |

### Block-level failure distribution

| Block | Records | Failures | Failure rate |
|---|---:|---:|---:|
| Training | 6,000 | 255 | 4.25% |
| Validation | 2,000 | 45 | 2.25% |
| Test | 2,000 | 39 | 1.95% |

The failure rate differs between blocks. We will preserve
the original row-order split and report these differences
when interpreting model results.

The test block will not be used to select model settings
or decision thresholds.

Exact row membership is saved in
`artifacts/example/split_manifest.csv`, with supporting
metadata in `artifacts/example/split_metadata.json`.


## Decision costs

For the primary experiment, we will assume:

- False positive: 10 penalty units
- False negative: 500 penalty units

We will also examine false-negative penalties of 100 and 1,000 units.

These values are assumptions for comparing decision policies. They are not measured maintenance costs or financial savings.

## Limitations

The dataset is synthetic and contains relatively few failure observations.

Results cannot establish real-world reliability, downtime prevention or maintenance savings without further validation on operational data.

The model's role in this project is to demonstrate reproducible classification, evaluation and decision-making.
