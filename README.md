# Cellular Network Anomaly Detection Pipeline

An end-to-end Python pipeline that extracts, cleans, and analyzes real cellular network KPI data, then applies unsupervised machine learning (Isolation Forest) to detect anomalous cell behavior — with a full investigation into *why* the initial model underperformed and what that reveals about the dataset itself.

## Project Overview

This project processes real eNodeB (cell site) performance data — PRB usage, throughput, and connected user counts — from a public Kaggle competition dataset, and attempts to detect cells flagged as "Unusual" using an unsupervised anomaly detection model, without using the label during training.

Rather than stopping at a single model run, this project documents an iterative investigation: several domain-informed hypotheses were tested, evaluated against the ground-truth label, and either confirmed or ruled out with evidence.

## Data Source

[Anomaly detection in cellular networks](https://www.kaggle.com/c/anomaly-detection-in-cellular-networks) (Kaggle competition).

Raw data files are not included in this repository (see `.gitignore`) due to competition redistribution terms. To reproduce this pipeline:
1. Join the competition on Kaggle and download the data
2. Place the training CSV at `data/raw/cell_kpi_train.csv`

## Pipeline Architecture

```
src/
├── etl.py            # Extract: loads raw CSV (handles non-UTF-8 encoding)
├── transform.py      # Transform: cleans data, engineers Site/Sector/Time features
├── load.py           # Load: writes clean data to a SQLite database
├── main_pipeline.py  # Orchestrates extract -> transform -> load
└── model.py          # Unsupervised anomaly detection + investigation

### Transform step highlights
- Fixed a Spanish-locale Excel error string (`#¡VALOR!`) found in the `maxUE_UL+DL` column, using `pd.to_numeric(errors='coerce')`
- Dropped ~89 rows (0.24%) with missing user-equipment counts
- Extracted `Site` and `Sector` from `CellName` (e.g., `10BLTE` -> Site 10, Sector B) via regex
- Converted `Time` to `Time_Minutes` (minutes since midnight) for numeric use

## Modeling: Isolation Forest

An Isolation Forest was trained on 12 KPI features (PRB usage, throughput, user counts, time), without using the `Unusual` label during training. The model's Normal/Unusual predictions were then compared, row by row, against that label purely for evaluation — checking whether the model's own statistical notion of "outlier" aligned with the dataset's definition of "anomalous."

**Baseline result** (`contamination='auto'`, scikit-learn's default assumption of a low, sparse anomaly rate): recall of just 0.07 on the "Unusual" class. The default assumption didn't match this dataset's actual anomaly rate of 27.6% (10,167 of 36,815 rows), so the model only flagged the most extreme statistical outliers — a far smaller group than the true proportion of labeled anomalies.

## Investigation: Why the Model Underperformed

Five modeling attempts were run in sequence, each testing a specific hypothesis:

| # | Hypothesis | Approach | Result (precision / recall, Unusual class) |
|---|---|---|---|
| 1 | Default contamination assumption is too conservative | `contamination='auto'` on raw features | 0.17 / 0.07 |
| 2 | Matching contamination to the real ~27.6% anomaly rate would let the model flag the right proportion of rows | `contamination=0.276` on raw features | 0.20 / 0.20 |
| 3 | Anomalies may be specific to a physical site | One-hot encoded `Site` added to raw features | 0.20 / 0.20 (no change) |
| 4 | Anomalies are unusual relative to each site's own normal range, not the dataset-wide average | Per-site z-scored KPI features | 0.19 / 0.19 (no improvement) |
| 5 | Congestion (high resource usage + low throughput) drives anomalies | Engineered throughput-per-PRB-usage efficiency ratios | 0.20 / 0.20 (no improvement) |

None of attempts 2-5 meaningfully improved on each other — a signal that the plateau wasn't caused by *which* feature transformation was used, but by something more fundamental about the label or the available data itself.

### Direct data investigation

With modeling attempts plateaued, the investigation turned to the raw training data and its label directly, without any model involved:

- **Normal vs. Unusual KPI averages**: Unusual rows tend to have *lower* PRB usage, throughput, and user counts than Normal rows — the opposite of a congestion signature, and more consistent with degraded/faulty radio conditions serving fewer users. However, the gap was modest (roughly 15-25%) with substantial overlap between the two classes on every feature.
- **By hour of day**: the Unusual rate stayed within a narrow 25-29% band across every single hour, including after excluding early-morning low-traffic hours — ruling out a time-of-day or peak-load explanation.
- **By cell**: the Unusual rate stayed within an equally narrow 25.3-28.9% band across all 33 cells — ruling out the idea that a handful of "chronically faulty" cells were driving the pattern.
- **Time structure**: the `Time` column contains only clock time (e.g., `"9:45"`), with no date component. Counting rows per `CellName + Time` combination showed each combination repeats 5-15 times; dividing each cell's total row count by 96 (15-minute intervals per day) consistently lands around 11-12 for every cell — indicating this dataset spans roughly 11-12 days, folded together with date information removed.

### Conclusion

The "Unusual" label is applied at a very consistent ~27% rate across every hour and every individual cell, with only modest, heavily overlapping KPI differences between classes. Because the dataset's `Time` column has no date, any day-specific context that might explain the label — such as a multi-day fault episode, or a comparison against the same hour on a different day — is not recoverable from the data as provided.

This combination of findings is **consistent with, though not proof of**, a synthetic or statistically injected anomaly label — a common practice in benchmark datasets, where a fixed proportion of rows are artificially flagged for model evaluation purposes, independent of the underlying KPI values in a way the available features can fully capture. It would explain why no amount of domain-informed feature engineering — contamination tuning, per-site context, or congestion ratios — could substantially close the gap. A firm conclusion on the label's origin isn't possible with this data alone; this is the best-supported interpretation given the evidence gathered, not a confirmed mechanism.

## What This Project Demonstrates

- A complete, working ETL pipeline handling real-world data quality issues (encoding errors, malformed values, missing data)
- Applied unsupervised machine learning with honest evaluation against ground truth
- A structured, evidence-based investigation: forming hypotheses, testing them, and correctly ruling out incorrect explanations rather than overfitting a narrative to a convenient result
- Clear distinction throughout between what was directly confirmed by evidence and what remains a well-supported but unproven interpretation
- Domain expertise applied to a data science problem (RF/network engineering background informing hypotheses about congestion, per-site baselines, and cell-level fault behavior)

## Tech Stack
Python, pandas, scikit-learn, SQLite, regex

## Next to investigate
- Investigate whether "Unusual" correlates with specific value combinations not yet tested — e.g., a distance/density-based clustering method (k-means, DBSCAN) computes anomalies differently than Isolation Forest's isolation-based approach and might surface a different pattern
- Try alternative unsupervised methods (e.g., Gaussian/density-based anomaly detection, or a simple autoencoder) to see whether a different model architecture captures the label differently
- Cross-reference against the competition's leaderboard/discussion forum for insight into how the label was originally generated