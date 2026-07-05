import sqlite3
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

DB_NAME = "network_kpi_analytics.db"
TABLE_NAME = "cell_kpi_clean"

FEATURE_COLUMNS = [
    'PRBUsageUL', 'PRBUsageDL',
    'meanThr_DL', 'meanThr_UL', 'maxThr_DL', 'maxThr_UL',
    'meanUE_DL', 'meanUE_UL', 'maxUE_DL', 'maxUE_UL', 'maxUE_UL+DL',
    'Time_Minutes'
]


def load_clean_data(db_name=DB_NAME, table_name=TABLE_NAME):
    """Loads the cleaned KPI data from SQLite via a SQL query."""
    conn = sqlite3.connect(db_name)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    print(f"Loaded {len(df)} rows from '{table_name}' for modeling.")
    return df


# --- Attempt 1: raw features, contamination='auto' ---
# Result: recall 0.07 for Unusual — too low, contamination assumption
# (few, rare anomalies) didn't match this dataset's actual ~27.6% anomaly rate.
#
# --- Attempt 2: raw features, contamination=0.276 ---
# Result: recall 0.20 — improved, but still weak.
#
# --- Attempt 3: standardized (scaled) features ---
# Result: identical to raw features. Confirmed Isolation Forest is scale-
# invariant (splits are based on each feature's own range, not
# cross-feature distance), so scaling doesn't affect results for this model.
#
# --- Attempt 4: one-hot encoded Site added as features ---
# Result: no meaningful change. One-hot Site labels don't teach the model
# what "normal" looks like per site — they just add extra columns to split on.
def prepare_features_onehot_site(df, feature_columns=FEATURE_COLUMNS):
    X = df[feature_columns].copy()
    site_dummies = pd.get_dummies(df['Site'], prefix='Site')
    X = pd.concat([X, site_dummies], axis=1)
    return X


# --- Attempt 5 (not yet tested): per-site z-scores ---
# Idea: encode "unusual relative to this specific cell's own normal range"
# directly, rather than relying on the model to infer site-specific context.
def prepare_features_site_zscore(df, feature_columns=FEATURE_COLUMNS):
    X = df[feature_columns].copy()
    for col in feature_columns:
        site_mean = df.groupby('Site')[col].transform('mean')
        site_std = df.groupby('Site')[col].transform('std')
        X[col] = (df[col] - site_mean) / site_std
    X = X.fillna(0)
    return X


# --- Attempt 6 (ACTIVE): throughput-per-PRB-usage efficiency ratios ---
# Hypothesis: high PRB usage combined with low throughput indicates
# congestion. Isolation Forest has no built-in concept of a ratio between
# columns, so encoding this relationship explicitly should help the model
# find it, rather than relying on chance multivariate splits.
def prepare_features_with_efficiency(df, feature_columns=FEATURE_COLUMNS):
    X = df[feature_columns].copy()
    epsilon = 0.01
    X['DL_Efficiency'] = df['meanThr_DL'] / (df['PRBUsageDL'] + epsilon)
    X['UL_Efficiency'] = df['meanThr_UL'] / (df['PRBUsageUL'] + epsilon)
    return X


def train_isolation_forest(X, contamination='auto', random_state=42):
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(X)
    print(f"Trained Isolation Forest on {X.shape[1]} features, contamination='{contamination}'.")
    return model


def predict_anomalies(model, X):
    """
    IsolationForest.predict() returns 1 for normal, -1 for anomaly.
    We convert to 0/1 to match the dataset's own 'Unusual' label convention
    (0 = normal, 1 = anomalous) for easy comparison.
    """
    raw_predictions = model.predict(X)
    predictions = (raw_predictions == -1).astype(int)
    return predictions


def evaluate_against_labels(df, predictions):
    """
    Compares model predictions against the dataset's ground-truth 'Unusual'
    label. Note: the model never saw this label during training, so this is
    a genuine out-of-sample style check of unsupervised performance.
    """
    true_labels = df['Unusual']
    print("\n--- Confusion Matrix ---")
    print("Rows = actual, Columns = predicted")
    print(confusion_matrix(true_labels, predictions))
    print("\n--- Classification Report ---")
    print(classification_report(true_labels, predictions, target_names=['Normal', 'Unusual']))


if __name__ == "__main__":
    df = load_clean_data()
    X = prepare_features_with_efficiency(df)
    model = train_isolation_forest(X, contamination=0.276)
    predictions = predict_anomalies(model, X)
    df['Predicted_Anomaly'] = predictions
    evaluate_against_labels(df, predictions)