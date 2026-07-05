import sqlite3
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

DB_NAME = "network_kpi_analytics.db"
TABLE_NAME = "cell_kpi_clean"

FEATURE_COLUMNS = [
    'PRBUsageUL', 'PRBUsageDL',
    'meanThr_DL', 'meanThr_UL', 'maxThr_DL', 'maxThr_UL',
    'meanUE_DL', 'meanUE_UL', 'maxUE_DL', 'maxUE_UL', 'maxUE_UL+DL',
    'Time_Minutes'
]
def prepare_features(df, feature_columns=FEATURE_COLUMNS):
    """One-hot encodes Site so the model can learn per-site normal behavior,
    rather than treating all cells as one undifferentiated population."""
    X = df[feature_columns].copy()
    site_dummies = pd.get_dummies(df['Site'], prefix='Site')
    X = pd.concat([X, site_dummies], axis=1)
    return X

def load_clean_data(db_name=DB_NAME, table_name=TABLE_NAME):
    """Loads the cleaned KPI data from SQLite via a SQL query."""
    conn = sqlite3.connect(db_name)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    print(f"Loaded {len(df)} rows from '{table_name}' for modeling.")
    return df


"""def train_isolation_forest(df, feature_columns=FEATURE_COLUMNS, contamination='auto', random_state=42):
    Trains an Isolation Forest on standardized features. Standardization
    (zero mean, unit variance) prevents features with naturally larger
    numeric ranges (e.g. Time_Minutes, 0-825) from dominating the model's
    splits over smaller-scale but equally meaningful features (e.g. mean
    throughput, often < 1.0). The 'Unusual' label is NOT used for training,
    only for evaluation afterward.
    
    X_raw = df[feature_columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(X_scaled)
    print(f"Trained Isolation Forest on {len(feature_columns)} standardized features, contamination='{contamination}'.")
    return model, X_scaled"""

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


"""if __name__ == "__main__":
    df = load_clean_data()
    model, X = train_isolation_forest(df, contamination=0.276)
    predictions = predict_anomalies(model, X)
    df['Predicted_Anomaly'] = predictions
    evaluate_against_labels(df, predictions)"""

if __name__ == "__main__":
    df = load_clean_data()
    X = prepare_features(df)
    model = train_isolation_forest(X, contamination=0.276)
    predictions = predict_anomalies(model, X)
    df['Predicted_Anomaly'] = predictions
    evaluate_against_labels(df, predictions)