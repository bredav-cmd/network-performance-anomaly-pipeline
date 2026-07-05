import sqlite3
import pandas as pd

DB_NAME = "network_kpi_analytics.db"
TABLE_NAME = "cell_kpi_clean"


def load_to_sqlite(df, db_name=DB_NAME, table_name=TABLE_NAME):
    """
    Loads the transformed DataFrame into a SQLite database.
    if_exists='replace' overwrites the table each run, keeping the
    pipeline idempotent (safe to re-run without duplicating rows).
    """
    conn = sqlite3.connect(db_name)
    try:
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        row_count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table_name}", conn).iloc[0]['count']
        print(f"Loaded {row_count} rows into table '{table_name}' in database '{db_name}'.")
    finally:
        conn.close()


if __name__ == "__main__":
    from etl import extract_train_data
    from transform import transform

    raw_df = extract_train_data()
    clean_df = transform(raw_df)
    load_to_sqlite(clean_df)