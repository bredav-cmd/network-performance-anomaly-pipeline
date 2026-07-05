from etl import extract_train_data
from transform import transform
from load import load_to_sqlite

def run_pipeline():
    print("===== Starting Cell KPI ETL Pipeline =====")
    raw_df = extract_train_data()
    clean_df = transform(raw_df)
    load_to_sqlite(clean_df)
    print("===== Pipeline Finished =====")
    return clean_df

if __name__ == "__main__":
    run_pipeline()