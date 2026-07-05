import pandas as pd
import re


def fix_max_ue_total(df):
    """
    The 'maxUE_UL+DL' column contains a Spanish-locale Excel error string
    ('#¡VALOR!') in place of a number for some rows, likely where the
    underlying maxUE_UL/maxUE_DL values were missing when the source
    spreadsheet computed this column.
    errors='coerce' converts any value that isn't a valid number into NaN
    instead of raising an error.
    """
    df['maxUE_UL+DL'] = pd.to_numeric(df['maxUE_UL+DL'], errors='coerce')
    return df


def drop_incomplete_rows(df):
    """
    Drops rows with missing values in the UE (user equipment) count columns.
    These are a small fraction of the dataset (~89 of 36,904 rows, ~0.24%),
    so dropping is preferable to imputing and risking distorted anomaly signals.
    """
    before = len(df)
    df = df.dropna(subset=['maxUE_DL', 'maxUE_UL', 'maxUE_UL+DL'])
    after = len(df)
    print(f"Dropped {before - after} rows with missing UE count data ({before} -> {after})")
    return df


def extract_site_sector(df):
    """
    CellName follows the pattern {siteNumber}{sectorLetter}LTE, e.g. '10BLTE'
    -> Site 10, Sector B. Extracting these as separate columns lets the model
    (and any later analysis) account for site- and sector-level differences.
    """
    extracted = df['CellName'].str.extract(r'(\d+)([A-Z])LTE')
    df['Site'] = extracted[0]
    df['Sector'] = extracted[1]
    return df


def convert_time_to_minutes(df):
    """
    Converts 'Time' (e.g. '9:45') into minutes since midnight (e.g. 585),
    a numeric feature usable by the model, while preserving the original
    column for readability/reporting.
    """
    time_parts = df['Time'].str.split(':', expand=True).astype(int)
    df['Time_Minutes'] = time_parts[0] * 60 + time_parts[1]
    return df


def transform(df):
    """Runs the full transformation pipeline in sequence."""
    print("Starting transformation...")
    df = fix_max_ue_total(df)
    df = drop_incomplete_rows(df)
    df = extract_site_sector(df)
    df = convert_time_to_minutes(df)
    print("Transformation complete.")
    return df


if __name__ == "__main__":
    from etl import extract_train_data

    raw_df = extract_train_data()
    clean_df = transform(raw_df)

    pd.set_option('display.max_columns', None)
    clean_df.info()
    print("\n--- First 5 rows ---")
    print(clean_df.head())