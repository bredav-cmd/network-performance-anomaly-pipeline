import pandas as pd

RAW_TRAIN_PATH = "data/raw/cell_kpi_train.csv"

def extract_train_data(path=RAW_TRAIN_PATH):
    """Loads the raw cell KPI training data from CSV."""
    # Source file is not standard UTF-8 (raises UnicodeDecodeError) —
    # using ISO-8859-1 (Latin-1) as a permissive fallback encoding.
    df = pd.read_csv(path, encoding='ISO-8859-1')
    print(f"Loaded {len(df)} rows from {path}")
    return df

if __name__ == "__main__":
    df = extract_train_data()
    pd.set_option('display.max_columns', None)
    df.info()
    print("\n--- First 5 rows ---")
    print(df.head())
    print("\n--- Missing values per column ---")
    print(df.isnull().sum())

    print("\n--- Unique Time values (sample) ---")
    print(sorted(df['Time'].unique())[:20])

    print("\n--- Unique CellName count ---")
    print(df['CellName'].nunique())

    print("\n--- Unusual value counts ---")
    print(df['Unusual'].value_counts())
    print("\n--- All unique CellNames ---")
    print(sorted(df['CellName'].unique()))

    print("\n--- Technology suffix (last 3 chars) counts ---")
    print(df['CellName'].str[-3:].value_counts())

    print("\n--- Sample of maxUE_UL+DL raw values ---")
    print(df['maxUE_UL+DL'].unique()[:30])
    import re

    print("\n--- Site/Sector extraction test ---")
    #first 10 under CellName
    sample_names = df['CellName'].unique()[:10]
    for name in sample_names:
        #capture one-or-more digits (site number), then capture one letter (sector), then require literal "LTE"
        match = re.match(r'(\d+)([A-Z])LTE', name)
        if match:
            print(f"{name} -> Site: {match.group(1)}, Sector: {match.group(2)}")
        else:
            print(f"{name} -> NO MATCH")
