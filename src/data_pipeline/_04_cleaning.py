import pandas as pd

def clean_data(df_path: str) -> pd.DataFrame:
    df = pd.read_csv(df_path)

    print("--- Find duplicates ---")
    df_duplicate = df.duplicated()
    print(f"Number of duplicate rows: {df_duplicate.sum()}")

    print("--- Find Missing/Null values ---")
    missing_values = df.isnull().sum()
    print(f"Missing values: {missing_values}")
    return df


if __name__ == "__main__":
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]
    DATASET_PATH = BASE_DIR / "datasets" / "data-pipeline"
    EDA_PATH = DATASET_PATH / "03_eda_df.csv"
    CLEANED_PATH = DATASET_PATH / "04_cleaned_df.csv"

    cleaned_df = clean_data(EDA_PATH)

    cleaned_df.to_csv(CLEANED_PATH, index=False)
