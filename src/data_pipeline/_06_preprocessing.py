import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(df_path: str) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    df_pp = pd.read_csv(df_path)

    # Separate features and target
    X = df_pp.drop(columns=['Attrition'])
    y = df_pp['Attrition']

    # Preprocess
    # 1. Scale Numeric cols: Scaling is about meaning, not datatype. Does the distance between values mean something numeric?
    #  - distance matter
    #  - magnitudes matter
    NUMERIC_COLS = ['Years at Company', 'Company Tenure', 'AnnualIncome', 'RoleStagnationRatio', 'TenureGap', 'Number of Promotions', 'Number of Dependents']
    
    # 2. Binary (0/1): Do not scale
    #  - 0/1 is a state, not quantity
    #  - scaling destroys interpretability
    BINARY_COLS = ["Overtime", "Remote Work", "EarlyCompanyTenureRisk", "LongTenureLowRoleRisk"]

    # 3. Ordinal Categorical: they look numeric but NOT
    #  - Check if distance between 1 and 2 is same as 3 and 4 ?
    #  - use OneHotEnoder unless you have strong reason not to.
    CATEGORICAL_COLS = ["Education Level", "Job Level", "Company Size", "Performance Rating", "AgeGroup", "OverallSatisfaction", "Opportunities", "Company Reputation" ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(), CATEGORICAL_COLS),
            ("bin", "passthrough", BINARY_COLS),
        ],
        remainder="passthrough"
    )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    print("Preprocessing completed and data saved.")
    return train_df, test_df, preprocessor


if __name__ == "__main__":
    import os
    from pathlib import Path
    import joblib
    
    BASE_DIR = Path(__file__).resolve().parents[2]    
    DATASET_PATH = BASE_DIR / "datasets" / "data-pipeline"
    FEATURED_PATH = DATASET_PATH / "05_feature_engg_df.csv"
    TRAIN_PATH = DATASET_PATH / "06_preprocess_train_df.csv"
    TEST_PATH = DATASET_PATH / "06_preprocess_test_df.csv"

    ARTIFACTS_PATH = BASE_DIR / "artifacts" / "model_v1"
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)
    
    PREPROCESSOR_PATH = ARTIFACTS_PATH / "preprocessor.pkl"

    train_df, test_df, preprocesor = preprocess_data(FEATURED_PATH)

    # Save preprocessed data
    joblib.dump(preprocesor, PREPROCESSOR_PATH)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
