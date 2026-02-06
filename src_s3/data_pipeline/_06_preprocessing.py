from typing import Union
import pandas as pd
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(df: Union[str, BytesIO, bytes]) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    df_pp = pd.read_csv(df)

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
    import boto3
    import joblib
    from io import BytesIO
    from dotenv import load_dotenv

    load_dotenv()

    S3_BUCKET = os.environ.get("S3_BUCKET", "datasets")
    S3_KEY = os.environ.get("S3_KEY", "raw")

    s3 = boto3.client('s3')

    input_key = f"{S3_KEY}/feature_engg/feature_engg.csv"
    obj = s3.get_object(Bucket=S3_BUCKET, Key=input_key)
    df = BytesIO(obj['Body'].read())

    train_df, test_df, preprocessor = preprocess_data(df)

    def upload_df_to_s3(df: pd.DataFrame, bucket: str, key: str):
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        s3.upload_fileobj(buffer, Bucket=bucket, Key=key)
        print(f"✅ Uploaded {key} to s3://{bucket}/{key}")

    upload_df_to_s3(train_df, S3_BUCKET, f"{S3_KEY}/preprocessing/train_df.csv")
    upload_df_to_s3(test_df, S3_BUCKET, f"{S3_KEY}/preprocessing/test_df.csv")

    # --- 4. Upload preprocessor.pkl to S3 ---
    def upload_object_to_s3(obj, bucket: str, key: str):
        buffer = BytesIO()
        joblib.dump(obj, buffer)
        buffer.seek(0)
        s3.upload_fileobj(buffer, Bucket=bucket, Key=key)
        print(f"✅ Uploaded {key} to s3://{bucket}/{key}")

    upload_object_to_s3(preprocessor, S3_BUCKET, f"{S3_KEY}/preprocessing/preprocessor.pkl")
