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
    import os
    import boto3
    from io import BytesIO
    from dotenv import load_dotenv

    load_dotenv()

    S3_BUCKET = os.environ.get("S3_BUCKET", "datasets")
    S3_KEY = os.environ.get("S3_KEY", "raw")

    s3 = boto3.client('s3')

    input_key = f"{S3_KEY}/eda/eda.csv"
    obj = s3.get_object(Bucket=S3_BUCKET, Key=input_key)
    df = BytesIO(obj['Body'].read())

    cleaned_df = clean_data(df)

    if cleaned_df is not None:
        output_key = f"{S3_KEY}/cleaned/cleaned.csv"
        buffer = BytesIO()
        cleaned_df.to_csv(buffer, index=False)
        buffer.seek(0)
        
        s3.upload_fileobj(buffer, Bucket=S3_BUCKET, Key=output_key)
        print(f"✅ Validated dataset uploaded to s3://{S3_BUCKET}/{output_key}")

