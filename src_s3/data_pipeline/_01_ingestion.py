import os
import pandas as pd
import boto3
from io import BytesIO

s3 = boto3.client('s3')

def ingestion(bucket: str, input_key: str) -> pd.DataFrame:
    try:
        obj = s3.get_object(Bucket=bucket, Key=input_key)
        df = pd.read_csv(obj['Body'])

        df = pd.read_csv(df_path)

        print(f"✅ Loaded dataset from s3://{bucket}/{input_key}")
        print(f"➡️ Dataset shape: {df.shape}")

        return df
    
    except Exception as e:
        print(f"Error downloading file from S3: {e}")
        return None


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    S3_BUCKET = os.environ.get("S3_BUCKET", "datasets")
    S3_KEY = os.environ.get("S3_KEY", "raw")
    RAW_DATASET = os.environ.get("RAW_DATASET", "employee_attrition.csv")

    input_key = f"{S3_KEY}/{RAW_DATASET}"
    output_key = f"{S3_KEY}/ingestion/ingestion.csv"

    df = ingestion(S3_BUCKET, input_key)

    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    s3.upload_fileobj(buffer, Bucket=S3_BUCKET, Key=output_key)

