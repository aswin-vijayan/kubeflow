import pandera.pandas as pa
from pandera import Column, Check
import pandas as pd

# 1.Define the schema
employee_schema = pa.DataFrameSchema(
    {
    "Employee ID": Column(int, Check.ge(1), unique=True, nullable=False),
    "Attrition": Column(str, Check.isin(['Stayed', 'Left']), nullable=False),
    "Age": Column(int, Check.ge(18), nullable=False),
    "Gender": Column(str, Check.isin(['Male', 'Female']), nullable=False),
    "Years at Company": Column(int, Check.ge(0), nullable=False),
    "Job Role": Column(str, nullable=False),
    "Monthly Income": Column(float, Check.ge(0), nullable=False),
    "Job Level": Column(str, Check.isin(["Entry", "Mid", "Senior", "Executive"]), nullable=False),
    "Monthly Income": Column(int, Check.ge(0), nullable=False),
    "Work-Life Balance": Column(str, Check.isin(["Poor", "Fair", "Good", "Excellent"]), nullable=False),
    "Job Satisfaction": Column(str, Check.isin(["Low", "Medium", "High", "Very High"]), nullable=False),
    "Performance Rating": Column(str, Check.isin(["Low", "Below Average", "Average", "High"]), nullable=False),
    "Number of Promotions": Column(int, Check.ge(0), nullable=False),
    "Company Size": Column(str,Check.isin(["Small", "Medium", "Large"]),nullable=False),
    "Company Tenure": Column(int,Check.ge(0),nullable=False),
    "Remote Work": Column(str, Check.isin(["Yes", "No"]), nullable=False),
    "Leadership Opportunities": Column(str,Check.isin(["Yes", "No"]),nullable=False),
    "Innovation Opportunities": Column(str, Check.isin(["Yes", "No"]), nullable=False),
    "Company Reputation": Column(str, Check.isin(["Poor", "Fair", "Good", "Excellent"]), nullable=False),
    "Employee Recognition": Column(str, Check.isin(["Low", "Medium", "High", "Very High"]), nullable=False),
    },
    strict=False,
    coerce=True
)



def validate_data(df: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(df)
        validate_df = employee_schema(df, lazy=True)
        print("Data validation successful.")
        return validate_df
    except pa.errors.SchemaErrors as e:
        print("Data validation errors found:")
        print(e.failure_cases)
        return None
    

if __name__ == "__main__":
    import os
    import boto3
    from io import BytesIO
    from dotenv import load_dotenv

    load_dotenv()

    S3_BUCKET = os.environ.get("S3_BUCKET", "datasets")
    S3_KEY = os.environ.get("S3_KEY", "raw")

    s3 = boto3.client('s3')

    input_key = f"{S3_KEY}/ingestion/ingestion.csv"
    obj = s3.get_object(Bucket=S3_BUCKET, Key=input_key)
    df = BytesIO(obj['Body'].read())

    validated_df = validate_data(df)

    if validated_df is not None:
        output_key = f"{S3_KEY}/validation/validation.csv"
        buffer = BytesIO()
        validated_df.to_csv(buffer, index=False)
        buffer.seek(0)
        s3.upload_fileobj(buffer, Bucket=S3_BUCKET, Key=output_key)
        print(f"✅ Validated dataset uploaded to s3://{S3_BUCKET}/{output_key}")

