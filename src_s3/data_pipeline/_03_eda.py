import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def eda_data(df: pd.DataFrame) -> pd.DataFrame:
    # Basic statistics
    print("Basic Statistics:")
    print(df.head(5))
    print("------")
    print(f"Information: {df.info()}")
    print("------")
    print(df.describe(include='all'))

    # Distribution of Age
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Age'], bins=30, kde=True)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()

    # Attrition count plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Attrition', data=df)
    plt.title('Attrition Count')
    plt.xlabel('Attrition')
    plt.ylabel('Count')
    plt.show()

    # Monthly Income vs Job Level
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Job Level', y='Monthly Income', data=df)
    plt.title('Monthly Income by Job Level')
    plt.xlabel('Job Level')
    plt.ylabel('Monthly Income')
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    corr = df.select_dtypes(include=['int64', 'float64']).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

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

    input_key = f"{S3_KEY}/validation/validation.csv"
    obj = s3.get_object(Bucket=S3_BUCKET, Key=input_key)
    df = pd.read_csv(BytesIO(obj['Body'].read()))

    eda_df = eda_data(df)

    if eda_df is not None:
        output_key = f"{S3_KEY}/eda/eda.csv"
        buffer = BytesIO()
        eda_df.to_csv(buffer, index=False)
        buffer.seek(0)
        
        s3.upload_fileobj(buffer, Bucket=S3_BUCKET, Key=output_key)
        print(f"✅ Validated dataset uploaded to s3://{S3_BUCKET}/{output_key}")
