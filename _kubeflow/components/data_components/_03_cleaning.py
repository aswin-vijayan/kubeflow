
from kfp.dsl import component, Input, Output, Dataset


@component(
    base_image="aswinvj/kubeflow:latest", 
)
def cleaned_component(
    input_data: Input[Dataset], 
    output_data: Output[Dataset]
):
    import os
    # import boto3
    from src.data_pipeline._04_cleaning import clean_data

    input_path = os.path.join(input_data.path, "validation.csv")

    clean_df = clean_data(input_path)

    os.makedirs(output_data.path, exist_ok=True)
    output_path = os.path.join(output_data.path, "cleaned.csv")
    clean_df.to_csv(output_path, index=False)

    # save in s3
    # s3 = boto3.client('s3')
    # bucket = "ml-basics"
    # key = "employee-attrition/cleaned"

    # s3.upload_file(output_path, bucket, f"{key}/cleaned.csv")

    print(f"Cleaning completed. Saved to {output_path}")


