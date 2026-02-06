from kfp.dsl import component, Output, Dataset

@component(
    base_image="aswinvj/kubeflow:latest"
)
def ingestion_component(
    output_data: Output[Dataset]
):
# def ingestion_component(
#     bucket: str,
#     key: str,
    # output_data: Output[Dataset]
# ):
    import os
    # import boto3
    from src.data_pipeline._01_ingestion import ingestion


    # df = ingestion(bucket, key)
    df = ingestion()

    # KFP v2: output_data.path is a DIRECTORY, not a file
    os.makedirs(output_data.path, exist_ok=True)
    output_path = os.path.join(output_data.path, "ingestion.csv")

    df.to_csv(output_path, index=False)

    # save in s3
    # s3 = boto3.client('s3')
    # bucket = "ml-basics"
    # key = "employee-attrition/ingestion"

    # s3.upload_file(output_path, bucket, f"{key}/ingestion.csv")

    print(f"Ingestion completed. Saved to {output_path}")

