from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset

@component(
    base_image="aswinvj/kubeflow:latest", 
)
def feature_engg_component(
    input_data: Input[Dataset], 
    output_data: Output[Dataset]
):
    import os
    # import boto3
    from src.data_pipeline._05_feature_engg import feature_data

    input_path = os.path.join(input_data.path, "cleaned.csv")
    
    feature_df = feature_data(input_path)

    os.makedirs(output_data.path, exist_ok=True)
    output_path = os.path.join(output_data.path, "feature_engg.csv")
    feature_df.to_csv(output_path, index=False)

    # save in s3
    # s3 = boto3.client('s3')
    # bucket = "ml-basics"
    # key = "employee-attrition/feature_engg"

    # s3.upload_file(output_path, bucket, f"{key}/feature_engg.csv")
    
    print(f"Feature engg is completed. Saved to {output_path}")


