from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model

@component(
    base_image="aswinvj/kubeflow:latest", 
)
def preprocessed_component(
    input_data: Input[Dataset],
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    preprocessor_model: Output[Model],
):
    import os
    import joblib
    # import boto3
    from src.data_pipeline._06_preprocessing import preprocess_data

    input_path = os.path.join(input_data.path, "feature_engg.csv")

    train_df, test_df, preprocessor_obj = preprocess_data(input_path)

    os.makedirs(train_data.path, exist_ok=True)
    os.makedirs(test_data.path, exist_ok=True)
    os.makedirs(preprocessor_model.path, exist_ok=True)

    train_output_path = os.path.join(train_data.path, "train.csv")
    train_df.to_csv(train_output_path, index=False)

    test_output_path = os.path.join(test_data.path, "test.csv")
    test_df.to_csv(test_output_path, index=False)

    preprocessor_output_path = os.path.join(preprocessor_model.path, "preprocessor.pkl")
    joblib.dump(preprocessor_obj, preprocessor_output_path)

    # save in s3
    # s3 = boto3.client('s3')
    # bucket = "ml-basics"
    # key = "employee-attrition/preprocessing"

    # s3.upload_file(train_output_path, bucket, f"{key}/train.csv")
    # s3.upload_file(test_output_path, bucket, f"{key}/test.csv")
    # s3.upload_file(preprocessor_output_path, bucket, f"{key}/preprocessor.pkl")

    print("Preprocessing completed.")
    print(f"Train data saved to: {train_output_path}")
    print(f"Test data saved to: {test_output_path}")
    print(f"Scaler saved to: {preprocessor_output_path}")
