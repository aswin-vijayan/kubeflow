from kfp import dsl
from kfp.dsl import Input, Output, Model, Dataset

@dsl.component(
    base_image="aswinvj/kubeflow:latest",
)
def tuning_component(
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    preprocessor_model: Input[Model],
    tuning_metadata: Output[Dataset],
):
    import os
    # import boto3
    import json
    from src.model_pipeline._07_tuning import tuning_data

    train_path = os.path.join(train_data.path, "train.csv")
    test_path = os.path.join(test_data.path, "test.csv")
    preprocessor_output = os.path.join(preprocessor_model.path, "preprocessor.pkl")

    tuning_metadata_output = tuning_data(train_path, test_path, preprocessor_output)

    os.makedirs(tuning_metadata.path, exist_ok=True)
    tuning_file = os.path.join(tuning_metadata.path, "tuning_metadata.json")
    with open(tuning_file, 'w') as f:
        json.dump(tuning_metadata_output, f)

    # save in s3
    # s3 = boto3.client('s3')
    # s3.put_object(
    #     Bucket="ml-basics",
    #     Key="employee-attrition/artifacts/tuning_metadata.json",
    #     Body=json.dumps(tuning_metadata).encode("utf-8")
    # )

    print("Tuning completed successfully.")

