import argparse
import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from _mlflow.registry import MLflowRegistry


def training_data(train_path: str, preprocessor_path: str, best_params_path: str, tracking_uri: str, experiment_name: str, artifact_name: str):
    registry = MLflowRegistry(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name
    )

    # load data
    df = pd.read_csv(train_path)
    X_train = df.drop(columns=['Attrition'])
    y_train = df['Attrition']

    params = json.load(best_params_path)

    best_params = {
        key.replace("model__", ""): value
        for key, value in params.items()
        if key.startswith("model__")
    }

    # load scaler
    preprocessor = joblib.load(preprocessor_path)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(**best_params))
    ])

    with registry.start_run(run_name='model-training-run'):
        print("Training the model....")
        pipeline.fit(X_train, y_train)

        print(pipeline)
        print("training completed...")

        registry.log_model(
            model=pipeline,
            X_train=X_train,
            parameters=best_params,
            artifact_name=artifact_name,
        )

    print(f"✓ Training complete")
    return True


if __name__ == "__main__":
    import boto3
    import pandas as pd
    import joblib
    from io import BytesIO

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--preprocessor_path", required=True)
    parser.add_argument("--best_params_path", required=True)
    args = parser.parse_args()
    
    s3 = boto3.client('s3')

    def load_from_s3(s3_uri):
        bucket, key = s3_uri.replace("s3://", "").split("/", 1)
        print('key: ', key)
        obj = s3.get_object(Bucket=bucket, Key=key)
        return BytesIO(obj['Body'].read())
    
    train_df = load_from_s3(args.train_path)
    preprocessor = load_from_s3(args.preprocessor_path)
    best_params = load_from_s3(args.best_params_path)


    training_data(
        train_path=train_df,
        preprocessor_path=preprocessor,
        best_params_path=best_params,
        tracking_uri="http://localhost:5000", 
        experiment_name="employee-attrition-v1", 
        artifact_name="employee-attrition-model"
    )



# run command
# python -m src.model_pipeline._08_training --train_path s3://ml-basics/employee-attrition/preprocessing/train_df.csv --preprocessor_path s3://ml-basics/employee-attrition/preprocessing/preprocessor.pkl --best_params_path s3://ml-basics/employee-attrition/artifacts/tuning_metadata.json

# model_path=args.model_path, 
# feature_store_path=args.feature_store_path,