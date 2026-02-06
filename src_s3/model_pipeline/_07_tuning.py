import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score


def tuning_data(train_path: str, test_path: str, preprocessor_path: str) -> dict:
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train.drop(columns=['Attrition'])
    y_train = df_train['Attrition']

    X_test = df_test.drop(columns=['Attrition'])
    y_test = df_test['Attrition']

    # load preprocessor
    preprocessor = joblib.load(preprocessor_path)


    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])

    # set parameters
    param_grid = {
        'model__C': [0.01, 0.1, 1, 10, 100],
        'model__solver': ['liblinear', 'saga'],
        'model__l1_ratio': [0],     # equivalent to L2
        'model__max_iter': [1000]
    }

    # set cv
    strat_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # use gridserachcv for tuning model
    grid = GridSearchCV(
        estimator=pipeline, 
        param_grid=param_grid, 
        cv=strat_cv, 
        scoring='recall',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    # get best model and save in models/
    tuned_model = grid.best_estimator_

    best_parameters = grid.best_params_
        
    # predict the output with tuned_model
    y_pred = tuned_model.predict(X_test)

    # tuned model evaluation
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "train_score": tuned_model.score(X_train, y_train),
        "test_score": tuned_model.score(X_test, y_test)
    }        


    overall_parameters = {
        **best_parameters,
        **metrics,
    }

    print('parameters: ', overall_parameters)

    return overall_parameters



if __name__ == "__main__":
    import os
    import boto3
    import joblib
    from io import BytesIO
    import json
    from dotenv import load_dotenv

    load_dotenv()

    S3_BUCKET = os.environ.get("S3_BUCKET", "datasets")
    S3_KEY = os.environ.get("S3_KEY", "raw")

    s3 = boto3.client('s3')

    def load_from_s3(bucket: str, key: str):
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = BytesIO(obj['Body'].read())
        return df

    train_path = load_from_s3(S3_BUCKET, f"{S3_KEY}/preprocessing/train_df.csv")
    test_path = load_from_s3(S3_BUCKET, f"{S3_KEY}/preprocessing/test_df.csv")
    preprocessor = load_from_s3(S3_BUCKET, f"{S3_KEY}/preprocessing/preprocessor.pkl")


    tuning_metadata = tuning_data(train_path, test_path, preprocessor)

    TUNING_METADATA_KEY = f"{S3_KEY}/artifacts/tuning_metadata.json"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=TUNING_METADATA_KEY,
        Body=json.dumps(tuning_metadata).encode("utf-8")
    )

