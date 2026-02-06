import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from _mlflow.registry import MLflowRegistry
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_data(test_path: str, tracking_uri: str, experiment_name: str):

    registry = MLflowRegistry(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name
    )

    metadata = registry.get_model_uri_from_mlflow()
    print('metadata: ', metadata)

    with registry.start_run(run_id=metadata['run_id']):

        # load model form MLflow
        model = registry.load_model(metadata=metadata)

        df_test = pd.read_csv(test_path)
        X_test = df_test.drop(columns=['Attrition'])
        y_test = df_test['Attrition']

        # predict
        y_pred = model.predict(X_test)
        
        # metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred),
        }

        print('metrics: ', metrics)

        registry.log_evaluation_metrics(metrics)

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

    # feature importance
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    logreg = model.named_steps["model"]
    coef = logreg.coef_[0]
    coef_df = pd.DataFrame({
        'Feature': feature_names, 
        'Coefficient': coef,
        'Abs_Coefficient': np.abs(coef)
    }).sort_values(by='Abs_Coefficient', ascending=False)
    print(coef_df)

    plt.figure(figsize=(10,6))
    plt.barh(coef_df['Feature'], coef_df['Coefficient'])
    plt.gca().invert_yaxis()
    plt.xlabel("Coefficient")
    plt.title("Logistic Regression Feature Importance")
    plt.show()

    return metrics



if __name__ == "__main__":
    import os
    import boto3
    from io import BytesIO
    from dotenv import load_dotenv

    load_dotenv()

    S3_BUCKET = os.environ.get("S3_BUCKET", "datasets")
    S3_KEY = os.environ.get("S3_KEY", "raw")

    s3 = boto3.client('s3')

    key = f"{S3_KEY}/preprocessing/test_df.csv"
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)

    test_df = BytesIO(obj['Body'].read())

    evaluate_data(
        test_df,
        tracking_uri="http://localhost:5000", 
        experiment_name="employee-attrition-v1", 
    )
