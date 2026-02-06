import pandas as pd
import numpy as np
import os
from _mlflow.registry import MLflowRegistry
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_data(test_path: str, tracking_uri: str = None, experiment_name: str = None):
    """
    Evaluate model from MLflow and log metrics.
    
    Args:
        test_path: Path to test data CSV
        tracking_uri: MLflow tracking URI (optional, falls back to env var)
        experiment_name: MLflow experiment name (optional, falls back to env var)
    """
    # Use provided values or fall back to environment variables
    tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow.mlflow:80")
    experiment_name = experiment_name or os.environ.get("MLFLOW_EXPERIMENT_NAME", "employee-attrition-v1")
    
    print(f"MLflow Tracking URI: {tracking_uri}")
    print(f"MLflow Experiment: {experiment_name}")

    registry = MLflowRegistry(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name
    )

    metadata = registry.get_model_uri_from_mlflow()
    print('metadata: ', metadata)

    with registry.start_run(run_id=metadata['run_id']):

        # load model from MLflow
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

    return metrics


if __name__ == "__main__":
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]
    DATASET_PATH = BASE_DIR / "datasets" / "data-pipeline"
    TEST_PATH = DATASET_PATH / "06_preprocess_test_df.csv"

    evaluate_data(TEST_PATH)
