import os
from _mlflow.registry import MLflowRegistry


def register_model(
    registry_name: str, 
    recall_threshold: float,
    tracking_uri: str = None,
    experiment_name: str = None
):
    """
    Register model to MLflow Model Registry if recall meets threshold.
    
    Args:
        registry_name: Name for the registered model
        recall_threshold: Minimum recall to register the model
        tracking_uri: MLflow tracking URI (optional, falls back to env var)
        experiment_name: MLflow experiment name (optional, falls back to env var)
    """
    # Use provided values or fall back to environment variables
    tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow.mlflow:80")
    experiment_name = experiment_name or os.environ.get("MLFLOW_EXPERIMENT_NAME", "employee-attrition-v1")

    print(f"MLflow Tracking URI: {tracking_uri}")
    print(f"MLflow Experiment: {experiment_name}")
    print(f"Registry name: {registry_name}")
    print(f"Recall threshold: {recall_threshold}")

    registry = MLflowRegistry(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name
    )

    # Get model metadata from MLflow
    metadata = registry.get_model_uri_from_mlflow()
    print(f"Model metadata: {metadata}")

    # Get evaluation metrics
    metrics = registry.get_metric_from_mlfow()
    print(f"Metrics: {metrics}")

    recall = metrics.get("recall", 0)
    print(f"Recall: {recall}")

    # Register model if recall meets threshold
    if recall >= recall_threshold:
        print(f"Recall {recall} >= threshold {recall_threshold}, registering model...")
        registered_model = registry.register_model(metadata, registry_name)
        
        # Promote model based on recall
        stage = registry.promote_model(
            model_name=registry_name,
            version=registered_model.version,
            metric_value=recall,
            threshold=recall_threshold
        )
        print(f"Model promoted to: {stage}")
    else:
        print(f"Recall {recall} < threshold {recall_threshold}, skipping registration")

    print("Registration completed!")
    return True


if __name__ == "__main__":
    register_model(
        registry_name="employee-attrition-model",
        recall_threshold=0.70
    )
