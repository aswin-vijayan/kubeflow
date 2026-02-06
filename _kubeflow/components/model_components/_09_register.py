from kfp import dsl
from kfp.dsl import component


@component(
    base_image="aswinvj/kubeflow:latest",
)
def register_model_component(
    registry_name: str,
    recall_threshold: float,
    tracking_uri: str,
    experiment_name: str,
):
    """Register model to MLflow Model Registry."""
    from src.model_pipeline._10_registry import register_model

    print(f"Registry name: {registry_name}")
    print(f"Recall threshold: {recall_threshold}")
    print(f"Tracking URI: {tracking_uri}")
    print(f"Experiment name: {experiment_name}")

    # Pass MLflow params directly
    register_model(
        registry_name=registry_name,
        recall_threshold=recall_threshold,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name
    )
    
    print("Register component completed!")
