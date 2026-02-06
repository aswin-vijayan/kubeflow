from kfp import dsl
from kfp.dsl import component, Input, Dataset


@component(
    base_image="aswinvj/kubeflow:latest",
)
def evaluation_component(
    test_data: Input[Dataset],
    tracking_uri: str,
    experiment_name: str,
):
    """Evaluate model and log metrics to MLflow."""
    import os
    from src.model_pipeline._09_evaluation import evaluate_data

    # Build the test path (artifact folder + filename)
    test_path = os.path.join(test_data.path, "test.csv")
    
    print(f"Test data path: {test_path}")
    print(f"Tracking URI: {tracking_uri}")
    print(f"Experiment name: {experiment_name}")

    # Pass MLflow params directly instead of using env vars
    metrics = evaluate_data(
        test_path=test_path,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name
    )
    
    print(f"Evaluation metrics: {metrics}")
