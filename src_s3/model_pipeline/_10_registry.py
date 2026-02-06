from _mlflow.registry import MLflowRegistry


def register_model_to_mlflow(
    tracking_uri: str,
    experiment_name: str,
    registry_name: str
) -> object:
    
    registry = MLflowRegistry(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name
    )

    metadata = registry.get_model_uri_from_mlflow()
    print('metadata: ', metadata)

    registered = registry.register_model(
        metadata=metadata,
        registry_name=registry_name
    )

    metrics = registry.get_metric_from_mlfow()

    return registered, metrics


def promote_to_production(
    metric: float,
    model_name: str,
    version: int,
    tracking_uri: str,
    experiment_name: str,
    recall_threshold: float = 0.70,
) -> str:

    registry = MLflowRegistry(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name
    )

    stage = registry.promote_model(
        metric_value=metric,
        model_name=model_name,
        version=version,
        threshold=recall_threshold,
    )

    print(f"Model promoted to: {stage}")
    return stage


if __name__ == "__main__":
    registered_model, metrics = register_model_to_mlflow(
        tracking_uri="http://localhost:5000",
        experiment_name="employee-attrition-v1",
        registry_name="register-employee-attrition-model"
    )
    
    promote_to_production(
        metric=metrics['recall'],
        model_name=registered_model.name,
        version=registered_model.version,
        tracking_uri="http://localhost:5000",
        experiment_name="employee-attrition-v1",
        recall_threshold=0.6,
    )

