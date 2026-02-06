import json
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient


class MLflowRegistry:
    def __init__(self, tracking_uri: str, experiment_name: str):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.client = MlflowClient()

    # start run
    def start_run(self, run_name: str = None, run_id: str = None):
        return mlflow.start_run(run_name=run_name, run_id=run_id)

    def log_model(
        self, 
        model, 
        X_train, 
        parameters: dict, 
        artifact_name: str
    ):
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.log_params(parameters)

        # Fixed: use artifact_path instead of name
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_name,  # ✅ Correct parameter
            signature=signature,
            input_example=X_train.iloc[:5],
        )

        mlflow_metadata = {
            "run_id": mlflow.active_run().info.run_id,
            "model_uri": model_info.model_uri,
            "artifact_name": artifact_name,
        }
        
        features = list(X_train.columns)
        features = {"raw_features": features}
        mlflow.log_dict(features, "features_schema.json")
        mlflow.log_param("num_features", len(features['raw_features']))
        mlflow.log_param("feature_list", features)

        mlflow.set_tag("artifact_name", artifact_name)
        mlflow.log_dict(mlflow_metadata, "mlflow_metadata.json")

        print(f"Model logged successfully!")
        print(f"Model URI: {model_info.model_uri}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

        return True

    def get_model_uri_from_mlflow(self):
        runs = mlflow.search_runs(
            filter_string="tags.artifact_name = 'employee-attrition-model'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        
        if runs.empty:
            raise ValueError("No runs found with artifact_name = 'employee-attrition-model'")
        
        run_id = runs.iloc[0].run_id

        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="mlflow_metadata.json"
        )

        with open(local_path) as f:
            metadata = json.load(f)
        
        return metadata

    def get_metric_from_mlfow(self):
        runs = mlflow.search_runs(
            filter_string="tags.artifact_name = 'employee-attrition-model'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        
        if runs.empty:
            raise ValueError("No runs found with artifact_name = 'employee-attrition-model'")
            
        run_id = runs.iloc[0].run_id

        run = self.client.get_run(run_id)
        metrics = run.data.metrics

        return metrics

    def log_evaluation_metrics(self, metrics):
        mlflow.log_metrics(metrics)
        return True

    def load_model(self, metadata: dict = None, model_uri: str = None):
        if metadata:
            model_uri = f"runs:/{metadata['run_id']}/{metadata['artifact_name']}"
        print('model-uri: ', model_uri)
        model = mlflow.sklearn.load_model(model_uri)
        return model

    def load_features_from_mlflow(self):
        runs = mlflow.search_runs(
            filter_string="tags.artifact_name = 'employee-attrition-model'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        
        if runs.empty:
            raise ValueError("No runs found with artifact_name = 'employee-attrition-model'")
            
        run_id = runs.iloc[0].run_id

        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="features_schema.json"
        )

        with open(local_path) as f:
            features = json.load(f)
        
        return features['raw_features']

    def register_model(self, metadata, registry_name):
        model_uri = f"runs:/{metadata['run_id']}/{metadata['artifact_name']}"
        print('model-uri: ', model_uri)

        register_model = mlflow.register_model(
            model_uri=model_uri,
            name=registry_name
        )
        version = register_model.version

        # set tags
        self.client.set_model_version_tag(
            name=register_model.name,
            version=version,
            key="source_run_id",
            value=metadata['run_id']
        )

        self.client.set_model_version_tag(
            name=register_model.name,
            version=version,
            key="production_ready",
            value="pending_approval",
        )

        # Transition to stage
        self.client.transition_model_version_stage(
            name=register_model.name,
            version=version,
            stage='Staging',
        )

        print("\nModel registered successfully!")
        print(f"\nVersion: {version}")

        return register_model

    def promote_model(self, model_name: str, version: int, metric_value: float, threshold: float):
        stage = "Production" if metric_value >= threshold else "Staging"
        self.client.transition_model_version_stage( 
            name=model_name, 
            version=version, 
            stage=stage, 
        ) 
        self.client.set_model_version_tag( 
            name=model_name, 
            version=version, 
            key="promotion_decision", 
            value=f"{stage} (metric={metric_value}, threshold={threshold})", 
        ) 
        return stage
