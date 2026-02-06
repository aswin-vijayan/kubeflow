from kfp import dsl
from kfp.dsl import Input, Artifact, OutputPath


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["kubernetes"]
)
def trainer_model_component(
    job_name: str,
    namespace: str,
    image: str,
    cpu: str,
    memory: str,
    train_path: Input[Artifact],
    preprocessor_model: Input[Artifact],
    tuning_metadata: Input[Artifact],
    job_output: OutputPath(str),
):
    """Creates a Kubeflow TrainJob for model training with MinIO access."""
    
    from kubernetes import client, config
    
    # Load in-cluster config
    config.load_incluster_config()
    
    # Create custom objects API
    api = client.CustomObjectsApi()
    
    # TrainJob specification with MinIO credentials and S3 artifact storage
    train_job = {
        "apiVersion": "trainer.kubeflow.org/v1alpha1",
        "kind": "TrainJob",
        "metadata": {
            "name": job_name,
            "namespace": namespace
        },
        "spec": {
            "suspend": False,
            "runtimeRef": {
                "apiGroup": "trainer.kubeflow.org",
                "kind": "ClusterTrainingRuntime",
                "name": "torch-distributed"
            },
            "trainer": {
                "image": image,
                "command": ["python", "-m", "src.model_pipeline._08_training"],
                "args": [
                    "--train_path", train_path.uri,
                    "--preprocessor_path", preprocessor_model.uri,
                    "--best_params_path", tuning_metadata.uri,
                ],
                "numNodes": 1,
                "resourcesPerNode": {
                    "requests": {
                        "cpu": cpu,
                        "memory": memory
                    },
                    "limits": {
                        "cpu": cpu,
                        "memory": memory
                    }
                },
                "env": [
                    # MLflow settings
                    {"name": "MLFLOW_TRACKING_URI", "value": "http://mlflow.mlflow:80"},
                    {"name": "MLFLOW_EXPERIMENT_NAME", "value": "employee-attrition-v1"},
                    
                    # Use MLflow server as artifact proxy (artifacts go through MLflow server to S3)
                    {"name": "MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "value": "false"},
                    
                    # Direct S3 artifact storage - artifacts written directly to S3
                    {"name": "MLFLOW_S3_ENDPOINT_URL", "value": "http://minio-service.kubeflow:9000"},
                    {"name": "MLFLOW_S3_IGNORE_TLS", "value": "true"},
                    {"name": "MLFLOW_ARTIFACT_ROOT", "value": "s3://mlpipeline/mlflow-artifacts"},
                    
                    # MinIO endpoint for downloading pipeline artifacts
                    {"name": "MINIO_ENDPOINT", "value": "http://minio-service.kubeflow:9000"},
                    
                    # MinIO/AWS credentials from secret
                    {
                        "name": "AWS_ACCESS_KEY_ID",
                        "valueFrom": {
                            "secretKeyRef": {
                                "name": "mlpipeline-minio-artifact",
                                "key": "accesskey"
                            }
                        }
                    },
                    {
                        "name": "AWS_SECRET_ACCESS_KEY",
                        "valueFrom": {
                            "secretKeyRef": {
                                "name": "mlpipeline-minio-artifact",
                                "key": "secretkey"
                            }
                        }
                    }
                ]
            }
        }
    }
    
    # Create the TrainJob
    api.create_namespaced_custom_object(
        group="trainer.kubeflow.org",
        version="v1alpha1",
        namespace=namespace,
        plural="trainjobs",
        body=train_job
    )
    
    print(f"Created TrainJob: {job_name}")
    
    # Write job name to output
    with open(job_output, "w") as f:
        f.write(job_name)