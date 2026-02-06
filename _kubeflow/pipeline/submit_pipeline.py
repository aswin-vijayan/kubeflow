import os
from dotenv import load_dotenv
import kfp
from _kubeflow.pipeline.full_pipeline import full_pipeline

load_dotenv()

PIPELINE_ENDPOINT = os.environ.get("PIPELINE_ENDPOINT", "http://167.71.235.127:32084")
# print('endpoint: ', PIPELINE_ENDPOINT)

EXPERIMENT_NAME = "full-pipeline-experiments"

def submit_pipeline():
    client = kfp.Client(host=PIPELINE_ENDPOINT)

    try:
        experiment = client.get_experiment(experiment_name=EXPERIMENT_NAME)
        print('experiment: ', experiment)
    except Exception:
        experiment = client.create_experiment(name=EXPERIMENT_NAME)
        print('experiment: ', experiment)
    
    run = client.create_run_from_pipeline_func(
        pipeline_func=full_pipeline,
        arguments={},
        experiment_name=EXPERIMENT_NAME,
        run_name="emp-attrition-pipeline"
    )

    print('run-id: ', run.run_id)
    print(f'run url: {PIPELINE_ENDPOINT}/#/runs/details/{run.run_id}')

if __name__ == "__main__":
    submit_pipeline()