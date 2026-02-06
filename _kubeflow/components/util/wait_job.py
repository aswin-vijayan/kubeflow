from kfp.dsl import component

@component(
    base_image="sandy345/kubeflow-employee-attrition:latest",
    packages_to_install=['kubernetes']
)
def wait_for_training(job_name: str, namespace: str):
    from kubernetes import client, config
    import time

    config.load_incluster_config()
    api = client.CustomObjectsApi()

    time.sleep(5)
    poll_interval = 30
    elapsed = 0

    while elapsed < 360:
        job = api.get_namespaced_custom_object(
            group="trainer.kubeflow.org",
            version="v1alpha1",
            namespace=namespace,
            plural="trainjobs",
            name=job_name,
        )

        print('job: ', job, flush=True)

        status = job.get("status", {})
        print('status: ', status, flush=True)

        conditions = status.get("conditions", [])
        print('conditions: ', conditions)

        if not conditions:
            print(f"[{job_name}] Waiting... ({elapsed}s elapsed)", flush=True)
        else:
            for condition in conditions:
                cond_type = condition.get("type", "")
                cond_status = condition.get("status", "")
                cond_message = condition.get("message", "")

                if cond_type == "Succeeded" and cond_status == "True":
                    print(f"TrainJob [{job_name}] ✅ Succeeded", flush=True)
                    return

                elif cond_type == "Failed" and cond_status == "True":
                    raise Exception(f"TrainJob [{job_name}] failed: {cond_message}")

                elif cond_type == "Complete" and cond_status == "True":
                    print(f"TrainJob [{job_name}] completed: {cond_message}")
                    return
            
        time.sleep(poll_interval)
        elapsed += poll_interval

    raise Exception(f"TrainJob '{job_name}' timed out after 1800s")

