import subprocess
from datetime import datetime, timezone

import mlflow


def get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def get_or_create_experiment(name: str) -> str:
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(name)


def build_component_tags(
    component: str,
    input_artifact: str,
    pipeline_run_id: str = "",
) -> dict[str, str]:
    """Returns standard MLflow tags attached to every component run.

    These tags are the primary mechanism for lineage tracing:
    given a run_id, you can always answer:
      - which component produced this artifact?
      - which pipeline execution does it belong to?
      - what was the input artifact?
      - which git commit was the code at?
    """
    return {
        "project": "regression-mlops-e2e",
        "version": "v2",
        "component": component,
        "git_sha": get_git_sha(),
        "input_artifact": input_artifact,
        "pipeline_run_id": pipeline_run_id,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }
