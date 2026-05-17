"""Pipeline orchestrator for Regression MLOps v2.

Each component is executed via mlflow.run(), which isolates it in its own
virtualenv (defined by python_env.yaml) and tracks it as a child run.

Usage:
    # Full pipeline
    python orchestrator.py

    # Specific steps only
    python orchestrator.py --steps data_ingestion preprocessing feature_engineering

    # Custom experiment name
    python orchestrator.py --experiment-name my-experiment

    # Resume with a specific pipeline_run_id (links runs across executions)
    python orchestrator.py --steps training tuning --pipeline-run-id abc12345
"""

import argparse
import uuid

import mlflow

from mlflow_utils import get_or_create_experiment

MLFLOW_EXPERIMENT = "regression-mlops-e2e-v2"

ARTIFACT_NAMES = {
    "raw": "house_ts_raw:latest",
    "validation": "house_ts_validation:latest",
    "cleaned": "house_ts_cleaned:latest",
    "features": "house_ts_features:latest",
    "best_model": "house_ts_best_model:latest",
    "tuned_model": "house_ts_tuned_model:latest",
    "eval_report": "house_ts_eval_report:latest",
}

ALL_STEPS = [
    "data_ingestion",
    "data_validation",
    "preprocessing",
    "feature_engineering",
    "training",
    "tuning",
    "evaluation",
    "registry",
]


def _run(component: str, params: dict) -> None:
    mlflow.run(
        f"components/{component}",
        entry_point="main",
        parameters=params,
        env_manager="virtualenv",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regression MLOps v2 pipeline orchestrator."
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS + ["all"],
        default=["all"],
        help="Pipeline steps to execute (default: all).",
    )
    parser.add_argument(
        "--experiment-name",
        default=MLFLOW_EXPERIMENT,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--pipeline-run-id",
        default="",
        help="Shared ID linking all component runs (auto-generated if omitted).",
    )
    args = parser.parse_args()

    steps = ALL_STEPS if "all" in args.steps else args.steps
    pipeline_run_id = args.pipeline_run_id or uuid.uuid4().hex[:8]
    experiment_id = get_or_create_experiment(args.experiment_name)

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"pipeline_{pipeline_run_id}",
        tags={
            "type": "pipeline",
            "version": "v2",
            "pipeline_run_id": pipeline_run_id,
            "steps": ",".join(steps),
        },
    ):
        if "data_ingestion" in steps:
            _run("data_ingestion", {
                "raw_data": "data/raw_data/HouseTS.csv",
                "reference_data": "data/raw_data/usmetros.csv",
                "output_artifact": "house_ts_raw",
                "output_type": "raw_dataset",
                "output_description": "Raw house time series + geospatial reference data",
                "pipeline_run_id": pipeline_run_id,
            })

        if "data_validation" in steps:
            _run("data_validation", {
                "input_artifact": ARTIFACT_NAMES["raw"],
                "output_artifact": "house_ts_validation",
                "output_type": "validation_report",
                "output_description": "Schema and quality validation report",
                "pipeline_run_id": pipeline_run_id,
            })

        if "preprocessing" in steps:
            _run("preprocessing", {
                "input_artifact": ARTIFACT_NAMES["raw"],
                "output_artifact": "house_ts_cleaned",
                "output_type": "cleaned_dataset",
                "output_description": "Temporally split and cleaned dataset",
                "eval_start_date": "2020-01-01",
                "holdout_start_date": "2022-01-01",
                "pipeline_run_id": pipeline_run_id,
            })

        if "feature_engineering" in steps:
            _run("feature_engineering", {
                "input_artifact": ARTIFACT_NAMES["cleaned"],
                "output_artifact": "house_ts_features",
                "output_type": "feature_dataset",
                "output_description": "Feature-engineered splits with fitted encoders",
                "pipeline_run_id": pipeline_run_id,
            })

        if "training" in steps:
            _run("training", {
                "input_artifact": ARTIFACT_NAMES["features"],
                "output_artifact": "house_ts_best_model",
                "output_type": "model",
                "output_description": "Best model from baseline candidate comparison",
                "pipeline_run_id": pipeline_run_id,
            })

        if "tuning" in steps:
            _run("tuning", {
                "input_artifact": ARTIFACT_NAMES["features"],
                "input_model": ARTIFACT_NAMES["best_model"],
                "output_artifact": "house_ts_tuned_model",
                "output_type": "model",
                "output_description": "XGBoost model tuned with Optuna",
                "hydra_config": "xgboost/default",
                "pipeline_run_id": pipeline_run_id,
            })

        if "evaluation" in steps:
            _run("evaluation", {
                "input_model": ARTIFACT_NAMES["tuned_model"],
                "input_features": ARTIFACT_NAMES["features"],
                "output_artifact": "house_ts_eval_report",
                "output_type": "evaluation_report",
                "output_description": "Holdout evaluation metrics for the tuned model",
                "pipeline_run_id": pipeline_run_id,
            })

        if "registry" in steps:
            _run("registry", {
                "input_model": ARTIFACT_NAMES["tuned_model"],
                "model_name": "house-price-regressor",
                "stage": "Staging",
                "pipeline_run_id": pipeline_run_id,
            })


if __name__ == "__main__":
    main()
