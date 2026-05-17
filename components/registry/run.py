"""Registry component.

Downloads the tuned model from W&B and registers it in the MLflow Model Registry
with the specified stage. This is what enables formal promotion workflows
(Staging → Production) and gives the model a versioned, queryable identity
in MLflow beyond a raw run artifact.

Input:  W&B artifact "house_ts_tuned_model:latest"
Output: MLflow Registered Model (name: house-price-regressor, stage: Staging)
        W&B artifact tag: registered_model_uri
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import joblib
import mlflow
import mlflow.sklearn
import wandb
from loguru import logger

from mlflow_utils import build_component_tags, get_or_create_experiment

COMPONENT_NAME = "registry"
WANDB_PROJECT = "regression-mlops-e2e"
MLFLOW_EXPERIMENT = "regression-mlops-e2e-v2"


def go(args: argparse.Namespace) -> None:
    experiment_id = get_or_create_experiment(MLFLOW_EXPERIMENT)

    with wandb.init(
        project=WANDB_PROJECT,
        job_type=COMPONENT_NAME,
        config=vars(args),
        tags=["v2", COMPONENT_NAME],
    ) as run:
        model_artifact = run.use_artifact(args.input_model, type="model")
        model_dir = model_artifact.download()

        model_file = next(
            f for f in os.listdir(model_dir) if f.endswith(".pkl") and "tuned" in f
        )
        model = joblib.load(os.path.join(model_dir, model_file))

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=COMPONENT_NAME,
            tags=build_component_tags(
                component=COMPONENT_NAME,
                input_artifact=args.input_model,
                pipeline_run_id=args.pipeline_run_id,
            ),
        ) as active_run:
            mlflow.log_params({
                "model_name": args.model_name,
                "stage": args.stage,
                "input_model": args.input_model,
            })

            # Register model in MLflow Model Registry
            model_uri = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=args.model_name,
            ).model_uri

            # Transition to the requested stage
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_latest_versions(
                args.model_name, stages=["None"]
            )[0].version
            client.transition_model_version_stage(
                name=args.model_name,
                version=model_version,
                stage=args.stage,
            )

            registered_uri = f"models:/{args.model_name}/{model_version}"
            mlflow.set_tag("registered_model_uri", registered_uri)
            mlflow.set_tag("model_stage", args.stage)

            logger.info(
                "Model registered: {} | Stage: {} | URI: {}",
                args.model_name, args.stage, registered_uri,
            )

            # Tag the W&B artifact with the MLflow registry URI for cross-tool lineage
            model_artifact.description = (
                f"Registered in MLflow as '{registered_uri}' (stage: {args.stage})"
            )

    logger.info("registry complete. Model '{}' promoted to {}.", args.model_name, args.stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Register the tuned model in the MLflow Model Registry."
    )
    parser.add_argument("--input_model", type=str, required=True,
                        help="W&B artifact name for the tuned model (e.g. house_ts_tuned_model:latest)")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name to register the model under in MLflow Model Registry")
    parser.add_argument("--stage", type=str, default="Staging",
                        choices=["Staging", "Production", "Archived"],
                        help="MLflow model stage to transition to after registration")
    parser.add_argument("--pipeline_run_id", type=str, default="")
    args = parser.parse_args()
    go(args)
