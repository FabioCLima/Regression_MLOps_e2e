"""Training component.

Downloads feature-engineered splits from W&B, trains all candidate models
(Dummy, LinearRegression, Ridge, RandomForest, XGBoost), selects the best
by RMSE on the eval split, and logs the best model as an artifact.

Input:  W&B artifact "house_ts_features:latest"
Output: W&B artifact "house_ts_best_model" (best_model.pkl + metrics.json)
        MLflow metrics: rmse/mae/r2 for every candidate model
"""

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import mlflow
import wandb
from loguru import logger

from mlflow_utils import build_component_tags, get_or_create_experiment
from src.models.train_model import load_feature_splits, train_and_compare_models

COMPONENT_NAME = "training"
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
        artifact = run.use_artifact(args.input_artifact, type="feature_dataset")
        artifact_dir = artifact.download()

        train_df, eval_df = load_feature_splits(artifact_dir)

        with tempfile.TemporaryDirectory() as tmp_dir:
            results = train_and_compare_models(
                train_df=train_df,
                eval_df=eval_df,
                output_dir=tmp_dir,
                register_artifact=False,
            )

            best_model_name = results["best_model_name"]
            best_metrics = results["metrics"][best_model_name]
            all_metrics = results["metrics"]

            logger.info(
                "Best model: {} | RMSE: {:.2f}", best_model_name, best_metrics["rmse"]
            )

            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=COMPONENT_NAME,
                tags={
                    **build_component_tags(
                        component=COMPONENT_NAME,
                        input_artifact=args.input_artifact,
                        pipeline_run_id=args.pipeline_run_id,
                    ),
                    "best_model": best_model_name,
                },
            ):
                mlflow.log_params({
                    "input_artifact": args.input_artifact,
                    "best_model": best_model_name,
                    "candidates": ",".join(all_metrics.keys()),
                    "primary_metric": "rmse",
                })
                # Log metrics for every candidate — full comparison preserved
                for model_name, metrics in all_metrics.items():
                    mlflow.log_metrics({
                        f"{model_name}_eval_rmse": metrics.get("rmse", 0),
                        f"{model_name}_eval_mae": metrics.get("mae", 0),
                        f"{model_name}_eval_r2": metrics.get("r2", 0),
                    })
                mlflow.log_artifacts(tmp_dir, artifact_path="model")

            out_artifact = wandb.Artifact(
                name=args.output_artifact,
                type=args.output_type,
                description=args.output_description,
                metadata={
                    "best_model_name": best_model_name,
                    "eval_rmse": best_metrics.get("rmse"),
                    "eval_mae": best_metrics.get("mae"),
                    "eval_r2": best_metrics.get("r2"),
                    "candidates_compared": list(all_metrics.keys()),
                    "primary_metric": "rmse",
                    "component": COMPONENT_NAME,
                    "pipeline_run_id": args.pipeline_run_id,
                },
            )
            for fname in os.listdir(tmp_dir):
                out_artifact.add_file(os.path.join(tmp_dir, fname))
            run.log_artifact(out_artifact)

    logger.info("training complete. Best model: {} logged.", best_model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train candidate models and select the best by eval RMSE."
    )
    parser.add_argument("--input_artifact", type=str, required=True,
                        help="W&B artifact name for the feature dataset (e.g. house_ts_features:latest)")
    parser.add_argument("--output_artifact", type=str, required=True,
                        help="W&B artifact name for the best model")
    parser.add_argument("--output_type", type=str, required=True)
    parser.add_argument("--output_description", type=str, required=True)
    parser.add_argument("--pipeline_run_id", type=str, default="")
    args = parser.parse_args()
    go(args)
