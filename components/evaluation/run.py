"""Evaluation component.

Downloads the tuned model and feature dataset from W&B, runs inference on the
holdout split, and logs evaluation metrics. This is the only place where the
holdout set is used — keeping it isolated from training and tuning.

Input:  W&B artifact "house_ts_tuned_model:latest" + "house_ts_features:latest"
Output: W&B artifact "house_ts_eval_report" (evaluation_report.json)
        MLflow metrics: holdout RMSE, MAE, R²
"""

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import joblib
import mlflow
import numpy as np
import pandas as pd
import wandb
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mlflow_utils import build_component_tags, get_or_create_experiment
from src.features.feature_engineering import feature_engineered_split_filename

COMPONENT_NAME = "evaluation"
WANDB_PROJECT = "regression-mlops-e2e"
MLFLOW_EXPERIMENT = "regression-mlops-e2e-v2"
TARGET_COLUMN = "price"


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

        feat_artifact = run.use_artifact(args.input_features, type="feature_dataset")
        feat_dir = feat_artifact.download()

        model_file = next(
            f for f in os.listdir(model_dir)
            if f.endswith(".pkl") and "tuned" in f
        )
        model = joblib.load(os.path.join(model_dir, model_file))

        holdout_path = os.path.join(
            feat_dir, feature_engineered_split_filename("holdout")
        )
        holdout_df = pd.read_csv(holdout_path)

        X = holdout_df.drop(columns=[TARGET_COLUMN])
        y_true = holdout_df[TARGET_COLUMN]
        y_pred = model.predict(X)

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        logger.info("Holdout evaluation — RMSE: {:.2f} | MAE: {:.2f} | R²: {:.4f}",
                    rmse, mae, r2)

        report = {
            "holdout_rmse": rmse,
            "holdout_mae": mae,
            "holdout_r2": r2,
            "n_samples": len(holdout_df),
            "input_model": args.input_model,
            "input_features": args.input_features,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = os.path.join(tmp_dir, "evaluation_report.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=COMPONENT_NAME,
                tags={
                    **build_component_tags(
                        component=COMPONENT_NAME,
                        input_artifact=args.input_model,
                        pipeline_run_id=args.pipeline_run_id,
                    ),
                    "input_features": args.input_features,
                },
            ):
                mlflow.log_params({
                    "input_model": args.input_model,
                    "input_features": args.input_features,
                    "target_column": TARGET_COLUMN,
                    "split_evaluated": "holdout",
                })
                mlflow.log_metrics({
                    "holdout_rmse": rmse,
                    "holdout_mae": mae,
                    "holdout_r2": r2,
                })
                mlflow.log_artifact(report_path, artifact_path="evaluation")

            out_artifact = wandb.Artifact(
                name=args.output_artifact,
                type=args.output_type,
                description=args.output_description,
                metadata={
                    "holdout_rmse": rmse,
                    "holdout_mae": mae,
                    "holdout_r2": r2,
                    "n_samples": len(holdout_df),
                    "component": COMPONENT_NAME,
                    "pipeline_run_id": args.pipeline_run_id,
                },
            )
            out_artifact.add_file(report_path)
            run.log_artifact(out_artifact)

    logger.info("evaluation complete. Artifact '{}' logged.", args.output_artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the tuned model on the holdout split."
    )
    parser.add_argument("--input_model", type=str, required=True,
                        help="W&B artifact name for the tuned model (e.g. house_ts_tuned_model:latest)")
    parser.add_argument("--input_features", type=str, required=True,
                        help="W&B artifact name for the feature dataset (e.g. house_ts_features:latest)")
    parser.add_argument("--output_artifact", type=str, required=True,
                        help="W&B artifact name for the evaluation report")
    parser.add_argument("--output_type", type=str, required=True)
    parser.add_argument("--output_description", type=str, required=True)
    parser.add_argument("--pipeline_run_id", type=str, default="")
    args = parser.parse_args()
    go(args)
