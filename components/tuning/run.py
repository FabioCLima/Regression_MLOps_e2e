"""Tuning component.

Uses Hydra for config management (search space, n_trials, direction) and Optuna
for hyperparameter optimization of the XGBoost model. Only runs if the best
baseline model is XGBoost.

Input:  W&B artifact "house_ts_features:latest" + "house_ts_best_model:latest"
Output: W&B artifact "house_ts_tuned_model" (xgboost_tuned_model.pkl + tuning_metrics.json)
        MLflow params: best hyperparameters found by Optuna
        MLflow metrics: tuned RMSE vs. baseline RMSE

Config (Hydra):
    components/tuning/conf/config.yaml          — base config
    components/tuning/conf/xgboost/default.yaml — standard search space
    components/tuning/conf/xgboost/extended.yaml — wider search space
"""

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import hydra
import mlflow
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from mlflow_utils import build_component_tags, get_or_create_experiment
from src.models.train_model import load_feature_splits
from src.models.tune_model import tune_model

COMPONENT_NAME = "tuning"
WANDB_PROJECT = "regression-mlops-e2e"
MLFLOW_EXPERIMENT = "regression-mlops-e2e-v2"


def _run_inner(cfg: DictConfig, args: argparse.Namespace) -> None:
    """Core logic extracted from main() so it can be called directly in tests
    without the @hydra.main decorator."""
    experiment_id = get_or_create_experiment(MLFLOW_EXPERIMENT)

    with wandb.init(
        project=WANDB_PROJECT,
        job_type=COMPONENT_NAME,
        config={**vars(args), **OmegaConf.to_container(cfg, resolve=True)},
        tags=["v2", COMPONENT_NAME],
    ) as run:
        feat_artifact = run.use_artifact(args.input_artifact, type="feature_dataset")
        feat_dir = feat_artifact.download()

        run.use_artifact(args.input_model, type="model").download()

        train_df, eval_df = load_feature_splits(feat_dir)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tuning_results = tune_model(
                train_df=train_df,
                eval_df=eval_df,
                output_dir=tmp_dir,
                n_trials=cfg.tuning.n_trials,
                direction=cfg.tuning.direction,
                register_artifact=False,
            )

            best_params = tuning_results["best_params"]
            tuned_rmse = tuning_results["tuned_metrics"]["rmse"]

            logger.info("Tuning complete. Best RMSE: {:.2f}", tuned_rmse)

            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=COMPONENT_NAME,
                tags=build_component_tags(
                    component=COMPONENT_NAME,
                    input_artifact=args.input_artifact,
                    pipeline_run_id=args.pipeline_run_id,
                ),
            ):
                # Log the full resolved Hydra config — this is what answers
                # "what search space decision was used for this run?"
                mlflow.log_dict(
                    OmegaConf.to_container(cfg, resolve=True),
                    "hydra_config.yaml",
                )
                mlflow.log_params({
                    "n_trials": cfg.tuning.n_trials,
                    "direction": cfg.tuning.direction,
                    "input_artifact": args.input_artifact,
                    **{f"best_{k}": v for k, v in best_params.items()},
                })
                mlflow.log_metrics({"tuned_eval_rmse": tuned_rmse})
                mlflow.log_artifacts(tmp_dir, artifact_path="tuned_model")

            out_artifact = wandb.Artifact(
                name=args.output_artifact,
                type=args.output_type,
                description=args.output_description,
                metadata={
                    "tuned_eval_rmse": tuned_rmse,
                    "n_trials": cfg.tuning.n_trials,
                    "best_params": best_params,
                    "component": COMPONENT_NAME,
                    "pipeline_run_id": args.pipeline_run_id,
                },
            )
            for fname in os.listdir(tmp_dir):
                out_artifact.add_file(os.path.join(tmp_dir, fname))
            run.log_artifact(out_artifact)

    logger.info("tuning complete. Artifact '{}' logged.", args.output_artifact)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # argparse handles artifact pointers; Hydra handles the search space config
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input_artifact", type=str, required=True)
    parser.add_argument("--input_model", type=str, required=True)
    parser.add_argument("--output_artifact", type=str, required=True)
    parser.add_argument("--output_type", type=str, required=True)
    parser.add_argument("--output_description", type=str, required=True)
    parser.add_argument("--pipeline_run_id", type=str, default="")
    args, _ = parser.parse_known_args()
    _run_inner(cfg, args)


if __name__ == "__main__":
    main()
