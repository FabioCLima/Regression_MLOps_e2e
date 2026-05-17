"""Preprocessing component.

Downloads the raw artifact from W&B, applies temporal split and preprocessing
transformations (city normalization, metro merge, deduplication, outlier removal),
and logs the cleaned splits as a new artifact.

Input:  W&B artifact "house_ts_raw:latest"
Output: W&B artifact "house_ts_cleaned" (train/eval/holdout CSVs)
        MLflow params: split dates, row counts per split
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
from src.data.preprocess_data import preprocess_dataset
from src.data.split_data import save_data_splits, split_dataset_by_date

COMPONENT_NAME = "preprocessing"
WANDB_PROJECT = "regression-mlops-e2e"
MLFLOW_EXPERIMENT = "regression-mlops-e2e-v2"
SPLITS = ("train", "eval", "holdout")


def go(args: argparse.Namespace) -> None:
    import pandas as pd

    experiment_id = get_or_create_experiment(MLFLOW_EXPERIMENT)

    with wandb.init(
        project=WANDB_PROJECT,
        job_type=COMPONENT_NAME,
        config=vars(args),
        tags=["v2", COMPONENT_NAME],
    ) as run:
        artifact = run.use_artifact(args.input_artifact, type="raw_dataset")
        artifact_dir = artifact.download()

        raw_csv = next(
            f for f in os.listdir(artifact_dir) if "HouseTS" in f or f.endswith(".csv")
        )
        metros_csv = next(
            (f for f in os.listdir(artifact_dir) if "metro" in f.lower()), None
        )

        df = pd.read_csv(os.path.join(artifact_dir, raw_csv))
        metros_path = os.path.join(artifact_dir, metros_csv) if metros_csv else None

        train_df, eval_df, holdout_df = split_dataset_by_date(
            dataset=df,
            eval_start_date=args.eval_start_date,
            holdout_start_date=args.holdout_start_date,
        )

        splits = {"train": train_df, "eval": eval_df, "holdout": holdout_df}
        cleaned = {
            name: preprocess_dataset(df, metros_path=metros_path)
            for name, df in splits.items()
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_data_splits(
                cleaned["train"],
                cleaned["eval"],
                cleaned["holdout"],
                output_dir=tmp_dir,
            )

            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=COMPONENT_NAME,
                tags=build_component_tags(
                    component=COMPONENT_NAME,
                    input_artifact=args.input_artifact,
                    pipeline_run_id=args.pipeline_run_id,
                ),
            ):
                mlflow.log_params({
                    "eval_start_date": args.eval_start_date,
                    "holdout_start_date": args.holdout_start_date,
                    "train_rows": len(cleaned["train"]),
                    "eval_rows": len(cleaned["eval"]),
                    "holdout_rows": len(cleaned["holdout"]),
                    "input_artifact": args.input_artifact,
                })
                mlflow.log_artifacts(tmp_dir, artifact_path="cleaned_splits")

            out_artifact = wandb.Artifact(
                name=args.output_artifact,
                type=args.output_type,
                description=args.output_description,
                metadata={
                    "eval_start_date": args.eval_start_date,
                    "holdout_start_date": args.holdout_start_date,
                    "train_rows": len(cleaned["train"]),
                    "eval_rows": len(cleaned["eval"]),
                    "holdout_rows": len(cleaned["holdout"]),
                    "component": COMPONENT_NAME,
                    "pipeline_run_id": args.pipeline_run_id,
                },
            )
            for fname in os.listdir(tmp_dir):
                out_artifact.add_file(os.path.join(tmp_dir, fname))
            run.log_artifact(out_artifact)

    logger.info("preprocessing complete. Artifact '{}' logged.", args.output_artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Temporal split and preprocessing of the raw housing dataset."
    )
    parser.add_argument("--input_artifact", type=str, required=True,
                        help="W&B artifact name (e.g. house_ts_raw:latest)")
    parser.add_argument("--output_artifact", type=str, required=True,
                        help="W&B artifact name for the cleaned splits")
    parser.add_argument("--output_type", type=str, required=True)
    parser.add_argument("--output_description", type=str, required=True)
    parser.add_argument("--eval_start_date", type=str, default="2020-01-01",
                        help="First date of the eval split (temporal boundary)")
    parser.add_argument("--holdout_start_date", type=str, default="2022-01-01",
                        help="First date of the holdout split (temporal boundary)")
    parser.add_argument("--pipeline_run_id", type=str, default="")
    args = parser.parse_args()
    go(args)
