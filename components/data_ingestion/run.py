"""Data Ingestion component.

Reads raw source files (HouseTS.csv + usmetros.csv) and logs them as
versioned artifacts in W&B and MLflow. This is the entry point of the pipeline.

Input:  local file paths (raw_data, reference_data)
Output: W&B artifact "house_ts_raw" + MLflow artifact files
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import mlflow
import pandas as pd
import wandb
from loguru import logger

from mlflow_utils import build_component_tags, get_or_create_experiment

COMPONENT_NAME = "data_ingestion"
WANDB_PROJECT = "regression-mlops-e2e"
MLFLOW_EXPERIMENT = "regression-mlops-e2e-v2"


def go(args: argparse.Namespace) -> None:
    raw_df = pd.read_csv(args.raw_data)
    ref_df = pd.read_csv(args.reference_data)
    logger.info("raw dataset: {}", raw_df.shape)
    logger.info("reference dataset: {}", ref_df.shape)

    experiment_id = get_or_create_experiment(MLFLOW_EXPERIMENT)

    with wandb.init(
        project=WANDB_PROJECT,
        job_type=COMPONENT_NAME,
        config=vars(args),
        tags=["v2", COMPONENT_NAME],
    ) as run:
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=COMPONENT_NAME,
            tags=build_component_tags(
                component=COMPONENT_NAME,
                input_artifact=args.raw_data,
                pipeline_run_id=args.pipeline_run_id,
            ),
        ):
            mlflow.log_params({
                "raw_data_path": args.raw_data,
                "reference_data_path": args.reference_data,
                "raw_rows": raw_df.shape[0],
                "raw_columns": raw_df.shape[1],
                "reference_rows": ref_df.shape[0],
            })
            mlflow.log_artifact(args.raw_data, artifact_path="raw_data")
            mlflow.log_artifact(args.reference_data, artifact_path="raw_data")

            artifact = wandb.Artifact(
                name=args.output_artifact,
                type=args.output_type,
                description=args.output_description,
                metadata={
                    "raw_shape": list(raw_df.shape),
                    "reference_shape": list(ref_df.shape),
                    "raw_columns": raw_df.columns.tolist(),
                    "component": COMPONENT_NAME,
                    "pipeline_run_id": args.pipeline_run_id,
                },
            )
            artifact.add_file(args.raw_data)
            artifact.add_file(args.reference_data)
            run.log_artifact(artifact)

    logger.info("data_ingestion complete. Artifact '{}' logged.", args.output_artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest raw data files and log them as versioned artifacts."
    )
    parser.add_argument("--raw_data", type=str, required=True,
                        help="Path to HouseTS.csv")
    parser.add_argument("--reference_data", type=str, required=True,
                        help="Path to usmetros.csv")
    parser.add_argument("--output_artifact", type=str, required=True,
                        help="W&B artifact name for the raw dataset")
    parser.add_argument("--output_type", type=str, required=True,
                        help="W&B artifact type (e.g. raw_dataset)")
    parser.add_argument("--output_description", type=str, required=True,
                        help="Human-readable description of the artifact")
    parser.add_argument("--pipeline_run_id", type=str, default="",
                        help="Shared ID that correlates all component runs in a pipeline execution")
    args = parser.parse_args()
    go(args)
