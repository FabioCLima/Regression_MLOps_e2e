"""Feature Engineering component.

Downloads the cleaned splits from W&B, applies frequency encoding (zipcode)
and target encoding (city_full), adds date features, and logs the
feature-engineered splits and fitted encoders as artifacts.

Input:  W&B artifact "house_ts_cleaned:latest"
Output: W&B artifact "house_ts_features" (engineered CSVs + encoder .pkl files)
        MLflow params: encoding types, feature count, target column
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
from src.features.feature_engineering import run_feature_engineering

COMPONENT_NAME = "feature_engineering"
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
        artifact = run.use_artifact(args.input_artifact, type="cleaned_dataset")
        artifact_dir = artifact.download()

        with tempfile.TemporaryDirectory() as tmp_dir:
            engineered_splits = run_feature_engineering(
                input_dir=artifact_dir,
                output_dir=tmp_dir,
                register_artifact=False,
            )

            feature_count = engineered_splits["train"].shape[1]
            logger.info("Feature count after engineering: {}", feature_count)

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
                    "input_artifact": args.input_artifact,
                    "frequency_encode_column": "zipcode",
                    "target_encode_column": "city_full",
                    "date_features": "year,quarter,month",
                    "feature_count": feature_count,
                    "train_rows": len(engineered_splits["train"]),
                })
                mlflow.log_artifacts(tmp_dir, artifact_path="features")

            out_artifact = wandb.Artifact(
                name=args.output_artifact,
                type=args.output_type,
                description=args.output_description,
                metadata={
                    "feature_count": feature_count,
                    "frequency_encode_column": "zipcode",
                    "target_encode_column": "city_full",
                    "date_features": ["year", "quarter", "month"],
                    "component": COMPONENT_NAME,
                    "pipeline_run_id": args.pipeline_run_id,
                },
            )
            for fname in os.listdir(tmp_dir):
                out_artifact.add_file(os.path.join(tmp_dir, fname))
            run.log_artifact(out_artifact)

    logger.info(
        "feature_engineering complete. Artifact '{}' logged.", args.output_artifact
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply feature engineering to cleaned splits."
    )
    parser.add_argument("--input_artifact", type=str, required=True,
                        help="W&B artifact name for the cleaned dataset (e.g. house_ts_cleaned:latest)")
    parser.add_argument("--output_artifact", type=str, required=True,
                        help="W&B artifact name for the feature-engineered dataset")
    parser.add_argument("--output_type", type=str, required=True)
    parser.add_argument("--output_description", type=str, required=True)
    parser.add_argument("--pipeline_run_id", type=str, default="")
    args = parser.parse_args()
    go(args)
