"""Data Validation component.

Downloads the raw artifact from W&B, validates schema and data quality,
and logs a validation report as an artifact in W&B and MLflow.

Input:  W&B artifact "house_ts_raw:latest"
Output: W&B artifact "house_ts_validation" (validation_report.json)
        MLflow metrics: schema_valid, total_nulls, price_out_of_range, validation_passed
"""

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import mlflow
import pandas as pd
import wandb
from loguru import logger

from mlflow_utils import build_component_tags, get_or_create_experiment

COMPONENT_NAME = "data_validation"
WANDB_PROJECT = "regression-mlops-e2e"
MLFLOW_EXPERIMENT = "regression-mlops-e2e-v2"

EXPECTED_COLUMNS = {
    "date", "city_full", "city", "metro_full",
    "zipcode", "price", "median_list_price",
}
MIN_PRICE = 50_000
MAX_PRICE = 50_000_000


def _validate(df: pd.DataFrame) -> dict:
    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    schema_valid = len(missing_cols) == 0

    checkable = list(EXPECTED_COLUMNS & set(df.columns))
    total_nulls = int(df[checkable].isnull().sum().sum())

    price_out_of_range = 0
    if "price" in df.columns:
        prices = pd.to_numeric(df["price"], errors="coerce")
        price_out_of_range = int(((prices < MIN_PRICE) | (prices > MAX_PRICE)).sum())

    date_format_valid = True
    if "date" in df.columns:
        try:
            pd.to_datetime(df["date"])
        except Exception:
            date_format_valid = False

    passed = (
        schema_valid
        and total_nulls == 0
        and price_out_of_range == 0
        and date_format_valid
    )

    return {
        "schema_valid": schema_valid,
        "missing_columns": list(missing_cols),
        "total_nulls": total_nulls,
        "price_out_of_range": price_out_of_range,
        "date_format_valid": date_format_valid,
        "total_rows": len(df),
        "passed": passed,
    }


def go(args: argparse.Namespace) -> None:
    experiment_id = get_or_create_experiment(MLFLOW_EXPERIMENT)

    with wandb.init(
        project=WANDB_PROJECT,
        job_type=COMPONENT_NAME,
        config=vars(args),
        tags=["v2", COMPONENT_NAME],
    ) as run:
        artifact = run.use_artifact(args.input_artifact, type="raw_dataset")
        artifact_dir = artifact.download()

        csv_files = [f for f in os.listdir(artifact_dir) if "House" in f]
        df = pd.read_csv(os.path.join(artifact_dir, csv_files[0]))

        results = _validate(df)
        status = "PASSED" if results["passed"] else "FAILED"
        logger.info("Validation {}: {}", status, results)

        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = os.path.join(tmp_dir, "validation_report.json")
            with open(report_path, "w") as f:
                json.dump(results, f, indent=2)

            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=COMPONENT_NAME,
                tags=build_component_tags(
                    component=COMPONENT_NAME,
                    input_artifact=args.input_artifact,
                    pipeline_run_id=args.pipeline_run_id,
                ),
            ):
                mlflow.log_params({"input_artifact": args.input_artifact})
                mlflow.log_metrics({
                    "schema_valid": int(results["schema_valid"]),
                    "total_nulls": results["total_nulls"],
                    "price_out_of_range": results["price_out_of_range"],
                    "date_format_valid": int(results["date_format_valid"]),
                    "total_rows": results["total_rows"],
                    "validation_passed": int(results["passed"]),
                })
                mlflow.log_artifact(report_path, artifact_path="validation")

            out_artifact = wandb.Artifact(
                name=args.output_artifact,
                type=args.output_type,
                description=args.output_description,
                metadata={
                    "validation_passed": results["passed"],
                    "total_rows": results["total_rows"],
                    "total_nulls": results["total_nulls"],
                    "component": COMPONENT_NAME,
                    "pipeline_run_id": args.pipeline_run_id,
                },
            )
            out_artifact.add_file(report_path)
            run.log_artifact(out_artifact)

    if not results["passed"]:
        raise ValueError(f"Data validation FAILED: {results}")

    logger.info("data_validation complete. Artifact '{}' logged.", args.output_artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate raw dataset schema and data quality."
    )
    parser.add_argument("--input_artifact", type=str, required=True,
                        help="W&B artifact name for the raw dataset (e.g. house_ts_raw:latest)")
    parser.add_argument("--output_artifact", type=str, required=True,
                        help="W&B artifact name for the validation report")
    parser.add_argument("--output_type", type=str, required=True)
    parser.add_argument("--output_description", type=str, required=True)
    parser.add_argument("--pipeline_run_id", type=str, default="")
    args = parser.parse_args()
    go(args)
