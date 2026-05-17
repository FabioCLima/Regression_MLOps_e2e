import argparse
from pathlib import Path

from loguru import logger

from pipelines.data_pipeline import run_data_pipeline
from pipelines.feature_pipeline import run_feature_pipeline
from pipelines.inference_pipeline import run_inference_pipeline
from pipelines.training_pipeline import run_training_pipeline
from pipelines.tuning_pipeline import run_tuning_pipeline
from src.config import paths, split_config
from src.logging_config import setup_logger


def run_machine_learning_pipeline(
    run_data: bool = True,
    run_features: bool = True,
    run_training: bool = True,
    run_tuning: bool = True,
    run_inference: bool = False,
    inference_input_path: Path | str = paths.processed_data_dir
    / split_config.holdout_filename,
    inference_output_dir: Path | str = paths.processed_data_dir,
    inference_model_path: Path | str | None = None,
) -> None:
    """Orchestrates the project pipeline from data loading to model tuning."""
    setup_logger(log_file="machine_learning_pipeline.log")

    logger.info("Starting machine learning pipeline orchestration")

    if run_data:
        run_data_pipeline(configure_logging=False)

    if run_features:
        run_feature_pipeline(configure_logging=False)

    if run_training:
        run_training_pipeline(configure_logging=False)

    if run_tuning:
        run_tuning_pipeline(configure_logging=False)

    if run_inference:
        run_inference_pipeline(
            input_path=inference_input_path,
            output_dir=inference_output_dir,
            model_path=inference_model_path,
            configure_logging=False,
        )

    logger.info("Finished machine learning pipeline orchestration")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the housing regression ML pipeline from data loading "
            "through feature engineering, training, and tuning."
        )
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data split and preprocessing.",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature engineering.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip baseline model training.",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip hyperparameter tuning.",
    )
    parser.add_argument(
        "--include-inference",
        action="store_true",
        help="Run inference after tuning.",
    )
    parser.add_argument(
        "--inference-input",
        default=str(paths.processed_data_dir / split_config.holdout_filename),
        help="Raw-compatible CSV used when --include-inference is enabled.",
    )
    parser.add_argument(
        "--inference-output-dir",
        default=str(paths.processed_data_dir),
        help="Directory where inference outputs are saved.",
    )
    parser.add_argument(
        "--inference-model",
        default=None,
        help="Optional model path used when --include-inference is enabled.",
    )
    args = parser.parse_args()

    run_machine_learning_pipeline(
        run_data=not args.skip_data,
        run_features=not args.skip_features,
        run_training=not args.skip_training,
        run_tuning=not args.skip_tuning,
        run_inference=args.include_inference,
        inference_input_path=args.inference_input,
        inference_output_dir=args.inference_output_dir,
        inference_model_path=args.inference_model,
    )


if __name__ == "__main__":
    main()
