import argparse
from pathlib import Path

from loguru import logger

from src.config import inference_config, paths, split_config
from src.inference.predict import run_inference
from src.logging_config import setup_logger


def run_inference_pipeline(
    input_path: Path | str = paths.processed_data_dir / split_config.holdout_filename,
    output_dir: Path | str = paths.processed_data_dir,
    model_path: Path | str | None = None,
    configure_logging: bool = True,
) -> None:
    """Runs inference on a raw-compatible input CSV."""
    if configure_logging:
        setup_logger()

    predictions, metrics = run_inference(
        input_path=input_path,
        output_dir=output_dir,
        model_path=model_path,
    )

    file_logger = logger.bind(file_only=True)
    file_logger.info(
        "Inference pipeline summary\n"
        "Input path: {}\n"
        "Output path: {}\n"
        "Predictions shape: {}\n"
        "Metrics: {}",
        input_path,
        Path(output_dir) / inference_config.predictions_filename,
        predictions.shape,
        metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run housing price inference.")
    parser.add_argument(
        "--input",
        default=str(paths.processed_data_dir / split_config.holdout_filename),
        help="Path to raw-compatible input CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(paths.processed_data_dir),
        help="Directory where predictions will be saved.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model path. Defaults to tuned model if present, otherwise best model.",
    )
    args = parser.parse_args()

    run_inference_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        model_path=args.model,
    )


if __name__ == "__main__":
    main()
