from loguru import logger

from src.logging_config import setup_logger
from src.models.tune_model import tune_xgboost_model


def run_tuning_pipeline(configure_logging: bool = True) -> None:
    """Runs the tuning pipeline for the selected baseline model."""
    if configure_logging:
        setup_logger()

    best_params, metrics = tune_xgboost_model()

    file_logger = logger.bind(file_only=True)
    file_logger.info(
        "Tuning pipeline summary\n"
        "Best params: {}\n"
        "Train metrics: {}\n"
        "Eval metrics: {}\n"
        "Generalization gap: {}",
        best_params,
        metrics["train"],
        metrics["eval"],
        metrics["gap"],
    )


if __name__ == "__main__":
    run_tuning_pipeline()
