from loguru import logger

from src.logging_config import setup_logger
from src.models.train_model import run_model_training


def run_training_pipeline(configure_logging: bool = True) -> None:
    """Runs the training pipeline from engineered features to model artifacts."""
    if configure_logging:
        setup_logger()

    _, metrics_by_model, best_model_name = run_model_training()

    file_logger = logger.bind(file_only=True)
    file_logger.info(
        "Training pipeline summary\n"
        "Best model: {}\n"
        "Metrics by model: {}",
        best_model_name,
        metrics_by_model,
    )


if __name__ == "__main__":
    run_training_pipeline()
