from loguru import logger

from src.features.feature_engineering import run_feature_engineering
from src.logging_config import setup_logger


def run_feature_pipeline(configure_logging: bool = True) -> None:
    """Runs the feature pipeline from cleaned data to model-ready features."""
    if configure_logging:
        setup_logger()

    train_df, eval_df, holdout_df, frequency_map, target_encoder = (
        run_feature_engineering()
    )

    file_logger = logger.bind(file_only=True)
    file_logger.info(
        "Feature pipeline summary\n"
        "Train shape: {}\n"
        "Eval shape: {}\n"
        "Holdout shape: {}\n"
        "Frequency encoder fitted: {}\n"
        "Target encoder fitted: {}",
        train_df.shape,
        eval_df.shape,
        holdout_df.shape,
        frequency_map is not None,
        target_encoder is not None,
    )


if __name__ == "__main__":
    run_feature_pipeline()
