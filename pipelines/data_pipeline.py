from loguru import logger

from src.config import paths
from src.data.preprocess_data import run_preprocessing
from src.data.split_data import load_and_split_data
from src.logging_config import setup_logger


def run_data_pipeline(configure_logging: bool = True) -> None:
    """Runs the data pipeline from raw artifact to processed data splits."""
    if configure_logging:
        setup_logger()

    train_df, eval_df, holdout_df = load_and_split_data()
    cleaned_splits = run_preprocessing(input_dir=paths.processed_data_dir)

    file_logger = logger.bind(file_only=True)
    file_logger.info(
        "Data split summary\n"
        "Train shape: {}\n"
        "Eval shape: {}\n"
        "Holdout shape: {}\n"
        "Train date range: {} to {}\n"
        "Eval date range: {} to {}\n"
        "Holdout date range: {} to {}",
        train_df.shape,
        eval_df.shape,
        holdout_df.shape,
        train_df["date"].min(),
        train_df["date"].max(),
        eval_df["date"].min(),
        eval_df["date"].max(),
        holdout_df["date"].min(),
        holdout_df["date"].max(),
    )
    file_logger.info(
        "Cleaned data summary\n"
        "Train shape: {}\n"
        "Eval shape: {}\n"
        "Holdout shape: {}",
        cleaned_splits["train"].shape,
        cleaned_splits["eval"].shape,
        cleaned_splits["holdout"].shape,
    )


if __name__ == "__main__":
    run_data_pipeline()
