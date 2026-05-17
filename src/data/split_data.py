from pathlib import Path

import pandas as pd
import wandb
from loguru import logger

from src.config import paths, split_config, wandb_config
from src.logging_config import log_step


def download_raw_dataset_artifact(
    project_name: str = wandb_config.project_name,
    artifact_name: str = f"{wandb_config.raw_dataset_artifact_name}:latest",
) -> Path:
    """Downloads the raw dataset artifact from W&B and returns the CSV path."""
    logger.info("Downloading raw dataset artifact from W&B: {}", artifact_name)

    with wandb.init(project=project_name, job_type="split_data") as run:
        artifact = run.use_artifact(artifact_name, type="dataset")
        artifact_dir = Path(artifact.download())

    csv_files = list(artifact_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in W&B artifact: {artifact_name}")

    raw_dataset_path = csv_files[0]
    logger.info("Raw dataset artifact downloaded to: {}", raw_dataset_path)

    return raw_dataset_path


def split_dataset_by_date(
    dataset: pd.DataFrame,
    eval_start_date: str = split_config.eval_start_date,
    holdout_start_date: str = split_config.holdout_start_date,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a dataset into train, eval, and holdout sets using date cutoffs."""
    date_column = split_config.date_column
    if date_column not in dataset.columns:
        raise ValueError(f"Dataset must contain a '{date_column}' column.")

    eval_cutoff = pd.Timestamp(eval_start_date)
    holdout_cutoff = pd.Timestamp(holdout_start_date)

    if eval_cutoff >= holdout_cutoff:
        raise ValueError("eval_start_date must be earlier than holdout_start_date.")

    dataset = dataset.copy()
    dataset[date_column] = pd.to_datetime(dataset[date_column])
    dataset = dataset.sort_values(date_column)

    train_df = dataset[dataset[date_column] < eval_cutoff]
    eval_df = dataset[
        (dataset[date_column] >= eval_cutoff)
        & (dataset[date_column] < holdout_cutoff)
    ]
    holdout_df = dataset[dataset[date_column] >= holdout_cutoff]

    return train_df, eval_df, holdout_df


def save_data_splits(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    output_dir: Path | str = paths.processed_data_dir,
) -> None:
    """Saves train, eval, and holdout datasets to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / split_config.train_filename, index=False)
    eval_df.to_csv(output_dir / split_config.eval_filename, index=False)
    holdout_df.to_csv(output_dir / split_config.holdout_filename, index=False)

    logger.info("Data splits saved to: {}", output_dir)


@log_step
def register_data_splits(
    output_dir: Path | str = paths.processed_data_dir,
    project_name: str = wandb_config.project_name,
    artifact_name: str = wandb_config.processed_dataset_artifact_name,
) -> None:
    """Registers train, eval, and holdout files as a W&B dataset artifact."""
    output_dir = Path(output_dir)
    split_files = [
        output_dir / split_config.train_filename,
        output_dir / split_config.eval_filename,
        output_dir / split_config.holdout_filename,
    ]

    missing_files = [file_path for file_path in split_files if not file_path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing split files: {missing_files}")

    logger.info("Registering processed data splits in W&B: {}", output_dir)

    with wandb.init(project=project_name, job_type="split_data") as run:
        artifact = wandb.Artifact(
            name=artifact_name,
            type="dataset",
            description="Train, eval, and holdout splits from the housing dataset.",
            metadata={
                "eval_start_date": split_config.eval_start_date,
                "holdout_start_date": split_config.holdout_start_date,
                "date_column": split_config.date_column,
                "train_filename": split_config.train_filename,
                "eval_filename": split_config.eval_filename,
                "holdout_filename": split_config.holdout_filename,
            },
        )

        for file_path in split_files:
            artifact.add_file(str(file_path), name=file_path.name)

        run.log_artifact(artifact)

    logger.info("Processed data splits registered as W&B artifact: {}", artifact_name)


@log_step
def load_and_split_data(
    raw_path: Path | str | None = None,
    output_dir: Path | str = paths.processed_data_dir,
    eval_start_date: str = split_config.eval_start_date,
    holdout_start_date: str = split_config.holdout_start_date,
    register_artifact: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads the raw dataset artifact, splits it by date, and saves the outputs."""
    raw_dataset_path = (
        Path(raw_path) if raw_path is not None else download_raw_dataset_artifact()
    )

    logger.info("Loading raw dataset for split from: {}", raw_dataset_path)
    dataset = pd.read_csv(raw_dataset_path)

    train_df, eval_df, holdout_df = split_dataset_by_date(
        dataset=dataset,
        eval_start_date=eval_start_date,
        holdout_start_date=holdout_start_date,
    )

    save_data_splits(
        train_df=train_df,
        eval_df=eval_df,
        holdout_df=holdout_df,
        output_dir=output_dir,
    )

    if register_artifact:
        register_data_splits(output_dir=output_dir)

    logger.info(
        "Data split completed. Train: {}, Eval: {}, Holdout: {}",
        train_df.shape,
        eval_df.shape,
        holdout_df.shape,
    )

    return train_df, eval_df, holdout_df
