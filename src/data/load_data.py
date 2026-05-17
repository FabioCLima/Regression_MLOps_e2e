from pathlib import Path

import pandas as pd
import wandb
from loguru import logger

from src.config import paths, wandb_config
from src.logging_config import log_step

RAW_DATASET_PATH = paths.raw_data_dir / "HouseTS.csv"


@log_step
def load_raw_dataset(data_path: Path = RAW_DATASET_PATH) -> pd.DataFrame:
    """Loads the raw housing dataset from disk."""
    if not data_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {data_path}")

    logger.info("Loading raw dataset from: {}", data_path)
    dataset = pd.read_csv(data_path)
    logger.info("Raw dataset loaded with shape: {}", dataset.shape)

    return dataset


@log_step
def register_raw_dataset(
    data_path: Path = RAW_DATASET_PATH,
    project_name: str = wandb_config.project_name,
    artifact_name: str = wandb_config.raw_dataset_artifact_name,
) -> None:
    """Registers the raw dataset file as a W&B dataset artifact."""
    if not data_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {data_path}")

    logger.info("Registering raw dataset in W&B: {}", data_path)

    with wandb.init(
        project=project_name,
        job_type="register_raw_data",
    ) as run:
        artifact = wandb.Artifact(
            name=artifact_name,
            type="dataset",
            description="Raw housing time series dataset.",
            metadata={
                "source_path": str(data_path),
                "project_dir": str(paths.project_dir),
            },
        )

        artifact.add_file(str(data_path))
        run.log_artifact(artifact)

    logger.info("Raw dataset registered as W&B artifact: {}", artifact_name)
