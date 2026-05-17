import re
from pathlib import Path
from typing import Any

import pandas as pd
import wandb
from loguru import logger

from src.config import paths, preprocessing_config, wandb_config
from src.logging_config import log_step


def cleaned_split_filename(split: str) -> str:
    """Returns the filename used for a cleaned dataset split."""
    return f"{preprocessing_config.cleaned_filename_prefix}_{split}.csv"


def normalize_city(value: Any) -> Any:
    """Normalizes city names while preserving missing values."""
    if pd.isna(value):
        return value

    normalized_value = str(value).strip().lower()
    normalized_value = re.sub(r"[–—-]", "-", normalized_value)
    normalized_value = re.sub(r"\s+", " ", normalized_value)

    return normalized_value


def normalize_city_mapping(city_mapping: dict[str, str]) -> dict[str, str]:
    """Normalizes city mapping keys and values."""
    return {
        normalize_city(source_city): normalize_city(target_city)
        for source_city, target_city in city_mapping.items()
    }


def normalize_metro_name(value: Any) -> Any:
    """Normalizes metro names and removes state suffixes from metro_full."""
    normalized_value = normalize_city(value)
    if pd.isna(normalized_value):
        return normalized_value

    return str(normalized_value).split(",", maxsplit=1)[0]


def clean_and_merge_metros(
    dataset: pd.DataFrame,
    metros_path: Path | str | None = paths.raw_data_dir
    / preprocessing_config.metros_filename,
) -> pd.DataFrame:
    """Normalizes city names and merges latitude/longitude from metros data."""
    dataset = dataset.copy()
    city_column = preprocessing_config.city_column
    metro_column = preprocessing_config.metro_column
    latitude_column = preprocessing_config.latitude_column
    longitude_column = preprocessing_config.longitude_column
    location_columns = {latitude_column, longitude_column}

    if city_column not in dataset.columns:
        logger.warning("Skipping metros merge: missing column '{}'.", city_column)
        return dataset

    dataset[city_column] = dataset[city_column].apply(normalize_city)
    dataset[city_column] = dataset[city_column].replace(
        normalize_city_mapping(preprocessing_config.city_mapping)
    )

    if location_columns.issubset(dataset.columns):
        logger.info("Skipping metros merge: lat/lng columns already exist.")
        return dataset

    if metros_path is None:
        logger.warning("Skipping metros merge: metros_path was not provided.")
        return dataset

    metros_path = Path(metros_path)
    if not metros_path.exists():
        logger.warning("Skipping metros merge: metros file not found: {}", metros_path)
        return dataset

    metros = pd.read_csv(metros_path)
    required_metros_columns = {metro_column, latitude_column, longitude_column}
    if not required_metros_columns.issubset(metros.columns):
        logger.warning(
            "Skipping metros merge: metros file missing required columns: {}",
            sorted(required_metros_columns - set(metros.columns)),
        )
        return dataset

    metros = metros.copy()
    metros[metro_column] = metros[metro_column].apply(normalize_metro_name)
    metros = metros.drop_duplicates(subset=[metro_column], keep="first")

    dataset = dataset.merge(
        metros[[metro_column, latitude_column, longitude_column]],
        how="left",
        left_on=city_column,
        right_on=metro_column,
    )
    dataset = dataset.drop(columns=[metro_column], errors="ignore")

    missing_locations = dataset[dataset[latitude_column].isna()][city_column].unique()
    if len(missing_locations) > 0:
        logger.warning("Missing lat/lng for cities: {}", sorted(missing_locations))
    else:
        logger.info("All cities matched with metros dataset.")

    return dataset


def drop_duplicate_records(dataset: pd.DataFrame) -> pd.DataFrame:
    """Drops duplicate records while ignoring configured time columns."""
    duplicate_subset = dataset.columns.difference(
        preprocessing_config.duplicate_ignore_columns
    )

    if len(duplicate_subset) == 0:
        logger.warning("Skipping duplicate removal: no columns available for comparison.")
        return dataset.copy()

    before_rows = dataset.shape[0]
    dataset = dataset.drop_duplicates(subset=duplicate_subset, keep="first").copy()
    removed_rows = before_rows - dataset.shape[0]
    logger.info("Removed {} duplicate rows.", removed_rows)

    return dataset


def remove_price_outliers(dataset: pd.DataFrame) -> pd.DataFrame:
    """Removes rows with extreme list prices."""
    outlier_column = preprocessing_config.outlier_column

    if outlier_column not in dataset.columns:
        logger.warning("Skipping outlier removal: missing column '{}'.", outlier_column)
        return dataset.copy()

    before_rows = dataset.shape[0]
    outlier_values = pd.to_numeric(dataset[outlier_column], errors="coerce")
    keep_rows = (
        outlier_values.isna()
        | (outlier_values <= preprocessing_config.max_median_list_price)
    )
    dataset = dataset[keep_rows].copy()
    removed_rows = before_rows - dataset.shape[0]
    logger.info("Removed {} price outlier rows.", removed_rows)

    return dataset


def preprocess_dataset(
    dataset: pd.DataFrame,
    metros_path: Path | str | None = paths.raw_data_dir
    / preprocessing_config.metros_filename,
) -> pd.DataFrame:
    """Applies the preprocessing transformations to a dataset."""
    dataset = clean_and_merge_metros(dataset=dataset, metros_path=metros_path)
    dataset = drop_duplicate_records(dataset)
    dataset = remove_price_outliers(dataset)

    return dataset


def download_processed_dataset_artifact(
    project_name: str = wandb_config.project_name,
    artifact_name: str = f"{wandb_config.processed_dataset_artifact_name}:latest",
) -> Path:
    """Downloads the processed split artifact from W&B and returns its directory."""
    logger.info("Downloading processed dataset artifact from W&B: {}", artifact_name)

    with wandb.init(project=project_name, job_type="preprocess_data") as run:
        artifact = run.use_artifact(artifact_name, type="dataset")
        artifact_dir = Path(artifact.download())

    logger.info("Processed dataset artifact downloaded to: {}", artifact_dir)

    return artifact_dir


def preprocess_split(
    split: str,
    input_dir: Path | str,
    output_dir: Path | str = paths.processed_data_dir,
    metros_path: Path | str | None = paths.raw_data_dir
    / preprocessing_config.metros_filename,
) -> pd.DataFrame:
    """Preprocesses one split and saves it to output_dir."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    input_path = input_dir / f"{split}.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Split file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_csv(input_path)
    cleaned_dataset = preprocess_dataset(dataset=dataset, metros_path=metros_path)

    output_path = output_dir / cleaned_split_filename(split)
    cleaned_dataset.to_csv(output_path, index=False)
    logger.info("Preprocessed split '{}' saved to: {}", split, output_path)

    return cleaned_dataset


@log_step
def register_cleaned_splits(
    output_dir: Path | str = paths.processed_data_dir,
    project_name: str = wandb_config.project_name,
    artifact_name: str = wandb_config.cleaned_dataset_artifact_name,
) -> None:
    """Registers cleaned train, eval, and holdout files as a W&B artifact."""
    output_dir = Path(output_dir)
    cleaned_files = [
        output_dir / cleaned_split_filename(split)
        for split in preprocessing_config.splits
    ]

    missing_files = [file_path for file_path in cleaned_files if not file_path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing cleaned split files: {missing_files}")

    logger.info("Registering cleaned data splits in W&B: {}", output_dir)

    with wandb.init(project=project_name, job_type="preprocess_data") as run:
        artifact = wandb.Artifact(
            name=artifact_name,
            type="dataset",
            description="Cleaned train, eval, and holdout housing dataset splits.",
            metadata={
                "splits": preprocessing_config.splits,
                "city_column": preprocessing_config.city_column,
                "outlier_column": preprocessing_config.outlier_column,
                "max_median_list_price": (
                    preprocessing_config.max_median_list_price
                ),
            },
        )

        for file_path in cleaned_files:
            artifact.add_file(str(file_path), name=file_path.name)

        run.log_artifact(artifact)

    logger.info("Cleaned data splits registered as W&B artifact: {}", artifact_name)


@log_step
def run_preprocessing(
    input_dir: Path | str | None = None,
    output_dir: Path | str = paths.processed_data_dir,
    splits: tuple[str, ...] = preprocessing_config.splits,
    metros_path: Path | str | None = paths.raw_data_dir
    / preprocessing_config.metros_filename,
    register_artifact: bool = True,
) -> dict[str, pd.DataFrame]:
    """Preprocesses all configured splits and optionally registers them in W&B."""
    input_dir = (
        Path(input_dir)
        if input_dir is not None
        else download_processed_dataset_artifact()
    )

    cleaned_splits = {
        split: preprocess_split(
            split=split,
            input_dir=input_dir,
            output_dir=output_dir,
            metros_path=metros_path,
        )
        for split in splits
    }

    if register_artifact:
        register_cleaned_splits(output_dir=output_dir)

    return cleaned_splits
