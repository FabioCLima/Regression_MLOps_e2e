from pathlib import Path

import pandas as pd
from category_encoders import TargetEncoder
from joblib import dump
from loguru import logger

from src.config import feature_config, paths, preprocessing_config, wandb_config
from src.data.preprocess_data import cleaned_split_filename
from src.logging_config import log_step


def feature_engineered_split_filename(split: str) -> str:
    """Returns the filename used for a feature-engineered split."""
    return f"{feature_config.output_filename_prefix}_{split}.csv"


def add_date_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Adds year, quarter, and month features from the configured date column."""
    date_column = feature_config.date_column
    if date_column not in dataset.columns:
        raise ValueError(f"Dataset must contain a '{date_column}' column.")

    dataset = dataset.copy()
    dataset[date_column] = pd.to_datetime(dataset[date_column])
    dataset["year"] = dataset[date_column].dt.year
    dataset["quarter"] = dataset[date_column].dt.quarter
    dataset["month"] = dataset[date_column].dt.month

    insert_position = dataset.columns.get_loc(date_column) + 1
    for column in ("month", "quarter", "year"):
        dataset.insert(insert_position, column, dataset.pop(column))

    return dataset


def fit_frequency_encoder(train_df: pd.DataFrame, column: str) -> pd.Series:
    """Fits a frequency encoding map using only the train split."""
    if column not in train_df.columns:
        raise ValueError(f"Train dataset must contain a '{column}' column.")

    return train_df[column].value_counts(dropna=False)


def apply_frequency_encoder(
    dataset: pd.DataFrame,
    column: str,
    frequency_map: pd.Series,
) -> pd.DataFrame:
    """Applies a fitted frequency encoding map to a dataset."""
    if column not in dataset.columns:
        logger.warning("Skipping frequency encoding: missing column '{}'.", column)
        return dataset.copy()

    dataset = dataset.copy()
    encoded_column = f"{column}_freq"
    dataset[encoded_column] = dataset[column].map(frequency_map).fillna(0).astype(int)

    return dataset


def fit_target_encoder(
    train_df: pd.DataFrame,
    column: str,
    target_column: str,
) -> TargetEncoder:
    """Fits a target encoder using only the train split."""
    missing_columns = {column, target_column} - set(train_df.columns)
    if missing_columns:
        raise ValueError(f"Train dataset missing required columns: {missing_columns}")

    encoder = TargetEncoder(cols=[column])
    encoder.fit(train_df[[column]], train_df[target_column])

    return encoder


def apply_target_encoder(
    dataset: pd.DataFrame,
    column: str,
    encoder: TargetEncoder,
) -> pd.DataFrame:
    """Applies a fitted target encoder to a dataset."""
    if column not in dataset.columns:
        logger.warning("Skipping target encoding: missing column '{}'.", column)
        return dataset.copy()

    dataset = dataset.copy()
    encoded_column = f"{column}_encoded"
    dataset[encoded_column] = encoder.transform(dataset[[column]])[column]

    return dataset


def drop_unused_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    """Drops leakage-prone and raw categorical columns after encoding."""
    return dataset.drop(columns=list(feature_config.drop_columns), errors="ignore")


def engineer_features(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series | None, TargetEncoder | None]:
    """Fits feature transformations on train and applies them to all splits."""
    train_df = add_date_features(train_df)
    eval_df = add_date_features(eval_df)
    holdout_df = add_date_features(holdout_df)

    frequency_map = None
    frequency_column = feature_config.frequency_encode_column
    if frequency_column in train_df.columns:
        frequency_map = fit_frequency_encoder(train_df, frequency_column)
        train_df = apply_frequency_encoder(train_df, frequency_column, frequency_map)
        eval_df = apply_frequency_encoder(eval_df, frequency_column, frequency_map)
        holdout_df = apply_frequency_encoder(holdout_df, frequency_column, frequency_map)
    else:
        logger.warning(
            "Skipping frequency encoding: train split missing column '{}'.",
            frequency_column,
        )

    target_encoder = None
    target_encode_column = feature_config.target_encode_column
    if target_encode_column in train_df.columns:
        target_encoder = fit_target_encoder(
            train_df=train_df,
            column=target_encode_column,
            target_column=feature_config.target_column,
        )
        train_df = apply_target_encoder(train_df, target_encode_column, target_encoder)
        eval_df = apply_target_encoder(eval_df, target_encode_column, target_encoder)
        holdout_df = apply_target_encoder(
            holdout_df,
            target_encode_column,
            target_encoder,
        )
    else:
        logger.warning(
            "Skipping target encoding: train split missing column '{}'.",
            target_encode_column,
        )

    train_df = drop_unused_columns(train_df)
    eval_df = drop_unused_columns(eval_df)
    holdout_df = drop_unused_columns(holdout_df)

    return train_df, eval_df, holdout_df, frequency_map, target_encoder


def download_cleaned_dataset_artifact(
    project_name: str = wandb_config.project_name,
    artifact_name: str = f"{wandb_config.cleaned_dataset_artifact_name}:latest",
) -> Path:
    """Downloads the cleaned dataset artifact from W&B and returns its directory."""
    import wandb  # training-only dependency
    logger.info("Downloading cleaned dataset artifact from W&B: {}", artifact_name)

    with wandb.init(project=project_name, job_type="feature_engineering") as run:
        artifact = run.use_artifact(artifact_name, type="dataset")
        artifact_dir = Path(artifact.download())

    logger.info("Cleaned dataset artifact downloaded to: {}", artifact_dir)

    return artifact_dir


def load_cleaned_splits(input_dir: Path | str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads cleaned train, eval, and holdout splits from disk."""
    input_dir = Path(input_dir)
    split_paths = [
        input_dir / cleaned_split_filename(split)
        for split in preprocessing_config.splits
    ]

    missing_files = [split_path for split_path in split_paths if not split_path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing cleaned split files: {missing_files}")

    return tuple(pd.read_csv(split_path) for split_path in split_paths)


def save_feature_engineered_splits(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    output_dir: Path | str = paths.processed_data_dir,
) -> None:
    """Saves feature-engineered train, eval, and holdout splits."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, dataset in zip(
        feature_config.splits,
        (train_df, eval_df, holdout_df),
        strict=True,
    ):
        dataset.to_csv(output_dir / feature_engineered_split_filename(split), index=False)

    logger.info("Feature-engineered splits saved to: {}", output_dir)


def save_feature_encoders(
    frequency_map: pd.Series | None,
    target_encoder: TargetEncoder | None,
    output_dir: Path | str = paths.models_dir,
) -> None:
    """Saves fitted feature encoders for inference reuse."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if frequency_map is not None:
        dump(frequency_map, output_dir / feature_config.frequency_encoder_filename)

    if target_encoder is not None:
        dump(target_encoder, output_dir / feature_config.target_encoder_filename)

    logger.info("Feature encoders saved to: {}", output_dir)


@log_step
def register_feature_engineering_artifacts(
    data_dir: Path | str = paths.processed_data_dir,
    models_dir: Path | str = paths.models_dir,
    project_name: str = wandb_config.project_name,
) -> None:
    """Registers feature-engineered splits and encoders in W&B."""
    data_dir = Path(data_dir)
    models_dir = Path(models_dir)
    engineered_files = [
        data_dir / feature_engineered_split_filename(split)
        for split in feature_config.splits
    ]
    encoder_files = [
        models_dir / feature_config.frequency_encoder_filename,
        models_dir / feature_config.target_encoder_filename,
    ]

    missing_engineered_files = [
        file_path for file_path in engineered_files if not file_path.exists()
    ]
    if missing_engineered_files:
        raise FileNotFoundError(
            f"Missing feature-engineered split files: {missing_engineered_files}"
        )

    import wandb  # training-only dependency
    with wandb.init(project=project_name, job_type="feature_engineering") as run:
        dataset_artifact = wandb.Artifact(
            name=wandb_config.feature_engineered_dataset_artifact_name,
            type="dataset",
            description="Feature-engineered train, eval, and holdout splits.",
            metadata={
                "target_column": feature_config.target_column,
                "frequency_encode_column": feature_config.frequency_encode_column,
                "target_encode_column": feature_config.target_encode_column,
                "drop_columns": feature_config.drop_columns,
            },
        )
        for file_path in engineered_files:
            dataset_artifact.add_file(str(file_path), name=file_path.name)
        run.log_artifact(dataset_artifact)

        existing_encoder_files = [
            file_path for file_path in encoder_files if file_path.exists()
        ]
        if existing_encoder_files:
            encoders_artifact = wandb.Artifact(
                name=wandb_config.feature_encoders_artifact_name,
                type="model",
                description="Fitted feature engineering encoders.",
            )
            for file_path in existing_encoder_files:
                encoders_artifact.add_file(str(file_path), name=file_path.name)
            run.log_artifact(encoders_artifact)

    logger.info("Feature engineering artifacts registered in W&B.")


@log_step
def run_feature_engineering(
    input_dir: Path | str | None = None,
    output_dir: Path | str = paths.processed_data_dir,
    encoders_dir: Path | str = paths.models_dir,
    register_artifact: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series | None, TargetEncoder | None]:
    """Runs feature engineering and optionally registers outputs in W&B."""
    input_dir = (
        Path(input_dir)
        if input_dir is not None
        else download_cleaned_dataset_artifact()
    )

    train_df, eval_df, holdout_df = load_cleaned_splits(input_dir=input_dir)
    train_df, eval_df, holdout_df, frequency_map, target_encoder = engineer_features(
        train_df=train_df,
        eval_df=eval_df,
        holdout_df=holdout_df,
    )

    save_feature_engineered_splits(
        train_df=train_df,
        eval_df=eval_df,
        holdout_df=holdout_df,
        output_dir=output_dir,
    )
    save_feature_encoders(
        frequency_map=frequency_map,
        target_encoder=target_encoder,
        output_dir=encoders_dir,
    )

    if register_artifact:
        register_feature_engineering_artifacts(
            data_dir=output_dir,
            models_dir=encoders_dir,
        )

    logger.info(
        "Feature engineering completed. Train: {}, Eval: {}, Holdout: {}",
        train_df.shape,
        eval_df.shape,
        holdout_df.shape,
    )

    return train_df, eval_df, holdout_df, frequency_map, target_encoder
