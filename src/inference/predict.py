import json
from pathlib import Path

import pandas as pd
from joblib import load
from loguru import logger

from src.config import (
    feature_config,
    inference_config,
    model_config,
    paths,
    preprocessing_config,
    tuning_config,
)
from src.data.preprocess_data import preprocess_dataset
from src.features.feature_engineering import (
    add_date_features,
    apply_frequency_encoder,
    apply_target_encoder,
    drop_unused_columns,
    feature_engineered_split_filename,
)
from src.models.train_model import evaluate_regression


def default_model_path() -> Path:
    """Returns the preferred model path for inference."""
    tuned_model_path = paths.models_dir / tuning_config.tuned_model_filename
    if tuned_model_path.exists():
        return tuned_model_path

    return paths.models_dir / model_config.best_model_filename


def load_training_feature_columns(
    train_features_path: Path | str = paths.processed_data_dir
    / feature_engineered_split_filename("train"),
    target_column: str = model_config.target_column,
) -> list[str]:
    """Loads the model feature schema from the feature-engineered train file."""
    train_features_path = Path(train_features_path)
    if not train_features_path.exists():
        raise FileNotFoundError(f"Training feature file not found: {train_features_path}")

    train_columns = pd.read_csv(train_features_path, nrows=1).columns.to_list()

    return [column for column in train_columns if column != target_column]


def load_feature_encoders(
    encoders_dir: Path | str = paths.models_dir,
) -> tuple[pd.Series | None, object | None]:
    """Loads fitted feature encoders when available."""
    encoders_dir = Path(encoders_dir)
    frequency_encoder_path = encoders_dir / feature_config.frequency_encoder_filename
    target_encoder_path = encoders_dir / feature_config.target_encoder_filename
    frequency_map = None
    target_encoder = None

    if frequency_encoder_path.exists():
        frequency_map = load(frequency_encoder_path)
    else:
        logger.warning("Frequency encoder not found: {}", frequency_encoder_path)

    if target_encoder_path.exists():
        target_encoder = load(target_encoder_path)
    else:
        logger.warning("Target encoder not found: {}", target_encoder_path)

    return frequency_map, target_encoder


def apply_inference_feature_engineering(
    dataset: pd.DataFrame,
    frequency_map: pd.Series | None,
    target_encoder: object | None,
) -> pd.DataFrame:
    """Applies inference-time feature engineering using fitted encoders."""
    dataset = add_date_features(dataset)

    if frequency_map is not None:
        dataset = apply_frequency_encoder(
            dataset=dataset,
            column=feature_config.frequency_encode_column,
            frequency_map=frequency_map,
        )

    if target_encoder is not None:
        dataset = apply_target_encoder(
            dataset=dataset,
            column=feature_config.target_encode_column,
            encoder=target_encoder,
        )

    return drop_unused_columns(dataset)


def align_features_to_training_schema(
    dataset: pd.DataFrame,
    training_feature_columns: list[str],
) -> pd.DataFrame:
    """Aligns inference features to the training feature schema."""
    return dataset.reindex(columns=training_feature_columns, fill_value=0)


def build_predictions_output(
    features: pd.DataFrame,
    predictions: list[float],
    actuals: pd.Series | None = None,
) -> pd.DataFrame:
    """Builds the prediction output DataFrame."""
    output = features.copy()
    output[inference_config.prediction_column] = predictions

    if actuals is not None:
        output[inference_config.actual_column] = actuals.to_numpy()
        output[inference_config.prediction_error_column] = (
            output[inference_config.prediction_column]
            - output[inference_config.actual_column]
        )

    return output


def calculate_inference_metrics(
    actuals: pd.Series | None,
    predictions: list[float],
) -> dict[str, float] | None:
    """Calculates inference metrics when actual target values are available."""
    if actuals is None:
        return None

    return evaluate_regression(actuals, pd.Series(predictions))


def predict(
    input_df: pd.DataFrame,
    model_path: Path | str | None = None,
    encoders_dir: Path | str = paths.models_dir,
    train_features_path: Path | str = paths.processed_data_dir
    / feature_engineered_split_filename("train"),
    metros_path: Path | str | None = paths.raw_data_dir
    / preprocessing_config.metros_filename,
    *,
    model=None,
    encoders: tuple | None = None,
    train_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, float] | None]:
    """Runs inference from raw input data and returns predictions plus metrics."""
    if model is None:
        model_path = Path(model_path) if model_path is not None else default_model_path()
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = load(model_path)

    if encoders is None:
        frequency_map, target_encoder = load_feature_encoders(encoders_dir)
    else:
        frequency_map, target_encoder = encoders

    training_feature_columns = (
        load_training_feature_columns(train_features_path)
        if train_columns is None
        else train_columns
    )

    dataset = preprocess_dataset(input_df, metros_path=metros_path)
    dataset = apply_inference_feature_engineering(
        dataset=dataset,
        frequency_map=frequency_map,
        target_encoder=target_encoder,
    )

    actuals = None
    if model_config.target_column in dataset.columns:
        actuals = dataset[model_config.target_column].copy()
        dataset = dataset.drop(columns=[model_config.target_column])

    features = align_features_to_training_schema(dataset, training_feature_columns)
    predictions = list(model.predict(features))
    output = build_predictions_output(features, predictions, actuals)
    metrics = calculate_inference_metrics(actuals, predictions)

    return output, metrics


def save_inference_outputs(
    predictions: pd.DataFrame,
    metrics: dict[str, float] | None,
    output_dir: Path | str = paths.processed_data_dir,
) -> None:
    """Saves predictions and optional metrics to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / inference_config.predictions_filename, index=False)

    if metrics is not None:
        with (output_dir / inference_config.inference_metrics_filename).open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(metrics, f, indent=2)

    logger.info("Inference outputs saved to: {}", output_dir)


def run_inference(
    input_path: Path | str,
    output_dir: Path | str = paths.processed_data_dir,
    model_path: Path | str | None = None,
    encoders_dir: Path | str = paths.models_dir,
    train_features_path: Path | str = paths.processed_data_dir
    / feature_engineered_split_filename("train"),
    metros_path: Path | str | None = paths.raw_data_dir
    / preprocessing_config.metros_filename,
) -> tuple[pd.DataFrame, dict[str, float] | None]:
    """Loads raw input data, runs inference, and saves outputs."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Inference input file not found: {input_path}")

    input_df = pd.read_csv(input_path)
    predictions, metrics = predict(
        input_df=input_df,
        model_path=model_path,
        encoders_dir=encoders_dir,
        train_features_path=train_features_path,
        metros_path=metros_path,
    )
    save_inference_outputs(
        predictions=predictions,
        metrics=metrics,
        output_dir=output_dir,
    )

    return predictions, metrics
