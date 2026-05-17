import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb
from joblib import dump
from loguru import logger
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.config import model_config, paths, wandb_config
from src.features.feature_engineering import feature_engineered_split_filename
from src.logging_config import log_step

Model = Pipeline


def model_filename(model_name: str) -> str:
    """Returns the filename used to persist a trained candidate model."""
    return model_config.model_filename_template.format(model_name=model_name)


def download_feature_dataset_artifact(
    project_name: str = wandb_config.project_name,
    artifact_name: str = f"{wandb_config.feature_engineered_dataset_artifact_name}:latest",
) -> Path:
    """Downloads feature-engineered splits from W&B and returns their directory."""
    logger.info("Downloading feature dataset artifact from W&B: {}", artifact_name)

    with wandb.init(project=project_name, job_type="train_model") as run:
        artifact = run.use_artifact(artifact_name, type="dataset")
        artifact_dir = Path(artifact.download())

    logger.info("Feature dataset artifact downloaded to: {}", artifact_dir)

    return artifact_dir


def load_feature_splits(input_dir: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads feature-engineered train and eval splits from disk."""
    input_dir = Path(input_dir)
    train_path = input_dir / feature_engineered_split_filename("train")
    eval_path = input_dir / feature_engineered_split_filename("eval")
    missing_files = [
        file_path for file_path in (train_path, eval_path) if not file_path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(f"Missing feature split files: {missing_files}")

    return pd.read_csv(train_path), pd.read_csv(eval_path)


def maybe_sample_dataset(
    dataset: pd.DataFrame,
    sample_frac: float | None = model_config.sample_frac,
    random_state: int = model_config.random_state,
) -> pd.DataFrame:
    """Optionally samples a dataset for faster training experiments."""
    if sample_frac is None:
        return dataset.copy()

    sample_frac = float(sample_frac)
    if sample_frac <= 0 or sample_frac >= 1:
        logger.warning("Ignoring invalid sample_frac: {}", sample_frac)
        return dataset.copy()

    return dataset.sample(frac=sample_frac, random_state=random_state).reset_index(
        drop=True
    )


def split_features_target(
    dataset: pd.DataFrame,
    target_column: str = model_config.target_column,
) -> tuple[pd.DataFrame, pd.Series]:
    """Splits a dataset into numeric features and target."""
    if target_column not in dataset.columns:
        raise ValueError(f"Dataset must contain target column: {target_column}")

    features = dataset.drop(columns=[target_column]).copy()
    target = dataset[target_column].copy()

    non_numeric_columns = features.select_dtypes(exclude="number").columns.to_list()
    if non_numeric_columns:
        raise ValueError(f"Features must be numeric. Non-numeric: {non_numeric_columns}")

    return features, target


def build_model(model_name: str, random_state: int = model_config.random_state) -> Model:
    """Builds one candidate model by name."""
    if model_name == "dummy":
        estimator = DummyRegressor(strategy="mean")
        return Pipeline([("imputer", SimpleImputer()), ("model", estimator)])

    if model_name == "linear_regression":
        estimator = LinearRegression()
        return Pipeline(
            [
                ("imputer", SimpleImputer()),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )

    if model_name == "ridge":
        estimator = Ridge(alpha=1.0, random_state=random_state)
        return Pipeline(
            [
                ("imputer", SimpleImputer()),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )

    if model_name == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=model_config.random_forest_n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        return Pipeline([("imputer", SimpleImputer()), ("model", estimator)])

    if model_name == "xgboost":
        estimator = XGBRegressor(
            n_estimators=model_config.xgboost_n_estimators,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            objective="reg:squarederror",
        )
        return Pipeline([("imputer", SimpleImputer()), ("model", estimator)])

    raise ValueError(f"Unsupported model_name: {model_name}")


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calculates regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_single_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    random_state: int = model_config.random_state,
) -> tuple[Model, dict[str, dict[str, float]]]:
    """Trains and evaluates one candidate model."""
    model = build_model(model_name=model_name, random_state=random_state)
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    eval_predictions = model.predict(X_eval)
    train_metrics = evaluate_regression(y_train, train_predictions)
    eval_metrics = evaluate_regression(y_eval, eval_predictions)
    metrics = build_train_eval_metrics(train_metrics, eval_metrics)
    logger.info("Model '{}' metrics: {}", model_name, metrics)

    return model, metrics


def build_train_eval_metrics(
    train_metrics: dict[str, float],
    eval_metrics: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Combines train/eval metrics and calculates generalization gaps."""
    common_metrics = train_metrics.keys() & eval_metrics.keys()
    gap_metrics = {
        metric: float(eval_metrics[metric] - train_metrics[metric])
        for metric in common_metrics
    }

    return {
        "train": train_metrics,
        "eval": eval_metrics,
        "gap": gap_metrics,
    }


def train_candidate_models(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    candidate_models: tuple[str, ...] = model_config.candidate_models,
    sample_frac: float | None = model_config.sample_frac,
    random_state: int = model_config.random_state,
) -> tuple[dict[str, Model], dict[str, dict[str, dict[str, float]]]]:
    """Trains and evaluates all configured candidate models."""
    train_df = maybe_sample_dataset(train_df, sample_frac, random_state)
    eval_df = maybe_sample_dataset(eval_df, sample_frac, random_state)
    X_train, y_train = split_features_target(train_df)
    X_eval, y_eval = split_features_target(eval_df)

    trained_models: dict[str, Model] = {}
    metrics_by_model: dict[str, dict[str, dict[str, float]]] = {}

    for model_name in candidate_models:
        trained_models[model_name], metrics_by_model[model_name] = train_single_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            y_eval=y_eval,
            random_state=random_state,
        )

    return trained_models, metrics_by_model


def select_best_model(
    metrics_by_model: dict[str, dict[str, dict[str, float]]],
    primary_metric: str = model_config.primary_metric,
) -> str:
    """Selects the best model using the configured primary metric."""
    if not metrics_by_model:
        raise ValueError("metrics_by_model cannot be empty.")

    missing_metric_models = [
        model_name
        for model_name, metrics in metrics_by_model.items()
        if primary_metric not in metrics.get("eval", {})
    ]
    if missing_metric_models:
        raise ValueError(
            f"Missing metric '{primary_metric}' for models: {missing_metric_models}"
        )

    return min(
        metrics_by_model,
        key=lambda model_name: metrics_by_model[model_name]["eval"][primary_metric],
    )


def save_training_outputs(
    trained_models: dict[str, Model],
    metrics_by_model: dict[str, dict[str, dict[str, float]]],
    best_model_name: str,
    output_dir: Path | str = paths.models_dir,
    save_candidate_models: bool = False,
) -> None:
    """Saves the best model and metrics to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_candidate_models:
        for name, model in trained_models.items():
            dump(model, output_dir / model_filename(name))

    dump(trained_models[best_model_name], output_dir / model_config.best_model_filename)

    metrics_payload: dict[str, Any] = {
        "best_model": best_model_name,
        "primary_metric": model_config.primary_metric,
        "metrics": metrics_by_model,
    }
    with (output_dir / model_config.metrics_filename).open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    logger.info("Training outputs saved to: {}", output_dir)


def build_best_model_description(metrics_payload: dict[str, Any]) -> str:
    """Builds a human-readable W&B artifact description for the best model."""
    best_model = metrics_payload["best_model"]
    primary_metric = metrics_payload["primary_metric"]
    primary_metric_value = metrics_payload["metrics"][best_model]["eval"][
        primary_metric
    ]

    return (
        "Best baseline regression model for the housing price prediction pipeline. "
        f"Selected model: {best_model}. "
        f"Selection metric: {primary_metric}={primary_metric_value:.4f}. "
        "This artifact contains the serialized best model and the full metrics "
        "comparison across candidate models."
    )


def build_best_model_tags(metrics_payload: dict[str, Any]) -> list[str]:
    """Builds tags that identify the purpose of the best model artifact."""
    return [
        "model",
        "best-model",
        "regression",
        "housing-price-prediction",
        "baseline-comparison",
        f"selected-{metrics_payload['best_model']}",
        f"metric-{metrics_payload['primary_metric']}",
    ]


@log_step
def register_training_artifacts(
    output_dir: Path | str = paths.models_dir,
    project_name: str = wandb_config.project_name,
) -> None:
    """Registers only the best model and its metrics in W&B."""
    output_dir = Path(output_dir)
    metrics_path = output_dir / model_config.metrics_filename
    best_model_path = output_dir / model_config.best_model_filename

    required_files = [metrics_path, best_model_path]
    missing_files = [file_path for file_path in required_files if not file_path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing training output files: {missing_files}")

    with metrics_path.open(encoding="utf-8") as f:
        metrics_payload = json.load(f)

    with wandb.init(project=project_name, job_type="train_model") as run:
        for model_name, metrics in metrics_payload["metrics"].items():
            for split_name, split_metrics in metrics.items():
                run.log(
                    {
                        f"{model_name}/{split_name}/{metric}": value
                        for metric, value in split_metrics.items()
                    }
                )
        run.summary["best_model"] = metrics_payload["best_model"]
        run.summary["primary_metric"] = metrics_payload["primary_metric"]

        best_model_artifact = wandb.Artifact(
            name=wandb_config.best_model_artifact_name,
            type="model",
            description=build_best_model_description(metrics_payload),
            metadata={
                "best_model": metrics_payload["best_model"],
                "primary_metric": metrics_payload["primary_metric"],
                "primary_metric_value": metrics_payload["metrics"][
                    metrics_payload["best_model"]
                ]["eval"][metrics_payload["primary_metric"]],
                "best_model_train_metrics": metrics_payload["metrics"][
                    metrics_payload["best_model"]
                ]["train"],
                "best_model_eval_metrics": metrics_payload["metrics"][
                    metrics_payload["best_model"]
                ]["eval"],
                "best_model_generalization_gap": metrics_payload["metrics"][
                    metrics_payload["best_model"]
                ]["gap"],
                "candidate_models": list(metrics_payload["metrics"]),
                "metrics_by_model": metrics_payload["metrics"],
                "target_column": model_config.target_column,
                "random_state": model_config.random_state,
                "sample_frac": model_config.sample_frac,
                "feature_dataset_artifact": (
                    f"{wandb_config.feature_engineered_dataset_artifact_name}:latest"
                ),
                "tags": build_best_model_tags(metrics_payload),
            },
        )
        best_model_artifact.add_file(str(best_model_path), name=best_model_path.name)
        best_model_artifact.add_file(str(metrics_path), name=metrics_path.name)
        run.log_artifact(
            best_model_artifact,
            aliases=[
                "latest",
                "best",
                metrics_payload["best_model"],
                f"best-{metrics_payload['primary_metric']}",
            ],
        )

    logger.info("Training artifacts registered in W&B.")


@log_step
def run_model_training(
    input_dir: Path | str | None = None,
    output_dir: Path | str = paths.models_dir,
    candidate_models: tuple[str, ...] = model_config.candidate_models,
    sample_frac: float | None = model_config.sample_frac,
    register_artifact: bool = True,
    save_candidate_models: bool = False,
) -> tuple[dict[str, Model], dict[str, dict[str, dict[str, float]]], str]:
    """Runs model comparison and optionally registers outputs in W&B."""
    input_dir = (
        Path(input_dir)
        if input_dir is not None
        else download_feature_dataset_artifact()
    )
    train_df, eval_df = load_feature_splits(input_dir=input_dir)
    trained_models, metrics_by_model = train_candidate_models(
        train_df=train_df,
        eval_df=eval_df,
        candidate_models=candidate_models,
        sample_frac=sample_frac,
    )
    best_model_name = select_best_model(metrics_by_model)

    save_training_outputs(
        trained_models=trained_models,
        metrics_by_model=metrics_by_model,
        best_model_name=best_model_name,
        output_dir=output_dir,
        save_candidate_models=save_candidate_models,
    )

    if register_artifact:
        register_training_artifacts(output_dir=output_dir)

    logger.info("Best model selected: {}", best_model_name)

    return trained_models, metrics_by_model, best_model_name
