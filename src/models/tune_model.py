import json
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import wandb
from joblib import dump
from loguru import logger
from xgboost import XGBRegressor

from src.config import model_config, paths, tuning_config, wandb_config
from src.logging_config import log_step
from src.models.train_model import (
    build_train_eval_metrics,
    download_feature_dataset_artifact,
    evaluate_regression,
    load_feature_splits,
    maybe_sample_dataset,
    split_features_target,
)


def load_best_model_name(
    metrics_path: Path | str = paths.models_dir / model_config.metrics_filename,
) -> str:
    """Loads the best model name selected by the baseline training pipeline."""
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Training metrics file not found: {metrics_path}")

    with metrics_path.open(encoding="utf-8") as f:
        metrics_payload = json.load(f)

    best_model_name = metrics_payload.get("best_model")
    if not best_model_name:
        raise ValueError("Training metrics file does not contain 'best_model'.")

    return str(best_model_name)


def validate_supported_tuning_model(best_model_name: str) -> None:
    """Ensures tuning only runs for the currently supported best model."""
    if best_model_name != tuning_config.supported_model_name:
        raise ValueError(
            "Tuning currently supports only "
            f"'{tuning_config.supported_model_name}', got '{best_model_name}'."
        )


def suggest_xgboost_params(
    trial: optuna.Trial,
    random_state: int = tuning_config.random_state,
) -> dict[str, Any]:
    """Suggests one XGBoost hyperparameter set for an Optuna trial."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
        "objective": "reg:squarederror",
    }


def build_xgboost_model(params: dict[str, Any]) -> XGBRegressor:
    """Builds an XGBoost regressor from a parameter dictionary."""
    return XGBRegressor(**params)


def calculate_model_metrics(
    model: XGBRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
) -> dict[str, dict[str, float]]:
    """Calculates train, eval, and gap metrics for a fitted model."""
    train_metrics = evaluate_regression(y_train, model.predict(X_train))
    eval_metrics = evaluate_regression(y_eval, model.predict(X_eval))

    return build_train_eval_metrics(train_metrics, eval_metrics)


def save_tuning_outputs(
    model: XGBRegressor,
    best_params: dict[str, Any],
    metrics: dict[str, dict[str, float]],
    trials: list[dict[str, Any]],
    output_dir: Path | str = paths.models_dir,
) -> None:
    """Saves tuned model, metrics, and trial history to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dump(model, output_dir / tuning_config.tuned_model_filename)

    payload = {
        "base_model": tuning_config.supported_model_name,
        "primary_metric": model_config.primary_metric,
        "best_params": best_params,
        "metrics": metrics,
        "n_trials": len(trials),
    }
    with (output_dir / tuning_config.tuning_metrics_filename).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(payload, f, indent=2)

    pd.DataFrame(trials).to_csv(output_dir / tuning_config.trials_filename, index=False)
    logger.info("Tuning outputs saved to: {}", output_dir)


def build_tuned_model_description(metrics_payload: dict[str, Any]) -> str:
    """Builds a human-readable W&B artifact description for the tuned model."""
    primary_metric = metrics_payload["primary_metric"]
    primary_metric_value = metrics_payload["metrics"]["eval"][primary_metric]

    return (
        "Fine-tuned XGBoost regression model for the housing price prediction "
        "pipeline. The model was optimized with Optuna using the eval split and "
        f"selected by {primary_metric}={primary_metric_value:.4f}."
    )


def build_tuned_model_tags(metrics_payload: dict[str, Any]) -> list[str]:
    """Builds tags that identify the purpose of the tuned model artifact."""
    return [
        "model",
        "tuned-model",
        "xgboost",
        "optuna",
        "regression",
        "housing-price-prediction",
        f"metric-{metrics_payload['primary_metric']}",
    ]


@log_step
def register_tuned_model_artifact(
    output_dir: Path | str = paths.models_dir,
    project_name: str = wandb_config.project_name,
) -> None:
    """Registers only the tuned model artifact in W&B."""
    output_dir = Path(output_dir)
    model_path = output_dir / tuning_config.tuned_model_filename
    metrics_path = output_dir / tuning_config.tuning_metrics_filename
    trials_path = output_dir / tuning_config.trials_filename
    required_files = [model_path, metrics_path, trials_path]
    missing_files = [file_path for file_path in required_files if not file_path.exists()]

    if missing_files:
        raise FileNotFoundError(f"Missing tuning output files: {missing_files}")

    with metrics_path.open(encoding="utf-8") as f:
        metrics_payload = json.load(f)

    with wandb.init(project=project_name, job_type="tune_model") as run:
        for split_name, split_metrics in metrics_payload["metrics"].items():
            run.log(
                {
                    f"tuned/{split_name}/{metric}": value
                    for metric, value in split_metrics.items()
                }
            )
        run.summary["base_model"] = metrics_payload["base_model"]
        run.summary["primary_metric"] = metrics_payload["primary_metric"]
        run.summary["n_trials"] = metrics_payload["n_trials"]

        artifact = wandb.Artifact(
            name=wandb_config.tuned_model_artifact_name,
            type="model",
            description=build_tuned_model_description(metrics_payload),
            metadata={
                "base_model": metrics_payload["base_model"],
                "primary_metric": metrics_payload["primary_metric"],
                "primary_metric_value": metrics_payload["metrics"]["eval"][
                    metrics_payload["primary_metric"]
                ],
                "best_params": metrics_payload["best_params"],
                "train_metrics": metrics_payload["metrics"]["train"],
                "eval_metrics": metrics_payload["metrics"]["eval"],
                "generalization_gap": metrics_payload["metrics"]["gap"],
                "n_trials": metrics_payload["n_trials"],
                "feature_dataset_artifact": (
                    f"{wandb_config.feature_engineered_dataset_artifact_name}:latest"
                ),
                "baseline_best_model_artifact": (
                    f"{wandb_config.best_model_artifact_name}:latest"
                ),
                "tags": build_tuned_model_tags(metrics_payload),
            },
        )
        artifact.add_file(str(model_path), name=model_path.name)
        artifact.add_file(str(metrics_path), name=metrics_path.name)
        artifact.add_file(str(trials_path), name=trials_path.name)
        run.log_artifact(
            artifact,
            aliases=[
                "latest",
                "tuned",
                "xgboost",
                f"tuned-{metrics_payload['primary_metric']}",
            ],
        )

    logger.info("Tuned model artifact registered in W&B.")


@log_step
def tune_xgboost_model(
    input_dir: Path | str | None = None,
    baseline_metrics_path: Path | str = paths.models_dir / "metrics.json",
    output_dir: Path | str = paths.models_dir,
    n_trials: int = tuning_config.n_trials,
    sample_frac: float | None = tuning_config.sample_frac,
    random_state: int = tuning_config.random_state,
    register_artifact: bool = True,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """Tunes the previously selected best model when it is XGBoost."""
    best_model_name = load_best_model_name(baseline_metrics_path)
    validate_supported_tuning_model(best_model_name)

    input_dir = (
        Path(input_dir)
        if input_dir is not None
        else download_feature_dataset_artifact()
    )
    train_df, eval_df = load_feature_splits(input_dir)
    train_df = maybe_sample_dataset(train_df, sample_frac, random_state)
    eval_df = maybe_sample_dataset(eval_df, sample_frac, random_state)
    X_train, y_train = split_features_target(train_df)
    X_eval, y_eval = split_features_target(eval_df)
    trial_records: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgboost_params(trial, random_state=random_state)
        model = build_xgboost_model(params)
        model.fit(X_train, y_train)
        metrics = calculate_model_metrics(model, X_train, y_train, X_eval, y_eval)

        trial_record = {
            "trial_number": trial.number,
            "rmse": metrics["eval"]["rmse"],
            "mae": metrics["eval"]["mae"],
            "r2": metrics["eval"]["r2"],
            **params,
        }
        trial_records.append(trial_record)

        return metrics["eval"][model_config.primary_metric]

    study = optuna.create_study(direction=tuning_config.direction)
    study.optimize(objective, n_trials=n_trials)

    best_params = {
        **study.best_trial.params,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
        "objective": "reg:squarederror",
    }
    best_model = build_xgboost_model(best_params)
    best_model.fit(X_train, y_train)
    best_metrics = calculate_model_metrics(best_model, X_train, y_train, X_eval, y_eval)

    save_tuning_outputs(
        model=best_model,
        best_params=best_params,
        metrics=best_metrics,
        trials=trial_records,
        output_dir=output_dir,
    )

    if register_artifact:
        register_tuned_model_artifact(output_dir=output_dir)

    logger.info("XGBoost tuning completed. Best params: {}", best_params)

    return best_params, best_metrics
