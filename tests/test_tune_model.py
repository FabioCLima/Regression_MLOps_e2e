import json

import pandas as pd
import pytest

from src.config import model_config, tuning_config
from src.features.feature_engineering import feature_engineered_split_filename
from src.models.tune_model import (
    build_tuned_model_description,
    build_tuned_model_tags,
    load_best_model_name,
    register_tuned_model_artifact,
    save_tuning_outputs,
    tune_xgboost_model,
    validate_supported_tuning_model,
)


class FakeTrial:
    number = 0
    params = {
        "n_estimators": 10,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }

    def suggest_int(self, name, *_args):
        return int(self.params[name])

    def suggest_float(self, name, *_args, **_kwargs):
        return float(self.params[name])


class FakeStudy:
    def __init__(self):
        self.best_trial = FakeTrial()

    def optimize(self, objective, n_trials):
        for _ in range(n_trials):
            objective(self.best_trial)


class TinyModel:
    def fit(self, X, y):
        self.prediction_ = float(y.mean())
        return self

    def predict(self, X):
        return [self.prediction_] * len(X)


def make_feature_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [10.0, 20.0, 30.0, 40.0],
            "price": [100.0, 120.0, 140.0, 160.0],
        }
    )


def write_baseline_metrics(path, best_model="xgboost"):
    payload = {
        "best_model": best_model,
        "primary_metric": model_config.primary_metric,
        "metrics": {
            best_model: {
                "train": {"rmse": 1.0},
                "eval": {"rmse": 2.0},
                "gap": {"rmse": 1.0},
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_best_model_name_reads_training_metrics(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    write_baseline_metrics(metrics_path)

    assert load_best_model_name(metrics_path) == "xgboost"


def test_validate_supported_tuning_model_rejects_non_xgboost_model():
    with pytest.raises(ValueError, match="supports only"):
        validate_supported_tuning_model("random_forest")


def test_tuned_model_artifact_description_and_tags_are_informative():
    metrics_payload = {
        "base_model": "xgboost",
        "primary_metric": "rmse",
        "best_params": {"max_depth": 3},
        "metrics": {
            "train": {"rmse": 50.0},
            "eval": {"rmse": 70.0},
            "gap": {"rmse": 20.0},
        },
        "n_trials": 3,
    }

    description = build_tuned_model_description(metrics_payload)
    tags = build_tuned_model_tags(metrics_payload)

    assert "Fine-tuned XGBoost regression model" in description
    assert "rmse=70.0000" in description
    assert "tuned-model" in tags
    assert "xgboost" in tags
    assert "optuna" in tags
    assert "metric-rmse" in tags


def test_save_tuning_outputs_writes_model_metrics_and_trials(tmp_path):
    model = TinyModel().fit(pd.DataFrame({"a": [1, 2]}), pd.Series([1, 2]))
    metrics = {
        "train": {"rmse": 1.0, "mae": 1.0, "r2": 0.9},
        "eval": {"rmse": 2.0, "mae": 2.0, "r2": 0.8},
        "gap": {"rmse": 1.0, "mae": 1.0, "r2": -0.1},
    }

    save_tuning_outputs(
        model=model,
        best_params={"max_depth": 3},
        metrics=metrics,
        trials=[{"trial_number": 0, "rmse": 2.0}],
        output_dir=tmp_path,
    )

    assert (tmp_path / tuning_config.tuned_model_filename).exists()
    assert (tmp_path / tuning_config.tuning_metrics_filename).exists()
    assert (tmp_path / tuning_config.trials_filename).exists()


def test_register_tuned_model_artifact_requires_outputs(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing tuning output files"):
        register_tuned_model_artifact(output_dir=tmp_path)


def test_tune_xgboost_model_runs_lightweight_tuning(monkeypatch, tmp_path):
    input_dir = tmp_path / "features"
    output_dir = tmp_path / "models"
    input_dir.mkdir()
    dataset = make_feature_dataset()
    dataset.to_csv(input_dir / feature_engineered_split_filename("train"), index=False)
    dataset.to_csv(input_dir / feature_engineered_split_filename("eval"), index=False)
    baseline_metrics_path = tmp_path / "baseline_metrics.json"
    write_baseline_metrics(baseline_metrics_path)

    monkeypatch.setattr(
        "src.models.tune_model.optuna.create_study",
        lambda direction: FakeStudy(),
    )
    monkeypatch.setattr(
        "src.models.tune_model.build_xgboost_model",
        lambda params: TinyModel(),
    )

    best_params, metrics = tune_xgboost_model(
        input_dir=input_dir,
        baseline_metrics_path=baseline_metrics_path,
        output_dir=output_dir,
        n_trials=2,
        register_artifact=False,
    )

    assert best_params["max_depth"] == 3
    assert set(metrics) == {"train", "eval", "gap"}
    assert (output_dir / tuning_config.tuned_model_filename).exists()
    assert (output_dir / tuning_config.tuning_metrics_filename).exists()
