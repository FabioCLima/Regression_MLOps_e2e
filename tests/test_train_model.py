import json

import pandas as pd
import pytest

from src.config import model_config
from src.features.feature_engineering import feature_engineered_split_filename
from src.models.train_model import (
    build_best_model_description,
    build_best_model_tags,
    build_model,
    build_train_eval_metrics,
    evaluate_regression,
    load_feature_splits,
    maybe_sample_dataset,
    model_filename,
    register_training_artifacts,
    run_model_training,
    select_best_model,
    split_features_target,
)


def make_training_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2020, 2020, 2021, 2021],
            "quarter": [1, 2, 1, 2],
            "month": [1, 4, 1, 4],
            "zipcode_freq": [10, 20, 10, 20],
            "city_full_encoded": [200000, 220000, 240000, 260000],
            "price": [210000, 230000, 250000, 270000],
        }
    )


def test_split_features_target_requires_target_column():
    dataset = pd.DataFrame({"feature": [1, 2]})

    with pytest.raises(ValueError, match="target column"):
        split_features_target(dataset)


def test_split_features_target_rejects_non_numeric_features():
    dataset = pd.DataFrame({"feature": ["a", "b"], "price": [1, 2]})

    with pytest.raises(ValueError, match="Non-numeric"):
        split_features_target(dataset)


def test_maybe_sample_dataset_ignores_invalid_sample_frac():
    dataset = make_training_dataset()

    result = maybe_sample_dataset(dataset, sample_frac=1.5)

    pd.testing.assert_frame_equal(result, dataset)
    assert result is not dataset


def test_build_model_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unsupported model_name"):
        build_model("not_a_model")


def test_evaluate_regression_returns_expected_metric_names():
    metrics = evaluate_regression(
        y_true=pd.Series([1.0, 2.0, 3.0]),
        y_pred=pd.Series([1.0, 2.5, 2.5]),
    )

    assert set(metrics) == {"mae", "rmse", "r2"}


def test_build_train_eval_metrics_calculates_generalization_gap():
    metrics = build_train_eval_metrics(
        train_metrics={"mae": 10.0, "rmse": 20.0, "r2": 0.95},
        eval_metrics={"mae": 15.0, "rmse": 35.0, "r2": 0.90},
    )

    assert metrics["train"]["rmse"] == 20.0
    assert metrics["eval"]["rmse"] == 35.0
    assert metrics["gap"]["mae"] == 5.0
    assert metrics["gap"]["rmse"] == 15.0
    assert metrics["gap"]["r2"] == pytest.approx(-0.05)


def test_select_best_model_uses_lowest_primary_metric():
    metrics_by_model = {
        "dummy": {"eval": {"rmse": 10.0}},
        "ridge": {"eval": {"rmse": 5.0}},
    }

    assert select_best_model(metrics_by_model) == "ridge"


def test_select_best_model_requires_metrics():
    with pytest.raises(ValueError, match="cannot be empty"):
        select_best_model({})


def test_best_model_artifact_description_and_tags_are_informative():
    metrics_payload = {
        "best_model": "xgboost",
        "primary_metric": "rmse",
        "metrics": {
            "dummy": {
                "train": {"rmse": 250000.0},
                "eval": {"rmse": 300000.0},
                "gap": {"rmse": 50000.0},
            },
            "xgboost": {
                "train": {"rmse": 50000.0},
                "eval": {"rmse": 70000.0},
                "gap": {"rmse": 20000.0},
            },
        },
    }

    description = build_best_model_description(metrics_payload)
    tags = build_best_model_tags(metrics_payload)

    assert "Best baseline regression model" in description
    assert "Selected model: xgboost" in description
    assert "rmse=70000.0000" in description
    assert "best-model" in tags
    assert "housing-price-prediction" in tags
    assert "selected-xgboost" in tags
    assert "metric-rmse" in tags


def test_load_feature_splits_requires_train_and_eval_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing feature split files"):
        load_feature_splits(tmp_path)


def test_run_model_training_saves_models_and_metrics(tmp_path):
    input_dir = tmp_path / "features"
    output_dir = tmp_path / "models"
    input_dir.mkdir()
    dataset = make_training_dataset()
    dataset.to_csv(input_dir / feature_engineered_split_filename("train"), index=False)
    dataset.to_csv(input_dir / feature_engineered_split_filename("eval"), index=False)

    _, metrics_by_model, best_model_name = run_model_training(
        input_dir=input_dir,
        output_dir=output_dir,
        candidate_models=("dummy", "linear_regression", "ridge"),
        register_artifact=False,
    )

    assert best_model_name in metrics_by_model
    assert (output_dir / model_config.metrics_filename).exists()
    assert (output_dir / model_config.best_model_filename).exists()
    assert not (output_dir / model_filename("dummy")).exists()
    assert not (output_dir / model_filename("linear_regression")).exists()
    assert not (output_dir / model_filename("ridge")).exists()

    with (output_dir / model_config.metrics_filename).open(encoding="utf-8") as f:
        metrics_payload = json.load(f)

    assert metrics_payload["best_model"] == best_model_name
    assert set(metrics_payload["metrics"]) == {"dummy", "linear_regression", "ridge"}
    assert set(metrics_payload["metrics"][best_model_name]) == {"train", "eval", "gap"}
    assert model_config.primary_metric in metrics_payload["metrics"][best_model_name]["eval"]


def test_register_training_artifacts_requires_outputs(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing training output files"):
        register_training_artifacts(output_dir=tmp_path)
