"""Mocked tests for the training component."""

import argparse
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from components.training.run import go


@pytest.fixture
def args():
    return argparse.Namespace(
        input_artifact="house_ts_features:latest",
        output_artifact="house_ts_best_model",
        output_type="model",
        output_description="Best model from baseline comparison",
        pipeline_run_id="test-run-001",
    )


@pytest.fixture
def training_results():
    return {
        "best_model_name": "xgboost",
        "metrics": {
            "dummy": {"rmse": 95000.0, "mae": 80000.0, "r2": 0.10},
            "linear_regression": {"rmse": 45000.0, "mae": 32000.0, "r2": 0.70},
            "ridge": {"rmse": 44500.0, "mae": 31500.0, "r2": 0.71},
            "random_forest": {"rmse": 22000.0, "mae": 15000.0, "r2": 0.88},
            "xgboost": {"rmse": 18500.0, "mae": 12000.0, "r2": 0.91},
        },
    }


@patch("components.training.run.mlflow")
@patch("components.training.run.wandb")
@patch("components.training.run.train_and_compare_models")
@patch("components.training.run.load_feature_splits")
@patch("components.training.run.os.listdir")
def test_go_logs_metrics_for_all_candidates(
    mock_listdir, mock_load, mock_train, mock_wandb, mock_mlflow,
    args, training_results
):
    mock_listdir.return_value = []
    df = pd.DataFrame({"price": [300_000], "feature_1": [1.0]})
    mock_load.return_value = (df, df.copy())
    mock_train.return_value = training_results

    mock_run = MagicMock()
    mock_run.use_artifact.return_value.download.return_value = "/tmp/fake"
    mock_wandb.init.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_wandb.init.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    go(args)

    # log_metrics is called once per candidate model
    assert mock_mlflow.log_metrics.call_count == 5


@patch("components.training.run.mlflow")
@patch("components.training.run.wandb")
@patch("components.training.run.train_and_compare_models")
@patch("components.training.run.load_feature_splits")
@patch("components.training.run.os.listdir")
def test_go_tags_best_model_in_mlflow(
    mock_listdir, mock_load, mock_train, mock_wandb, mock_mlflow,
    args, training_results
):
    mock_listdir.return_value = []
    df = pd.DataFrame({"price": [300_000], "feature_1": [1.0]})
    mock_load.return_value = (df, df.copy())
    mock_train.return_value = training_results

    mock_run = MagicMock()
    mock_run.use_artifact.return_value.download.return_value = "/tmp/fake"
    mock_wandb.init.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_wandb.init.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    go(args)

    call_kwargs = mock_mlflow.start_run.call_args.kwargs
    assert call_kwargs["tags"]["best_model"] == "xgboost"
    assert call_kwargs["tags"]["component"] == "training"
