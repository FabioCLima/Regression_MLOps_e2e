"""Mocked tests for the feature_engineering component."""

import argparse
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from components.feature_engineering.run import go


@pytest.fixture
def args():
    return argparse.Namespace(
        input_artifact="house_ts_cleaned:latest",
        output_artifact="house_ts_features",
        output_type="feature_dataset",
        output_description="Feature-engineered splits with fitted encoders",
        pipeline_run_id="test-run-001",
    )


@pytest.fixture
def engineered_splits():
    df = pd.DataFrame({
        "year": [2019, 2020],
        "quarter": [1, 2],
        "month": [1, 6],
        "zipcode_freq": [0.05, 0.03],
        "city_full_enc": [0.45, 0.32],
        "price": [300_000, 450_000],
    })
    return {"train": df, "eval": df.copy(), "holdout": df.copy()}


@patch("components.feature_engineering.run.mlflow")
@patch("components.feature_engineering.run.wandb")
@patch("components.feature_engineering.run.run_feature_engineering")
@patch("components.feature_engineering.run.os.listdir")
def test_go_calls_run_feature_engineering(
    mock_listdir, mock_feat_eng, mock_wandb, mock_mlflow,
    args, engineered_splits
):
    mock_listdir.return_value = []
    mock_feat_eng.return_value = engineered_splits

    mock_run = MagicMock()
    mock_run.use_artifact.return_value.download.return_value = "/tmp/fake"
    mock_wandb.init.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_wandb.init.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    go(args)

    mock_feat_eng.assert_called_once()
    call_kwargs = mock_feat_eng.call_args.kwargs
    assert call_kwargs["register_artifact"] is False


@patch("components.feature_engineering.run.mlflow")
@patch("components.feature_engineering.run.wandb")
@patch("components.feature_engineering.run.run_feature_engineering")
@patch("components.feature_engineering.run.os.listdir")
def test_go_logs_feature_count_as_mlflow_param(
    mock_listdir, mock_feat_eng, mock_wandb, mock_mlflow,
    args, engineered_splits
):
    mock_listdir.return_value = []
    mock_feat_eng.return_value = engineered_splits

    mock_run = MagicMock()
    mock_run.use_artifact.return_value.download.return_value = "/tmp/fake"
    mock_wandb.init.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_wandb.init.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    go(args)

    logged_params = mock_mlflow.log_params.call_args[0][0]
    assert logged_params["feature_count"] == 6  # columns in engineered_splits["train"]
    assert "frequency_encode_column" in logged_params
    assert "target_encode_column" in logged_params
