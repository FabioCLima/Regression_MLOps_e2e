"""Mocked tests for the evaluation component."""

import argparse
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from components.evaluation.run import go


@pytest.fixture
def args():
    return argparse.Namespace(
        input_model="house_ts_tuned_model:latest",
        input_features="house_ts_features:latest",
        output_artifact="house_ts_eval_report",
        output_type="evaluation_report",
        output_description="Holdout evaluation metrics",
        pipeline_run_id="test-run-001",
    )


@pytest.fixture
def holdout_df():
    return pd.DataFrame({
        "price": [300_000, 350_000, 400_000],
        "feature_1": [1.0, 2.0, 3.0],
        "feature_2": [0.5, 0.6, 0.7],
    })


@patch("components.evaluation.run.mlflow")
@patch("components.evaluation.run.wandb")
@patch("components.evaluation.run.joblib.load")
@patch("components.evaluation.run.pd.read_csv")
@patch("components.evaluation.run.os.listdir")
@patch("components.evaluation.run.feature_engineered_split_filename")
def test_go_logs_holdout_metrics_to_mlflow(
    mock_filename, mock_listdir, mock_read_csv, mock_joblib,
    mock_wandb, mock_mlflow, args, holdout_df
):
    mock_filename.return_value = "feature_engineered_holdout.csv"
    mock_listdir.return_value = ["xgboost_tuned_model.pkl"]
    mock_read_csv.return_value = holdout_df

    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([310_000, 340_000, 410_000])
    mock_joblib.return_value = fake_model

    mock_run = MagicMock()
    mock_run.use_artifact.return_value.download.return_value = "/tmp/fake"
    mock_wandb.init.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_wandb.init.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    go(args)

    mock_mlflow.log_metrics.assert_called_once()
    logged = mock_mlflow.log_metrics.call_args[0][0]
    assert "holdout_rmse" in logged
    assert "holdout_mae" in logged
    assert "holdout_r2" in logged


@patch("components.evaluation.run.mlflow")
@patch("components.evaluation.run.wandb")
@patch("components.evaluation.run.joblib.load")
@patch("components.evaluation.run.pd.read_csv")
@patch("components.evaluation.run.os.listdir")
@patch("components.evaluation.run.feature_engineered_split_filename")
def test_go_sets_split_evaluated_param(
    mock_filename, mock_listdir, mock_read_csv, mock_joblib,
    mock_wandb, mock_mlflow, args, holdout_df
):
    mock_filename.return_value = "feature_engineered_holdout.csv"
    mock_listdir.return_value = ["xgboost_tuned_model.pkl"]
    mock_read_csv.return_value = holdout_df

    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([310_000, 340_000, 410_000])
    mock_joblib.return_value = fake_model

    mock_run = MagicMock()
    mock_run.use_artifact.return_value.download.return_value = "/tmp/fake"
    mock_wandb.init.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_wandb.init.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    go(args)

    logged_params = mock_mlflow.log_params.call_args[0][0]
    assert logged_params["split_evaluated"] == "holdout"
