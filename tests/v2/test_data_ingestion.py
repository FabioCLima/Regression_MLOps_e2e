"""Mocked tests for the data_ingestion component.

These tests verify the component's behaviour (what it calls and with what arguments)
without touching the filesystem, W&B, or MLflow. Domain logic in src/ is tested
separately in the existing v1 test suite.
"""

import argparse
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from components.data_ingestion.run import go


@pytest.fixture
def args():
    return argparse.Namespace(
        raw_data="data/raw_data/HouseTS.csv",
        reference_data="data/raw_data/usmetros.csv",
        output_artifact="house_ts_raw",
        output_type="raw_dataset",
        output_description="Raw house time series dataset",
        pipeline_run_id="test-run-001",
    )


@pytest.fixture
def fake_df():
    return pd.DataFrame({
        "date": ["2019-01-01", "2020-06-01"],
        "city_full": ["boston", "miami"],
        "price": [300_000, 450_000],
    })


@patch("components.data_ingestion.run.mlflow")
@patch("components.data_ingestion.run.wandb")
@patch("components.data_ingestion.run.pd.read_csv")
def test_go_initialises_wandb_and_mlflow(mock_read_csv, mock_wandb, mock_mlflow, args, fake_df):
    mock_read_csv.return_value = fake_df
    mock_wandb.init.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_wandb.init.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    go(args)

    mock_wandb.init.assert_called_once()
    mock_mlflow.start_run.assert_called_once()


@patch("components.data_ingestion.run.mlflow")
@patch("components.data_ingestion.run.wandb")
@patch("components.data_ingestion.run.pd.read_csv")
def test_go_sets_component_tag_in_mlflow(mock_read_csv, mock_wandb, mock_mlflow, args, fake_df):
    mock_read_csv.return_value = fake_df
    mock_wandb.init.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_wandb.init.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    go(args)

    call_kwargs = mock_mlflow.start_run.call_args.kwargs
    assert call_kwargs["tags"]["component"] == "data_ingestion"
    assert call_kwargs["tags"]["pipeline_run_id"] == "test-run-001"


@patch("components.data_ingestion.run.mlflow")
@patch("components.data_ingestion.run.wandb")
@patch("components.data_ingestion.run.pd.read_csv")
def test_go_logs_row_and_column_counts_as_params(mock_read_csv, mock_wandb, mock_mlflow, args, fake_df):
    mock_read_csv.return_value = fake_df
    mock_wandb.init.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_wandb.init.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    go(args)

    logged_params = mock_mlflow.log_params.call_args[0][0]
    assert logged_params["raw_rows"] == 2
    assert logged_params["raw_columns"] == 3


@patch("components.data_ingestion.run.mlflow")
@patch("components.data_ingestion.run.wandb")
@patch("components.data_ingestion.run.pd.read_csv")
def test_go_creates_wandb_artifact_with_correct_name(mock_read_csv, mock_wandb, mock_mlflow, args, fake_df):
    mock_read_csv.return_value = fake_df
    mock_run = MagicMock()
    mock_wandb.init.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_wandb.init.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    go(args)

    mock_wandb.Artifact.assert_called_once()
    artifact_kwargs = mock_wandb.Artifact.call_args.kwargs
    assert artifact_kwargs["name"] == "house_ts_raw"
    assert artifact_kwargs["type"] == "raw_dataset"
