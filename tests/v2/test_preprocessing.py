"""Mocked tests for the preprocessing component."""

import argparse
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from components.preprocessing.run import go


@pytest.fixture
def args():
    return argparse.Namespace(
        input_artifact="house_ts_raw:latest",
        output_artifact="house_ts_cleaned",
        output_type="cleaned_dataset",
        output_description="Temporally split and cleaned dataset",
        eval_start_date="2020-01-01",
        holdout_start_date="2022-01-01",
        pipeline_run_id="test-run-001",
    )


@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "date": ["2019-01-01", "2021-01-01", "2023-01-01"],
        "city_full": ["boston", "miami", "denver"],
        "metro_full": ["boston-metro", "miami-metro", "denver-metro"],
        "price": [300_000, 450_000, 500_000],
        "median_list_price": [310_000, 460_000, 510_000],
        "zipcode": ["02101", "33101", "80201"],
    })


@patch("components.preprocessing.run.mlflow")
@patch("components.preprocessing.run.wandb")
@patch("components.preprocessing.run.save_data_splits")
@patch("components.preprocessing.run.preprocess_dataset")
@patch("components.preprocessing.run.split_dataset_by_date")
@patch("components.preprocessing.run.os.listdir")
@patch("components.preprocessing.run.pd.read_csv")
def test_go_calls_split_and_preprocess(
    mock_read_csv, mock_listdir, mock_split, mock_preprocess,
    mock_save, mock_wandb, mock_mlflow, args, raw_df
):
    mock_read_csv.return_value = raw_df
    mock_listdir.return_value = ["HouseTS.csv"]

    train = raw_df.iloc[:1]
    eval_ = raw_df.iloc[1:2]
    holdout = raw_df.iloc[2:]
    mock_split.return_value = (train, eval_, holdout)
    mock_preprocess.return_value = raw_df.iloc[:1]

    mock_run = MagicMock()
    mock_run.use_artifact.return_value.download.return_value = "/tmp/fake"
    mock_wandb.init.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_wandb.init.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    go(args)

    mock_split.assert_called_once_with(
        dataset=raw_df,
        eval_start_date="2020-01-01",
        holdout_start_date="2022-01-01",
    )
    assert mock_preprocess.call_count == 3  # train, eval, holdout


@patch("components.preprocessing.run.mlflow")
@patch("components.preprocessing.run.wandb")
@patch("components.preprocessing.run.save_data_splits")
@patch("components.preprocessing.run.preprocess_dataset")
@patch("components.preprocessing.run.split_dataset_by_date")
@patch("components.preprocessing.run.os.listdir")
@patch("components.preprocessing.run.pd.read_csv")
def test_go_logs_split_dates_as_mlflow_params(
    mock_read_csv, mock_listdir, mock_split, mock_preprocess,
    mock_save, mock_wandb, mock_mlflow, args, raw_df
):
    mock_read_csv.return_value = raw_df
    mock_listdir.return_value = ["HouseTS.csv"]
    train = raw_df.iloc[:1]
    eval_ = raw_df.iloc[1:2]
    holdout = raw_df.iloc[2:]
    mock_split.return_value = (train, eval_, holdout)
    mock_preprocess.return_value = train

    mock_run = MagicMock()
    mock_run.use_artifact.return_value.download.return_value = "/tmp/fake"
    mock_wandb.init.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_wandb.init.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    go(args)

    logged_params = mock_mlflow.log_params.call_args[0][0]
    assert logged_params["eval_start_date"] == "2020-01-01"
    assert logged_params["holdout_start_date"] == "2022-01-01"
