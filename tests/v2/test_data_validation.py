"""Mocked tests for the data_validation component."""

import argparse
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from components.data_validation.run import _validate, go


@pytest.fixture
def args():
    return argparse.Namespace(
        input_artifact="house_ts_raw:latest",
        output_artifact="house_ts_validation",
        output_type="validation_report",
        output_description="Schema and quality validation report",
        pipeline_run_id="test-run-001",
    )


@pytest.fixture
def valid_df():
    return pd.DataFrame({
        "date": ["2019-01-01", "2020-06-01"],
        "city_full": ["boston", "miami"],
        "city": ["boston", "miami"],
        "metro_full": ["boston-metro", "miami-metro"],
        "zipcode": ["02101", "33101"],
        "price": [300_000, 450_000],
        "median_list_price": [310_000, 460_000],
    })


def test_validate_passes_on_valid_data(valid_df):
    result = _validate(valid_df)
    assert result["passed"] is True
    assert result["schema_valid"] is True
    assert result["total_nulls"] == 0
    assert result["price_out_of_range"] == 0


def test_validate_fails_on_missing_columns():
    df = pd.DataFrame({"date": ["2019-01-01"], "price": [300_000]})
    result = _validate(df)
    assert result["schema_valid"] is False
    assert result["passed"] is False
    assert len(result["missing_columns"]) > 0


def test_validate_detects_price_outliers(valid_df):
    valid_df.loc[0, "price"] = 100  # below MIN_PRICE
    result = _validate(valid_df)
    assert result["price_out_of_range"] == 1
    assert result["passed"] is False


def test_validate_counts_rows(valid_df):
    result = _validate(valid_df)
    assert result["total_rows"] == 2


@patch("components.data_validation.run.mlflow")
@patch("components.data_validation.run.wandb")
@patch("components.data_validation.run.os.listdir")
@patch("components.data_validation.run.pd.read_csv")
def test_go_logs_validation_metrics_to_mlflow(
    mock_read_csv, mock_listdir, mock_wandb, mock_mlflow, args, valid_df
):
    mock_read_csv.return_value = valid_df
    mock_listdir.return_value = ["HouseTS.csv"]

    mock_run = MagicMock()
    mock_run.use_artifact.return_value.download.return_value = "/tmp/fake"
    mock_wandb.init.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_wandb.init.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    go(args)

    mock_mlflow.log_metrics.assert_called_once()
    logged = mock_mlflow.log_metrics.call_args[0][0]
    assert "validation_passed" in logged
    assert "total_nulls" in logged
    assert logged["schema_valid"] == 1
