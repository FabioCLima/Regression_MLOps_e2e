"""Mocked tests for the registry component."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from components.registry.run import go


@pytest.fixture
def args():
    return argparse.Namespace(
        input_model="house_ts_tuned_model:latest",
        model_name="house-price-regressor",
        stage="Staging",
        pipeline_run_id="test-run-001",
    )


@patch("components.registry.run.mlflow")
@patch("components.registry.run.wandb")
@patch("components.registry.run.joblib.load")
@patch("components.registry.run.os.listdir")
def test_go_registers_model_with_correct_name(
    mock_listdir, mock_joblib, mock_wandb, mock_mlflow, args
):
    mock_listdir.return_value = ["xgboost_tuned_model.pkl"]
    mock_joblib.return_value = MagicMock()

    mock_run = MagicMock()
    mock_run.use_artifact.return_value.download.return_value = "/tmp/fake"
    mock_wandb.init.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_wandb.init.return_value.__exit__ = MagicMock(return_value=False)

    mock_active_run = MagicMock()
    mock_active_run.info.run_id = "fake-run-id"
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_active_run)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    mock_model_info = MagicMock()
    mock_model_info.model_uri = "runs:/fake-run-id/model"
    mock_mlflow.sklearn.log_model.return_value = mock_model_info

    mock_version = MagicMock()
    mock_version.version = "1"
    mock_client = MagicMock()
    mock_client.get_latest_versions.return_value = [mock_version]
    mock_mlflow.tracking.MlflowClient.return_value = mock_client

    go(args)

    mock_mlflow.sklearn.log_model.assert_called_once()
    call_kwargs = mock_mlflow.sklearn.log_model.call_args.kwargs
    assert call_kwargs["registered_model_name"] == "house-price-regressor"


@patch("components.registry.run.mlflow")
@patch("components.registry.run.wandb")
@patch("components.registry.run.joblib.load")
@patch("components.registry.run.os.listdir")
def test_go_transitions_model_to_staging(
    mock_listdir, mock_joblib, mock_wandb, mock_mlflow, args
):
    mock_listdir.return_value = ["xgboost_tuned_model.pkl"]
    mock_joblib.return_value = MagicMock()

    mock_run = MagicMock()
    mock_run.use_artifact.return_value.download.return_value = "/tmp/fake"
    mock_wandb.init.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_wandb.init.return_value.__exit__ = MagicMock(return_value=False)

    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    mock_model_info = MagicMock()
    mock_model_info.model_uri = "runs:/fake-run-id/model"
    mock_mlflow.sklearn.log_model.return_value = mock_model_info

    mock_version = MagicMock()
    mock_version.version = "1"
    mock_client = MagicMock()
    mock_client.get_latest_versions.return_value = [mock_version]
    mock_mlflow.tracking.MlflowClient.return_value = mock_client

    go(args)

    mock_client.transition_model_version_stage.assert_called_once_with(
        name="house-price-regressor",
        version="1",
        stage="Staging",
    )


@patch("components.registry.run.mlflow")
@patch("components.registry.run.wandb")
@patch("components.registry.run.joblib.load")
@patch("components.registry.run.os.listdir")
def test_go_sets_component_tag(
    mock_listdir, mock_joblib, mock_wandb, mock_mlflow, args
):
    mock_listdir.return_value = ["xgboost_tuned_model.pkl"]
    mock_joblib.return_value = MagicMock()

    mock_run = MagicMock()
    mock_run.use_artifact.return_value.download.return_value = "/tmp/fake"
    mock_wandb.init.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_wandb.init.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    mock_model_info = MagicMock()
    mock_model_info.model_uri = "runs:/fake-run-id/model"
    mock_mlflow.sklearn.log_model.return_value = mock_model_info
    mock_client = MagicMock()
    mock_client.get_latest_versions.return_value = [MagicMock(version="1")]
    mock_mlflow.tracking.MlflowClient.return_value = mock_client

    go(args)

    call_kwargs = mock_mlflow.start_run.call_args.kwargs
    assert call_kwargs["tags"]["component"] == "registry"
