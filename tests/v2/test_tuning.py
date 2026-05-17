"""Mocked tests for the tuning component.

Tuning uses Hydra for config management. Tests verify that the Hydra config
is loaded correctly and that MLflow receives the resolved config as an artifact.
"""

import argparse
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from omegaconf import OmegaConf

from components.tuning.run import main as tuning_main


@pytest.fixture
def hydra_cfg():
    return OmegaConf.create({
        "tuning": {"n_trials": 5, "direction": "minimize"},
        "search_space": {
            "n_estimators": {"low": 100, "high": 300},
            "max_depth": {"low": 3, "high": 6},
            "learning_rate": {"low": 0.01, "high": 0.2, "log": True},
            "subsample": {"low": 0.6, "high": 1.0},
            "colsample_bytree": {"low": 0.6, "high": 1.0},
        },
    })


@pytest.fixture
def tuning_results():
    return {
        "best_params": {"n_estimators": 250, "max_depth": 5, "learning_rate": 0.05},
        "tuned_metrics": {"rmse": 16000.0, "mae": 11000.0, "r2": 0.93},
    }


@patch("components.tuning.run.mlflow")
@patch("components.tuning.run.wandb")
@patch("components.tuning.run.tune_model")
@patch("components.tuning.run.load_feature_splits")
@patch("components.tuning.run.os.listdir")
def test_go_logs_hydra_config_as_mlflow_artifact(
    mock_listdir, mock_load, mock_tune, mock_wandb, mock_mlflow,
    hydra_cfg, tuning_results
):
    mock_listdir.return_value = ["xgboost_tuned_model.pkl"]
    df = pd.DataFrame({"price": [300_000], "feature_1": [1.0]})
    mock_load.return_value = (df, df.copy())
    mock_tune.return_value = tuning_results

    mock_run = MagicMock()
    mock_run.use_artifact.return_value.download.return_value = "/tmp/fake"
    mock_wandb.init.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_wandb.init.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    fake_args = argparse.Namespace(
        input_artifact="house_ts_features:latest",
        input_model="house_ts_best_model:latest",
        output_artifact="house_ts_tuned_model",
        output_type="model",
        output_description="XGBoost model tuned with Optuna",
        pipeline_run_id="test-run-001",
    )

    with patch("components.tuning.run.argparse.ArgumentParser") as mock_parser_cls:
        mock_parser = MagicMock()
        mock_parser.parse_known_args.return_value = (fake_args, [])
        mock_parser_cls.return_value = mock_parser

        # Call the inner go logic directly (bypassing @hydra.main decorator)
        import components.tuning.run as tuning_module
        tuning_module._run_inner(hydra_cfg, fake_args)

    # The full Hydra config must be logged as a dict artifact
    mock_mlflow.log_dict.assert_called_once()
    logged_key = mock_mlflow.log_dict.call_args[0][1]
    assert logged_key == "hydra_config.yaml"


@patch("components.tuning.run.mlflow")
@patch("components.tuning.run.wandb")
@patch("components.tuning.run.tune_model")
@patch("components.tuning.run.load_feature_splits")
@patch("components.tuning.run.os.listdir")
def test_go_logs_n_trials_as_mlflow_param(
    mock_listdir, mock_load, mock_tune, mock_wandb, mock_mlflow,
    hydra_cfg, tuning_results
):
    mock_listdir.return_value = ["xgboost_tuned_model.pkl"]
    df = pd.DataFrame({"price": [300_000], "feature_1": [1.0]})
    mock_load.return_value = (df, df.copy())
    mock_tune.return_value = tuning_results

    mock_run = MagicMock()
    mock_run.use_artifact.return_value.download.return_value = "/tmp/fake"
    mock_wandb.init.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_wandb.init.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    fake_args = argparse.Namespace(
        input_artifact="house_ts_features:latest",
        input_model="house_ts_best_model:latest",
        output_artifact="house_ts_tuned_model",
        output_type="model",
        output_description="XGBoost model tuned with Optuna",
        pipeline_run_id="test-run-001",
    )

    import components.tuning.run as tuning_module
    tuning_module._run_inner(hydra_cfg, fake_args)

    logged_params = mock_mlflow.log_params.call_args[0][0]
    assert logged_params["n_trials"] == 5
    assert logged_params["direction"] == "minimize"
