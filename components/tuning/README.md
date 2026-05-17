# tuning

Hyperparameter optimization for XGBoost using Optuna. The search space is
defined in Hydra config files — swapping configs changes the experiment without
touching the code, and the full resolved config is logged to MLflow.

## What it does

1. Downloads `house_ts_features:latest` and `house_ts_best_model:latest` from W&B
2. Runs Optuna with `n_trials` trials using the configured search space
3. Logs the full resolved Hydra config to MLflow (makes the search space decision traceable)
4. Saves the tuned model and Optuna trial results as artifacts

## Hydra config structure

```
conf/
├── config.yaml           ← base config: n_trials, direction, default search space
└── xgboost/
    ├── default.yaml      ← standard search space (fast)
    └── extended.yaml     ← wider search space (more trials needed)
```

## Switching search space

```bash
# Standard (default)
python run.py ... xgboost=default

# Extended — set n_trials >= 50 in config.yaml first
python run.py ... xgboost=extended

# Override n_trials at runtime
python run.py ... tuning.n_trials=50 xgboost=extended
```

## Lineage answer: what search space was used?

```python
run = mlflow.get_run("<tuning_run_id>")
config = mlflow.artifacts.load_dict("<tuning_run_id>/hydra_config.yaml")
# → shows full resolved config: n_trials, direction, every search space bound
```

## Files

| File | Role |
|---|---|
| `run.py` | Component logic (argparse + Hydra entry point) |
| `MLproject` | MLflow contract: parameters and command |
| `python_env.yaml` | Isolated virtualenv for this component |
| `conf/config.yaml` | Base Hydra config |
| `conf/xgboost/default.yaml` | Standard search space |
| `conf/xgboost/extended.yaml` | Extended search space |

## Running locally

```bash
python components/tuning/run.py \
  --input_artifact house_ts_features:latest \
  --input_model house_ts_best_model:latest \
  --output_artifact house_ts_tuned_model \
  --output_type model \
  --output_description "XGBoost model tuned with Optuna" \
  --pipeline_run_id local-test-001
```

## Running via MLflow

```bash
mlflow run components/tuning \
  -P input_artifact=house_ts_features:latest \
  -P input_model=house_ts_best_model:latest \
  -P output_artifact=house_ts_tuned_model \
  -P output_type=model \
  -P output_description="XGBoost model tuned with Optuna" \
  -P hydra_config=xgboost/extended \
  --env-manager virtualenv
```

## Environment variables required

```bash
export WANDB_API_KEY=<your-key>
export MLFLOW_TRACKING_URI=<uri>   # default: local ./mlruns
```
