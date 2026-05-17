# training

Trains all candidate models and selects the best one by RMSE on the eval split.
Every candidate's metrics are logged to MLflow, not just the winner.

## What it does

1. Downloads `house_ts_features:latest` from W&B
2. Trains: Dummy, LinearRegression, Ridge, RandomForest, XGBoost
3. Evaluates all on the eval split using RMSE as primary metric
4. Saves the best model as a `.pkl` artifact
5. Logs per-model metrics to MLflow + W&B metadata

## MLflow metrics logged (per candidate)

Pattern: `{model_name}_eval_{metric}` — e.g. `xgboost_eval_rmse`, `random_forest_eval_r2`

This means you can always look back and see why XGBoost was chosen over Random Forest
on a specific run — the full comparison is preserved.

## Files

| File | Role |
|---|---|
| `run.py` | Component logic + argparse entry point |
| `MLproject` | MLflow contract: parameters and command |
| `python_env.yaml` | Isolated virtualenv for this component |

## Running locally

```bash
python components/training/run.py \
  --input_artifact house_ts_features:latest \
  --output_artifact house_ts_best_model \
  --output_type model \
  --output_description "Best model from baseline candidate comparison" \
  --pipeline_run_id local-test-001
```

## Running via MLflow

```bash
mlflow run components/training \
  -P input_artifact=house_ts_features:latest \
  -P output_artifact=house_ts_best_model \
  -P output_type=model \
  -P output_description="Best model from baseline candidate comparison" \
  --env-manager virtualenv
```

## Environment variables required

```bash
export WANDB_API_KEY=<your-key>
export MLFLOW_TRACKING_URI=<uri>   # default: local ./mlruns
```
