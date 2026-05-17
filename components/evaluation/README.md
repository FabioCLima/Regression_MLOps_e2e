# evaluation

Evaluates the tuned model on the holdout split. This is the **only** component
that touches the holdout set — it is deliberately isolated from training and tuning
to prevent any information leakage.

## What it does

1. Downloads `house_ts_tuned_model:latest` and `house_ts_features:latest` from W&B
2. Runs inference on the holdout split only
3. Computes RMSE, MAE, R² on holdout
4. Logs metrics to MLflow and saves an `evaluation_report.json` artifact

## MLflow metrics logged

| Metric | Meaning |
|---|---|
| `holdout_rmse` | Root mean squared error on holdout |
| `holdout_mae` | Mean absolute error on holdout |
| `holdout_r2` | R² coefficient on holdout |

## Files

| File | Role |
|---|---|
| `run.py` | Component logic + argparse entry point |
| `MLproject` | MLflow contract: parameters and command |
| `python_env.yaml` | Isolated virtualenv for this component |

## Running locally

```bash
python components/evaluation/run.py \
  --input_model house_ts_tuned_model:latest \
  --input_features house_ts_features:latest \
  --output_artifact house_ts_eval_report \
  --output_type evaluation_report \
  --output_description "Holdout evaluation metrics for the tuned model" \
  --pipeline_run_id local-test-001
```

## Running via MLflow

```bash
mlflow run components/evaluation \
  -P input_model=house_ts_tuned_model:latest \
  -P input_features=house_ts_features:latest \
  -P output_artifact=house_ts_eval_report \
  -P output_type=evaluation_report \
  -P output_description="Holdout evaluation metrics for the tuned model" \
  --env-manager virtualenv
```

## Environment variables required

```bash
export WANDB_API_KEY=<your-key>
export MLFLOW_TRACKING_URI=<uri>   # default: local ./mlruns
```
